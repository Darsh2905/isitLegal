"""
Dense (embedding) retrieval (contribution: Stage 2 of the architecture,
paper VII promise of an InLegalBERT-vs-general-embedding ablation).

Two arms, both loaded via HuggingFace `transformers` with manual mean-pooling
over the last hidden state (not a probabilistic judgement -- an embedding is a
fixed function of the input, deterministic given fixed model weights, so this
sits in retrieval like BM25 does, not inside verify/):

  domain  -- law-ai/InLegalBERT, pre-trained further on Indian legal text.
  general -- sentence-transformers/all-MiniLM-L6-v2, a strong general-purpose
             control with no legal-domain adaptation.

No FAISS or approximate search: the corpus is at most a few thousand
provisions, so exact cosine similarity over a dense numpy matrix is both
simpler and, at this scale, not meaningfully slower.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import MetadataDB

MODEL_IDS = {
    "domain": "law-ai/InLegalBERT",
    "general": "sentence-transformers/all-MiniLM-L6-v2",
}


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class DenseEncoder:
    """Thin wrapper: any HF encoder model -> mean-pooled, L2-normalised vectors."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=256, return_tensors="pt")
            out = self.model(**enc)
            pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs.append(pooled.numpy())
        return np.concatenate(vecs, axis=0) if vecs else np.zeros((0, 0))


# PASSAGE WINDOWS. encode() truncates at 256 tokens, and 678 of the real
# corpus's 3,263 provisions (21%) are longer than that -- median 127 tokens,
# 90th percentile 391, longest 3,331 (CrPC s.167) -- so everything past the
# first ~200 words of those provisions was invisible to dense retrieval. With
# passages=True each provision is split into overlapping word windows (each
# prefixed with the act, section and marginal note so no window loses its
# identity) and a provision scores as its best-matching window (max-pooling).
# passages=False reproduces the original whole-provision vectors.
_WINDOW_WORDS = 150
_STRIDE_WORDS = 100


def _passages(p) -> List[str]:
    head = f"{p.act} {p.section} {p.marginal_note}"
    words = (p.text or "").split()
    if len(words) <= _WINDOW_WORDS:
        return [f"{head} {' '.join(words)}"]
    out = []
    for start in range(0, len(words), _STRIDE_WORDS):
        out.append(f"{head} {' '.join(words[start:start + _WINDOW_WORDS])}")
        if start + _WINDOW_WORDS >= len(words):
            break
    return out


class DenseRetriever:
    """Embedding-based retriever over a MetadataDB's provisions."""

    def __init__(self, db: MetadataDB, encoder: DenseEncoder, passages: bool = True):
        self.provs = db.all_provisions()
        self.encoder = encoder
        self.passages = passages
        if passages:
            docs, owner = [], []
            for i, p in enumerate(self.provs):
                for w in _passages(p):
                    docs.append(w)
                    owner.append(i)
            self.owner = np.array(owner)
        else:
            docs = [f"{p.act} {p.section} {p.marginal_note} {p.text}" for p in self.provs]
            self.owner = np.arange(len(docs))
        self.doc_vecs = encoder.encode(docs)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        if len(self.provs) == 0:
            return []
        qvec = self.encoder.encode([query])[0]
        window_scores = self.doc_vecs @ qvec
        # Max-pool window scores back to their provision.
        scores = np.full(len(self.provs), -np.inf)
        np.maximum.at(scores, self.owner, window_scores)
        order = np.argsort(-scores)[:k]
        return [(self.provs[i].key, float(scores[i])) for i in order]


_ENCODER_CACHE = {}


def get_encoder(arm: str) -> DenseEncoder:
    """Cached: loading a transformer model is expensive, tests share one."""
    if arm not in _ENCODER_CACHE:
        _ENCODER_CACHE[arm] = DenseEncoder(MODEL_IDS[arm])
    return _ENCODER_CACHE[arm]
