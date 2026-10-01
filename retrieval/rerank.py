"""
Cross-encoder reranking (Stage 3, paper VII promise).

A cross-encoder scores a (query, document) PAIR jointly through one forward
pass, rather than comparing two independently-computed vectors -- strictly
more expressive than the bi-encoder dense retrieval in retrieval/dense.py, and
correspondingly more expensive, which is why it runs only over a small
candidate set (top-20 by default) rather than the whole corpus.

The paper commits to reporting BOTH MRR and nDCG for reranking, following
Rustamov et al.'s finding that reranking can improve one while degrading the
other -- see eval/run_rerank_eval.py, which reports both rather than
whichever looks better.
"""

import os
import sys
from typing import List, Tuple

from sentence_transformers import CrossEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import MetadataDB
from retrieval.dense import _passages

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_MODEL_CACHE = {}


def get_reranker() -> CrossEncoder:
    if "model" not in _MODEL_CACHE:
        _MODEL_CACHE["model"] = CrossEncoder(MODEL_ID)
    return _MODEL_CACHE["model"]


class Reranker:
    """
    passages=True scores each candidate by its best-scoring passage window
    (the same 150-word windows, stride 100, as retrieval/dense.py) instead of
    one pair truncated at the cross-encoder's 512-token limit. Measured on the
    real corpus over the fused top-20 (eval/run_rerank_eval.py): whole-
    provision reranking HURT once first-stage retrieval improved (MRR@10
    0.683 -> 0.626), and passage reranking reverses that (-> 0.821). The
    window sizes were fixed for dense retrieval before this measurement and
    reused unchanged, not tuned on these queries. passages=False keeps the
    original arm.
    """

    def __init__(self, db: MetadataDB, model: CrossEncoder = None,
                 passages: bool = True):
        self.by_key = {p.key: p for p in db.all_provisions()}
        self.model = model or get_reranker()
        self.passages = passages

    def rerank(self, query: str, candidates: List[Tuple[str, float]],
              top_k: int = 5) -> List[Tuple[str, float]]:
        """
        `candidates`: a (key, upstream_score) ranked list (e.g. from RRF
        fusion). Returns the same keys re-scored and re-ordered by the
        cross-encoder, truncated to top_k.
        """
        if not candidates:
            return []
        pairs, owner, keys = [], [], []
        for key, _score in candidates:
            prov = self.by_key.get(key)
            if prov is None:
                continue
            keys.append(key)
            docs = (_passages(prov) if self.passages else
                    [f"{prov.act} {prov.section} {prov.marginal_note} {prov.text}"])
            for doc in docs:
                pairs.append((query, doc))
                owner.append(len(keys) - 1)
        if not pairs:
            return []
        window_scores = self.model.predict(pairs)
        scores = [float("-inf")] * len(keys)
        for o, sc in zip(owner, window_scores):
            scores[o] = max(scores[o], float(sc))
        order = sorted(range(len(keys)), key=lambda i: -scores[i])[:top_k]
        return [(keys[i], scores[i]) for i in order]
