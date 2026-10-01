"""BM25 retrieval over judgment chunks, parametrised by chunking strategy
(see ingest/judgment_chunks.py) -- mirrors retrieval/bm25_baseline.py's
shape for the statutory corpus, kept separate rather than merged into it so
the well-tested statute-only retrieval path is never touched by this."""
import re

from rank_bm25 import BM25Okapi

from ingest.judgment_chunks import JudgmentChunk, naive_chunks, parent_document_chunks
from ingest.scrape_judgments import Judgment

_TOK = re.compile(r"[a-z0-9]+")


class JudgmentRetriever:
    def __init__(self, judgments: list[Judgment], strategy: str = "parent_doc"):
        if strategy not in ("naive", "parent_doc"):
            raise ValueError(f"unknown strategy: {strategy!r}")
        self.strategy = strategy
        self.judgments_by_id = {j.doc_id: j for j in judgments}
        self.chunks: list[JudgmentChunk] = []
        for j in judgments:
            fn = naive_chunks if strategy == "naive" else parent_document_chunks
            self.chunks.extend(fn(j))
        self._tokenised = [_TOK.findall(c.text.lower()) for c in self.chunks]
        self.bm25 = BM25Okapi(self._tokenised)

    def search(self, query: str, k: int = 10):
        scores = self.bm25.get_scores(_TOK.findall(query.lower()))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def context_for(self, chunk: JudgmentChunk, window: int = 2) -> str:
        """What a generator would actually receive for this matched chunk:
        the bare chunk text under naive chunking (there is no separate
        notion of context to expand into), or the chunk expanded into its
        surrounding paragraphs under parent-document chunking."""
        if self.strategy == "naive":
            return chunk.text
        judgment = self.judgments_by_id[chunk.doc_id]
        return chunk.parent_context(judgment, window=window)
