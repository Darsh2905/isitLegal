"""BM25-only retrieval baseline -- the ablation anchor (Phase 1).

`stem=True` (the default) applies Snowball stemming (retrieval/text.py);
`stem=False` reproduces the original unstemmed baseline, kept as its own
ablation arm so the stemming gain is reported rather than silently folded in.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rank_bm25 import BM25Okapi
from ingest.metadata_db import MetadataDB
from retrieval.text import tokenize


class BM25Retriever:
    def __init__(self, db: MetadataDB, stem: bool = True):
        self.stem = stem
        self.provs = db.all_provisions()
        self.docs = [f"{p.act} {p.section} {p.marginal_note} {p.text}"
                     for p in self.provs]
        self.bm25 = BM25Okapi([tokenize(d, stem) for d in self.docs])

    def search(self, query: str, k: int = 10):
        scores = self.bm25.get_scores(tokenize(query, self.stem))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.provs[i].key, float(scores[i])) for i in order]
