"""
Calibrated abstention: the confidence gate (contribution C3).

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The deterministic verifier guarantees that emitted citations resolve. It has
nothing to say about a query whose answer is not in the corpus at all: given
such a query the retriever still returns its top-k, the generator still writes
an answer over them, and every citation in that answer may verify perfectly
while the answer itself is worthless. Verification cannot detect this, because
nothing about the emitted text is malformed.

The gate addresses it by declining to answer. Crucially, it is a THRESHOLD ON A
RETRIEVAL SCORE, tuned on held-out data -- a statistical decision, not a
deterministic one. It therefore lives in retrieval/ rather than verify/, and it
must be described in the paper as a weaker kind of guarantee than the
verification layer provides. It can be wrong in both directions: it can abstain
on an answerable query (costing coverage) and answer an unanswerable one
(costing exactly what it exists to prevent).

THE SIGNAL
----------
A raw BM25 score is not comparable across queries: it scales with how many
substantive terms the query has and how rare they are. A short query of common
words scores low even when perfectly answered; a long query of rare words scores
high even when nothing matches well. We therefore normalise the top-ranked
score by the total inverse document frequency the query puts on the table:

    confidence(q) = bm25_top1(q) / sum of IDF over q's substantive terms

which asks what fraction of the retrievable evidence the query demanded was
actually found. Raw top-1 score, the top-1/top-2 margin, and query-term coverage
were all measured first and none separates the two classes; the normalised score
is the only one of the four that does. That comparison is reported rather than
omitted, because the other three are the obvious choices.
"""

import math
import re
import sys
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import MetadataDB

_WORD = re.compile(r"[a-z]{3,}")

# Interrogative and structural vocabulary. These appear in nearly every query
# and carry no retrieval demand, so including them would inflate the
# denominator for question-shaped queries specifically.
_STOP = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "the", "and", "for", "are", "can", "does", "did", "was", "were", "will",
    "would", "should", "could", "must", "have", "has", "had", "any", "all",
    "under", "with", "from", "into", "about", "that", "this", "there", "their",
    "his", "her", "its", "you", "your", "our", "law", "laws", "legal", "india",
    "indian", "act", "section", "sections", "code", "person", "someone",
}


@dataclass
class Confidence:
    score: float
    top_key: Optional[str]
    top_raw: float
    idf_mass: float
    terms: List[str]

    def explain(self) -> str:
        return (f"confidence={self.score:.3f} "
                f"(top={self.top_key or 'none'} raw={self.top_raw:.2f} "
                f"idf_mass={self.idf_mass:.2f}, {len(self.terms)} terms)")


class ConfidenceGate:
    """
    Abstention gate over a first-stage retriever.

    tau is calibrated on a development split and must be reported on a
    held-out split (see eval/run_confidence_eval.py). A threshold tuned and
    reported on the same queries measures nothing.
    """

    def __init__(self, db: MetadataDB, retriever, tau: float = 0.25):
        self.db = db
        self.retriever = retriever
        self.tau = tau
        self.idf = self._build_idf()

    def _build_idf(self) -> Dict[str, float]:
        provs = self.db.all_provisions()
        n = max(len(provs), 1)
        df = Counter()
        for p in provs:
            df.update(set(self._terms(f"{p.marginal_note} {p.text or ''}")))
        return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    def _idf(self, term: str) -> float:
        # A term absent from the whole corpus is maximally discriminative, so
        # it contributes the maximum demand -- which is what makes an
        # out-of-corpus query score low rather than high.
        n = max(self.db.count(), 1)
        return self.idf.get(term, math.log(n + 1) + 1.0)

    @staticmethod
    def _terms(text: str) -> List[str]:
        return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP]

    def confidence(self, query: str) -> Confidence:
        terms = sorted(set(self._terms(query)))
        results = self.retriever.search(query, 1)
        top_key, top_raw = (results[0] if results else (None, 0.0))
        idf_mass = sum(self._idf(t) for t in terms)
        score = (top_raw / idf_mass) if idf_mass > 0 else 0.0
        return Confidence(score=score, top_key=top_key, top_raw=float(top_raw),
                          idf_mass=idf_mass, terms=terms)

    def should_answer(self, query: str) -> bool:
        return self.confidence(query).score >= self.tau

    def gate(self, query: str):
        """
        Returns (answerable, Confidence). When answerable is False the caller
        must emit the insufficient-authority response rather than generating
        over whatever the retriever returned.
        """
        c = self.confidence(query)
        return c.score >= self.tau, c


ABSTENTION_MESSAGE = (
    "Insufficient authority: the retrieved material does not cover this "
    "question closely enough to answer it from the corpus."
)
