"""
Reciprocal Rank Fusion of BM25 + dense retrieval (Stage 2, paper VII promise).

RRF combines two RANKED LISTS without needing their scores to be on a
comparable scale -- exactly the problem that made the confidence gate need its
own IDF normalisation for BM25 alone (retrieval/confidence.py). BM25 scores and
cosine similarities live in different, unnormalised ranges; RRF sidesteps that
by using each ranker's ORDINAL rank rather than its raw score:

    RRF(d) = sum over rankers r of  1 / (k_rrf + rank_r(d))

A document missing from one ranker's top-K contributes nothing from that
ranker rather than a penalty score, so a document strong in one signal and
absent from the other is not unfairly punished. k_rrf=60 is the constant from
the original Cormack et al. (2009) formulation and the value most retrieval
literature reports results at; we do not re-tune it here for the same reason
theta and tau_support were not picked by eyeballing a single easy set --
this is a bibliographic choice, made explicit and cited, not a hidden default.
"""

from typing import Dict, List, Tuple

K_RRF = 60


def reciprocal_rank_fusion(rankings: List[List[Tuple[str, float]]],
                           k_rrf: int = K_RRF) -> List[Tuple[str, float]]:
    """
    `rankings`: one ranked (key, score) list per retriever. Scores are used
    only to break ties within a ranker's own list (i.e. not used at all,
    since ordering already reflects them) -- only rank position feeds RRF.

    Returns a single fused ranking, highest RRF score first.
    """
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, (key, _score) in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


class FusedRetriever:
    """Wraps any number of (name, retriever) pairs behind one .search(query, k)."""

    def __init__(self, retrievers: List[Tuple[str, object]], k_rrf: int = K_RRF,
                candidate_k: int = 50):
        self.retrievers = retrievers
        self.k_rrf = k_rrf
        self.candidate_k = candidate_k

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        rankings = [r.search(query, self.candidate_k) for _name, r in self.retrievers]
        fused = reciprocal_rank_fusion(rankings, self.k_rrf)
        return fused[:k]
