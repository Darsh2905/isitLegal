"""
Dense retrieval, RRF fusion, and reranking (Tier 2 items 7-8).

These tests use the small toy corpus (35 provisions) for speed and use the
general-purpose model only -- InLegalBERT takes ~2 minutes to embed even the
real corpus and is exercised by eval/run_dense_eval.py instead, not by every
test run. What's asserted here is behavioural correctness (the mechanisms do
what they're specified to do), not the retrieval-quality numbers, which live
in the real-corpus eval scripts where they are meaningful.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import build_seed_db
from retrieval.bm25_baseline import BM25Retriever
from retrieval.dense import DenseEncoder, DenseRetriever, MODEL_IDS
from retrieval.fusion import reciprocal_rank_fusion, FusedRetriever, K_RRF


@pytest.fixture(scope="module")
def db():
    return build_seed_db()


@pytest.fixture(scope="module")
def general_encoder():
    return DenseEncoder(MODEL_IDS["general"])


@pytest.fixture(scope="module")
def dense_retriever(db, general_encoder):
    return DenseRetriever(db, general_encoder)


# --------------------------------------------------------------------------
# Dense retrieval
# --------------------------------------------------------------------------

def test_encoder_produces_unit_vectors(general_encoder):
    import numpy as np
    vecs = general_encoder.encode(["a short legal question", "another one entirely"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_encoder_is_deterministic(general_encoder):
    a = general_encoder.encode(["What is the punishment for murder?"])
    b = general_encoder.encode(["What is the punishment for murder?"])
    import numpy as np
    assert np.allclose(a, b)


def test_dense_retriever_finds_the_obviously_relevant_provision(dense_retriever):
    # Not a claim about retrieval QUALITY (that's the real-corpus eval's job)
    # -- just that the mechanism retrieves something plausible for an easy,
    # lexically-aligned query on the small corpus.
    res = [k for k, _ in dense_retriever.search("What is the punishment for murder?", 5)]
    assert "IPC§302" in res or "BNS§103" in res


def test_dense_retriever_covers_the_whole_corpus(db, dense_retriever):
    # With passage windows a long provision owns several vectors, so the
    # invariant is that every provision owns at least one, and every vector
    # belongs to a provision.
    assert set(dense_retriever.owner.tolist()) == set(range(db.count()))
    assert dense_retriever.doc_vecs.shape[0] == len(dense_retriever.owner)


def test_dense_retriever_k_is_respected(dense_retriever):
    assert len(dense_retriever.search("cheating", 3)) == 3
    assert len(dense_retriever.search("cheating", 100)) <= 35  # corpus size


# --------------------------------------------------------------------------
# RRF fusion
# --------------------------------------------------------------------------

def test_rrf_formula_matches_hand_computation():
    # Two rankers, disjoint top document each.
    ranking_a = [("X", 9.0), ("Y", 5.0)]
    ranking_b = [("Y", 0.9), ("Z", 0.8)]
    fused = dict(reciprocal_rank_fusion([ranking_a, ranking_b], k_rrf=60))
    assert fused["X"] == pytest.approx(1 / 61)
    assert fused["Y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["Z"] == pytest.approx(1 / 62)


def test_rrf_ranks_documents_agreed_on_by_both_rankers_highest():
    ranking_a = [("agree", 1.0), ("only_a", 0.5)]
    ranking_b = [("agree", 1.0), ("only_b", 0.5)]
    fused = reciprocal_rank_fusion([ranking_a, ranking_b])
    assert fused[0][0] == "agree"


def test_rrf_handles_a_document_missing_from_one_ranker():
    # A document absent from a ranker's list contributes nothing from that
    # ranker rather than raising or being penalised with a sentinel score.
    ranking_a = [("only_in_a", 1.0)]
    ranking_b = [("something_else", 1.0)]
    fused = dict(reciprocal_rank_fusion([ranking_a, ranking_b]))
    assert fused["only_in_a"] == pytest.approx(1 / (K_RRF + 1))


def test_fused_retriever_respects_k(db, dense_retriever):
    bm25 = BM25Retriever(db)
    fused = FusedRetriever([("bm25", bm25), ("dense", dense_retriever)])
    assert len(fused.search("cheating", 5)) == 5


def test_fused_retriever_is_a_real_fusion_not_just_bm25(db, dense_retriever):
    # If fusion silently degenerated to one ranker, this project would have
    # exactly the Class-B blind-spot bug its own history warns about: a
    # component that looks like it works because the other arm was never
    # exercised. Confirm both rankers' top document appear somewhere in a
    # generous top-k fused result for at least one query where they disagree.
    bm25 = BM25Retriever(db)
    fused = FusedRetriever([("bm25", bm25), ("dense", dense_retriever)], candidate_k=20)
    q = "Can the police arrest me without a warrant?"
    bm25_top = bm25.search(q, 1)[0][0]
    dense_top = dense_retriever.search(q, 1)[0][0]
    fused_keys = [k for k, _ in fused.search(q, 20)]
    assert bm25_top in fused_keys
    assert dense_top in fused_keys


# --------------------------------------------------------------------------
# Cross-encoder reranking
# --------------------------------------------------------------------------

def test_reranker_reorders_candidates(db):
    from retrieval.rerank import Reranker
    reranker = Reranker(db)
    # A deliberately bad initial order: the relevant provision ranked last.
    candidates = [("CONST§14", 0.1), ("NIA§138", 0.05), ("IPC§302", 0.01)]
    out = reranker.rerank("What is the punishment for murder?", candidates, top_k=3)
    keys = [k for k, _ in out]
    assert keys[0] == "IPC§302", (
        "the cross-encoder should surface the genuinely relevant provision "
        "first regardless of the input order")


def test_reranker_respects_top_k(db):
    from retrieval.rerank import Reranker
    reranker = Reranker(db)
    candidates = [(p.key, 0.0) for p in db.all_provisions()[:10]]
    out = reranker.rerank("cheating", candidates, top_k=3)
    assert len(out) == 3


def test_reranker_drops_candidates_not_in_the_db(db):
    from retrieval.rerank import Reranker
    reranker = Reranker(db)
    out = reranker.rerank("cheating", [("NOTREAL§999", 0.0)], top_k=5)
    assert out == []
