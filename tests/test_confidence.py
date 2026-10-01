"""
Confidence gate / calibrated abstention (contribution C3).

This is the one component that is NOT part of the deterministic guarantee: a
tuned threshold on a retrieval score is a statistical decision. The tests
therefore assert the METHODOLOGY (tuned on dev, reported on held-out test) as
much as the numbers, because the numbers come from 8 queries per class per
split and no difference of one or two queries among them is meaningful.
"""
import pytest

from retrieval.confidence import ConfidenceGate, ABSTENTION_MESSAGE


def test_splits_are_disjoint_and_balanced(confidence):
    n = confidence["n"]
    assert n["ans_dev"] == n["ans_test"] == 8
    assert n["un_dev"] == n["un_test"] == 8


def test_dev_and_test_query_sets_do_not_overlap():
    from eval.run_confidence_eval import (ANSWERABLE_DEV, ANSWERABLE_TEST)
    from eval.unanswerable_qa import UNANSWERABLE_DEV, UNANSWERABLE_TEST
    assert not (set(ANSWERABLE_DEV) & set(ANSWERABLE_TEST))
    assert not (set(UNANSWERABLE_DEV) & set(UNANSWERABLE_TEST))


def test_gate_abstains_on_every_unanswerable_test_query(confidence):
    # The error that matters: answering a question the corpus cannot support.
    assert confidence["test"]["wrongly_answered"] == 0
    assert confidence["test"]["correct_abstention"] == 1.0


def test_gate_does_not_refuse_answerable_queries(confidence):
    # Invariant 5's logic applied to abstention: a gate that refuses everything
    # achieves perfect abstention trivially and is useless.
    assert confidence["test"]["coverage"] == 1.0
    assert confidence["test"]["wrongly_abstained"] == 0


def test_threshold_was_tuned_on_dev_not_test(confidence):
    # tau* must lie in the interval that was optimal on DEV.
    res = confidence["signals"][confidence["chosen"]]
    lo, hi = res["interval"]
    assert lo <= confidence["tau"] <= hi


def test_dev_selection_would_have_chosen_a_worse_signal(confidence):
    """
    The reportable methodological finding. Choosing the confidence signal by
    development-set performance picks raw BM25 top-1, which generalises worse
    than the IDF-normalised signal deployed on a prior argument. With 8 queries
    per class this is an illustration, not a significant result -- but it is
    the reason the signal is justified by an argument rather than by a
    small-sample comparison.
    """
    sig = confidence["signals"]
    assert confidence["dev_pick"] == "raw top-1"
    assert confidence["chosen"] == "normalised (ours)"
    assert (sig["raw top-1"]["dev"]["balanced_accuracy"]
            >= sig["normalised (ours)"]["dev"]["balanced_accuracy"])
    assert (sig["normalised (ours)"]["test"]["balanced_accuracy"]
            > sig["raw top-1"]["test"]["balanced_accuracy"])


def test_gate_is_deterministic():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingest.metadata_db import build_seed_db
    from retrieval.bm25_baseline import BM25Retriever
    db = build_seed_db()
    g1 = ConfidenceGate(db, BM25Retriever(db, stem=False))
    g2 = ConfidenceGate(build_seed_db(), BM25Retriever(build_seed_db(), stem=False))
    for q in ("What is the punishment for murder?", "How is agricultural income taxed?"):
        assert g1.confidence(q).score == pytest.approx(g2.confidence(q).score)


def test_abstention_message_does_not_pretend_to_answer():
    assert "Insufficient authority" in ABSTENTION_MESSAGE
