"""
Confidence gate / calibrated abstention (C3) -- recalibrated on the real
3,263-provision corpus (eval/run_confidence_real_eval.py), not the toy one.

WHY THIS FILE EXISTS SEPARATELY FROM test_confidence.py. The deployed gate
(retrieval/confidence.py, tau=0.305) was calibrated only against the toy
corpus; applying that threshold to the real corpus's different score
distribution produces a false positive -- confirmed here as a standing
regression test, not just a one-off finding. This file establishes a second,
independently-tuned threshold for the real corpus rather than assuming the
toy-corpus number transfers.
"""
import pytest


def test_splits_are_disjoint_and_balanced(confidence_real):
    n = confidence_real["n"]
    assert n["ans_dev"] == n["ans_test"] == 8
    assert n["un_dev"] == n["un_test"] == 8


def test_corpus_is_the_real_one_not_the_toy_one(confidence_real):
    assert confidence_real["corpus_size"] > 3000


def test_toy_corpus_threshold_reproduces_a_false_positive(confidence_real):
    # The motivating bug, pinned as a regression test: tau=0.305 (the
    # deployed toy-corpus threshold) answers a query the real-corpus gate
    # must refuse. If this ever starts failing, the real corpus's score
    # distribution has drifted enough that the toy threshold would no longer
    # even reproduce the original finding -- worth knowing either way.
    assert confidence_real["agri_confidence"] >= 0.305


def test_recalibrated_gate_fixes_the_false_positive(confidence_real):
    assert confidence_real["agri_fixed"] is True
    assert confidence_real["agri_confidence"] < confidence_real["tau"]


def test_gate_never_answers_an_unanswerable_test_query(confidence_real):
    # The error that matters most: answering a question the real corpus
    # cannot support. Zero tolerance, same as the toy-corpus gate.
    assert confidence_real["test"]["wrongly_answered"] == 0
    assert confidence_real["test"]["correct_abstention"] == 1.0


def test_coverage_cost_is_real_and_reported_honestly(confidence_real):
    # Unlike the toy corpus (coverage 1.000 on test), the real corpus's
    # larger, lexically competitive vocabulary costs one held-out answerable
    # query. Asserting the EXACT numbers, not just "coverage > 0", is what
    # would catch a regression that trades this away silently in either
    # direction -- CLAUDE.md invariant 5: report coverage alongside
    # precision, always, never just the flattering number.
    assert confidence_real["test"]["wrongly_abstained"] == 1
    assert confidence_real["test"]["coverage"] == pytest.approx(0.875, abs=0.001)


def test_balanced_accuracy_clears_the_toy_corpus_test_result(confidence_real):
    # Not a cherry-pick: the real corpus is harder (more lexically
    # competitive vocabulary across 3,263 provisions vs 35), so this is not
    # assumed -- it is checked, and reported as a genuine finding either way.
    assert confidence_real["test"]["balanced_accuracy"] >= 0.90


def test_threshold_was_tuned_on_dev_not_test(confidence_real):
    res = confidence_real["signals"][confidence_real["chosen"]]
    lo, hi = res["interval"]
    assert lo <= confidence_real["tau"] <= hi


def test_dev_selection_would_again_have_chosen_a_worse_signal(confidence_real):
    # The same methodological lesson as the toy corpus, re-checked rather
    # than assumed to hold: dev-optimal signal selection picks a signal that
    # then generalises worse than the deployed one.
    sig = confidence_real["signals"]
    assert confidence_real["dev_pick"] != confidence_real["chosen"]
    assert (sig[confidence_real["chosen"]]["test"]["balanced_accuracy"]
            > sig[confidence_real["dev_pick"]]["test"]["balanced_accuracy"])


def test_real_and_toy_thresholds_are_not_assumed_equal(confidence_real):
    # The whole point of this file: tau* on the real corpus is its own,
    # separately-tuned number, not a copy of the toy corpus's 0.305.
    assert confidence_real["tau"] != pytest.approx(0.305, abs=0.01)
