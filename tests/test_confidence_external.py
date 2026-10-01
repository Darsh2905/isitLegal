"""
Out-of-sample check: the deployed real-corpus confidence gate against eight
genuinely externally-sourced unanswerable queries (real page titles from
law4u.in, not authored by this project -- see
eval/external_unanswerable_qa.py). Addresses the "author-written, not
adversarially sourced" half of Limitation Seven that a larger corpus is not
needed to fix.

The result is a real, measured degradation (0.750 correct abstention here
against the perfect 1.000 on the author-written test split), reported
honestly rather than smoothed over -- a suspiciously perfect score on a
genuinely external set would be less credible, not more.
"""
import pytest


def test_eight_external_queries_checked(confidence_external):
    assert confidence_external["n"] == 8


def test_gate_is_not_retuned_to_these_queries(confidence_external):
    # The whole point: this must be the SAME tau the real-corpus gate
    # deploys with, not a threshold fit to this set. Tolerance covers only
    # the two scripts' different rounding precision (4 vs 3 decimals),
    # not a real drift -- a genuine threshold change should still fail this.
    from eval.run_confidence_real_eval import report as real_report
    assert confidence_external["tau"] == pytest.approx(real_report()["tau"], abs=0.001)


def test_external_generalisation_is_real_but_imperfect(confidence_external):
    # Two of eight are wrongly answered -- both explainable, genuine
    # near-misses (see run() below), not a design failure. Asserting the
    # exact count guards against silently reporting a different number than
    # what was actually measured.
    assert confidence_external["wrongly_answered"] == 2
    assert confidence_external["correct_abstention"] == pytest.approx(0.75, abs=0.001)


def test_recovery_suit_query_collides_with_a_real_adjacent_provision(confidence_external):
    # "What is the limitation period for filing a recovery suit in India?"
    # scores high because BNSS§514 genuinely discusses limitation periods
    # (for criminal cognizance, not the civil Limitation Act the question
    # means) -- a topically adjacent collision, the same class of hard case
    # V5's support screen documents, not random noise.
    rows = {r["query"]: r for r in confidence_external["rows"]}
    q = "What is the limitation period for filing a recovery suit in India?"
    assert rows[q]["answerable"] is True


def test_income_tax_act_query_collides_with_the_information_technology_act(confidence_external):
    # "Section 56(2) of the Income Tax Act" -- the corpus's "ITA" alias
    # means Information Technology Act, not Income Tax Act. Generic
    # procedural vocabulary ("provisions", "apply") plus this act-name
    # collision score just high enough to slip through, a genuinely
    # different failure mode than the topically-adjacent case above.
    rows = {r["query"]: r for r in confidence_external["rows"]}
    q = ("How Do the Provisions of Section 56(2) of the Income Tax Act Apply "
         "to Gifts?")
    assert rows[q]["answerable"] is True


def test_the_other_six_external_queries_correctly_abstain(confidence_external):
    rows = confidence_external["rows"]
    correct = [r for r in rows if not r["answerable"]]
    assert len(correct) == 6
