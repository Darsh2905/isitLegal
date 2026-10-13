"""
Live pre/post-verifier comparison (eval/run_live_generation_eval.py) --
a real, independently invoked model (Groq, openai/gpt-oss-20b), not this
project's own model. See eval/live_generation_answers.py for the full
methodological disclosure: real generation, still a project-authored query
set.

This suite makes no live API calls. Only the raw completion text is frozen
(eval/live_generation_answers.py); the gate decision, retrieval, and
verification are recomputed fresh by eval/run_live_generation_eval.py, and
this file asserts on that recomputation -- consistent with every other
eval/ script's tests, and with CLAUDE.md invariant 6 (reproducible without a
paid credential).
"""
import pytest


def test_gate_decision_reproduces_the_captured_run(live_generation):
    # The reproducibility check inside run_live_generation_eval.py's own
    # report() already asserts this on every query (it would raise, not
    # just fail an assertion, if the recomputed gate decision drifted from
    # what was actually captured) -- reaching this point at all is the test.
    assert live_generation["gate_n"] == 16


def test_gate_coverage_on_the_full_gold_set(live_generation):
    # Deliberately NOT the held-out test-split coverage (0.875) -- this is
    # the deployed gate applied to every gold query, dev-split included,
    # which is what a real user of the system actually experiences and is
    # measurably lower.
    assert live_generation["gate_pass"] == 10
    assert live_generation["gate_abstain"] == 6
    assert live_generation["gate_coverage_full_gold"] == pytest.approx(0.625, abs=0.001)


def test_gate_never_wrongly_answers_an_unanswerable_query(live_generation):
    assert live_generation["unanswerable_wrongly_answered"] == 0
    assert live_generation["unanswerable_n"] == 16


def test_ten_of_sixteen_gold_queries_were_actually_generated(live_generation):
    assert live_generation["n_generated"] == 10


def test_pre_verifier_resolution_accuracy(live_generation):
    # The independently invoked model made exactly ONE citation error that
    # does not resolve: "section 118 of the Bharatiya Samvidhan Adhiniyam" --
    # a real section attributed to an act that does not exist (the
    # Bharatiya SAKSHYA Adhiniyam). Earlier revisions reported 0 here
    # because citations to unrecognised act names were invisible to the
    # extractor (see ingest/normalize.extract_unresolved_act_refs); the
    # error was always in the output, the evaluation could not see it.
    #
    # The total rose from 29 to 49 in the fail-closed revision (2026-10)
    # without the answers changing: the audit now also counts bare sections
    # ("Section 41", 8 of them), the gpt-oss lenticular markers ("\u30104\u3011")
    # that V1 used to ignore, and "Const \u00a722", an abbreviation the alias
    # table lacked. Every one of the 20 added citations resolves; the one
    # unresolvable citation is still the Samvidhan Adhiniyam.
    assert live_generation["pre_bad"] == 1
    assert live_generation["pre_total"] == 49
    assert live_generation["pre_accuracy"] == pytest.approx(48 / 49)


def test_post_verifier_resolution_accuracy_is_the_core_guarantee(live_generation):
    assert live_generation["post_bad"] == 0
    assert live_generation["post_accuracy"] == 1.0


def test_coverage_cost_is_real_and_not_smoothed_away(live_generation):
    # Mean coverage is NOT 1.000 -- V4 still strips three quotations. It rose
    # from 0.930 once V4 stopped rejecting faithful elided quotations, the
    # segmentation fix stopped merging several claims into one strip unit,
    # and sub-section pinpoints stopped failing V2. Asserting the exact
    # figure, not just "< 1.0", is what would catch a regression that
    # silently changed how much verification costs.
    assert live_generation["mean_coverage"] == pytest.approx(0.972, abs=0.005)
    assert live_generation["mean_coverage"] < 1.0


def test_quotation_findings_are_real_misquotes_not_faithful_elisions(live_generation):
    # Earlier revisions reported "V4 caught eight real quotation slips".
    # Seven of those eight were quotations whose every segment was verbatim
    # and in order, with the omission marked by an ellipsis -- standard legal
    # quotation practice. With ellipsis-aware checking, five are accepted
    # (the omitted words change nothing: "for the time being", "to such
    # person", a definitional gloss) and three are still stripped: one
    # genuine misquotation and two elisions that omit a legal EXCEPTION
    # (CrPC s.57's "in the absence of a special order of a Magistrate",
    # Art. 22's "excluding the time necessary for the journey").
    by_type = live_generation["findings_by_type"]
    assert by_type.get("unsupported_quote") == 3
    assert live_generation["elided_quotes_accepted"] == 5
    # The one existence-level error the model made (a mis-named act).
    assert by_type.get("unknown_act") == 1
    assert by_type.get("nonexistent_provision", 0) == 0
    assert by_type.get("unresolved_marker", 0) == 0


def test_repealed_citations_are_annotated_not_stripped(live_generation):
    # V3 by design (CLAUDE.md invariant 2): repealed citations are annotated,
    # not stripped. With the full correspondence tables loaded, every one of
    # them now carries a recorded successor (this was 2 with only eleven
    # hand-tracked mappings).
    assert live_generation["findings_by_type"].get("repealed_provision", 0) >= 1
    assert live_generation["post_unmapped_repealed"] == 0
