"""
The pre/post-verifier hallucination comparison (Tier 2, item 10).

See eval/generation_answers.py for the full methodological disclosure: these
are real, unscripted answers composed by this project's own generating model,
not an independently invoked API call. These tests assert on the numbers that
run produced -- they do not re-validate that the recorded answer TEXT is
"good" (there is no oracle for that here beyond the deterministic checks
themselves), only that the experiment's own logic and the core guarantee hold.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.generation_answers import UNGROUNDED_ANSWERS, GROUNDED_ANSWERS
from eval.gold_qa import GOLD


@pytest.fixture(scope="module")
def generation():
    import importlib
    return importlib.import_module("eval.run_generation_eval").report()


def test_both_conditions_cover_every_gold_query():
    qids = {qid for qid, *_ in GOLD}
    assert set(UNGROUNDED_ANSWERS) == qids
    assert set(GROUNDED_ANSWERS) == qids


def test_post_verifier_resolution_accuracy_is_unity(generation):
    # THE core guarantee (V1/V2/V4), measured on genuine generated text for
    # the first time in this project rather than only on synthetic probes.
    # This is the load-bearing assertion in this test module.
    for cond in ("ungrounded", "grounded"):
        r = generation[cond]
        assert r["post_accuracy"] == 1.0, (
            f"{cond}: post-verifier resolution accuracy was not 1.0 -- "
            f"this would mean an unresolved marker or a nonexistent "
            f"provision survived verification, which breaks the paper's "
            f"central claim")
        assert r["post_bad"] == 0


def test_repealed_without_mapping_is_reported_not_hidden(generation):
    # V3 annotates repealed citations with no recorded successor rather than
    # stripping them (CLAUDE.md invariant 2/4). This is a real, separate
    # limitation -- of this project's mapping coverage, not of the verifier
    # -- and must stay visible as its own number rather than folding into
    # "bad" (which conflates two different guarantees; see
    # eval/run_generation_eval.py's _citation_stats docstring for the
    # earlier version of this script that made exactly that mistake).
    # This was 3 (IPC 500-502 in the recorded defamation answer) while only
    # eleven hand-tracked mappings were stored; the full correspondence
    # tables map all three. The mechanism itself is still checked directly:
    # a repealed section with no row in the tables (IPC 34) is annotated
    # "no mapping recorded" and kept, never stripped.
    ungrounded_unmapped = generation["ungrounded"]["post_unmapped_repealed"]
    assert ungrounded_unmapped == 0
    from ingest.india_code import build_real_db
    from verify.verifier import CitationVerifier, Chunk
    db = build_real_db()
    out, rep = CitationVerifier(db).verify(
        "Common intention is covered by Section 34 of the IPC [1].",
        [Chunk(1, "IPC§34", db.get("IPC§34").text, True)])
    assert rep.sentences_stripped == 0
    assert any("no mapping recorded" in a for a in rep.annotations)


def test_repair_actually_fires_on_real_generated_text(generation):
    # Every recorded UNGROUNDED answer cites at least one now-repealed Act by
    # its old name (by design -- these are real generated answers about
    # recodified law). R2 should repoint each one to current law with a
    # transparent annotation. A verifier that never repaired anything here
    # would mean the repair path was never actually exercised on genuine
    # text, only on hand-authored probes.
    rows = generation["ungrounded"]["rows"]
    repaired_queries = sum(1 for r in rows if r["findings"] > 0)
    # 9 of 16: the other 7 cite only already-current-law Acts (BNS, BNSS,
    # BSA, the Constitution, NIA, ICA, ITA) with nothing to repair.
    assert repaired_queries == 9, (
        f"{repaired_queries}/16 ungrounded answers triggered a finding; "
        f"expected exactly 9 for the recorded answers")


def test_a_real_quotation_error_was_caught(generation):
    # A genuine, unplanned finding from composing real text: the grounded
    # q12 (defamation) answer's quotation did not match its source closely
    # enough (91.7% < the 95% threshold) and was correctly stripped, dropping
    # that query's coverage. Not a synthetic probe -- an actual mistake made
    # while writing a real answer, caught by V4 exactly as designed.
    q12 = next(r for r in generation["grounded"]["rows"] if r["qid"] == "q12")
    assert q12["coverage"] < 1.0
    assert q12["findings"] > 0


def test_mean_coverage_is_high_but_not_universally_perfect(generation):
    # Coverage should not be trivially 1.0 everywhere (that would suggest the
    # experiment never exercised any check) nor collapsed (that would mean
    # the generated answers were mostly unusable).
    g = generation["grounded"]["mean_coverage"]
    assert 0.85 <= g < 1.0
