"""
Evidence-bounded repair R1/R2 (paper VII-E).

Invariant 3 (CLAUDE.md): repair must re-verify, never bypass. The independent
post-hoc audit in eval/run_repair_eval.py is what backs that claim, so the
tests below assert on the AUDIT, not on the verifier's own self-report.
"""


def test_probe_set_has_not_shrunk(repair):
    assert repair["n_probes"] == 27


def test_audit_passes_for_every_emitted_answer_in_both_modes(repair):
    # The load-bearing assertion. Every answer the system emits -- with repair
    # on or off -- must independently re-derive as citation-clean.
    for mode in ("strip-only", "strip+repair"):
        assert repair["audit_pass"][mode] == repair["n_probes"], (
            f"{mode}: an emitted answer failed the independent audit")


def test_repair_introduces_no_invariant_violations(repair):
    assert repair["invariant_violations"] == 0, (
        "repair traded precision for coverage -- this breaks the paper's "
        "central claim, not just a metric")


def test_repair_actually_fires(repair):
    # Class-B guard: R1 was once measured at zero coverage gain purely because
    # no probe exercised it. A repair eval where nothing is repaired proves
    # nothing about repair.
    assert repair["repairs"]["strip-only"] == 0
    assert repair["repairs"]["strip+repair"] == 9
    rules = {rule for rule, _ in repair["repairs_performed"]}
    assert rules == {"R1", "R2"}, f"a repair rule went unexercised: {rules}"


def test_repair_improves_coverage(repair):
    assert repair["coverage_gain_faulty"] >= 0.16
    assert repair["coverage_gain_all"] > 0
    assert (repair["coverage_all"]["strip+repair"]
            > repair["coverage_all"]["strip-only"])


def test_fabricated_content_is_never_repaired(repair):
    # Repairing a fabricated provision or quotation would mean inventing text,
    # which is the failure the system exists to prevent.
    for pid in ("f2-a", "f2-b", "f2-c", "q-f4-a", "q-f4-b", "q-f4-c"):
        assert pid in repair["unrepairable"]
