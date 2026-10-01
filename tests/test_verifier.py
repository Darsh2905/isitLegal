"""Verifier fault detection (paper VII-B). Claimed: 13/13 faults, 0 false alarms."""


def test_probe_set_has_not_shrunk(verifier):
    assert verifier["n"] == 17


def test_detects_every_planted_fault(verifier):
    assert verifier["fn"] == 0
    assert verifier["tp"] == 13
    assert verifier["recall"] == 1.0


def test_no_false_alarms_on_clean_probes(verifier):
    assert verifier["fp"] == 0
    assert verifier["tn"] == 4
    assert verifier["precision"] == 1.0


def test_clean_content_is_never_stripped(verifier):
    # Invariant 5: precision 1.0 is trivial if everything is stripped. A clean
    # answer must survive verification untouched.
    assert verifier["mean_coverage_clean"] == 1.0


def test_faulty_coverage_is_not_degenerate(verifier):
    # Invariant 5 again, from the other side: the verifier must retain a
    # meaningful fraction of faulty answers rather than deleting them wholesale.
    assert verifier["mean_coverage_faulty"] >= 0.40, (
        "coverage on faulty answers collapsed; a verifier that strips "
        "everything achieves precision 1.0 trivially and is useless")


def test_worked_example_surfaces_all_three_fault_classes(verifier):
    kinds = set(verifier["worked_example_findings"])
    assert kinds == {"repealed_provision", "unresolved_marker",
                     "nonexistent_provision"}
