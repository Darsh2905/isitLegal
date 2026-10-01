"""
Support screen V5 (paper VII-D).

V5 is ADVISORY ONLY and must never strip (CLAUDE.md invariant 4). Its measured
failure boundary on topically-adjacent misattribution is a reported FINDING,
so these tests assert the boundary still exists rather than assuming it moved.
"""


def test_easy_set_is_cleanly_separable(support):
    assert support["margin"] >= 0.50
    assert support["unsupported"]["max"] < support["supported"]["min"]


def test_easy_set_perfect_at_deployed_threshold(support):
    at20 = support["calibration"][0.2]
    assert at20["precision"] == 1.0 and at20["recall"] == 1.0


def test_adjacent_provision_boundary_still_overlaps(support_hard):
    # THE DOCUMENTED FAILURE BOUNDARY. Margin is -0.09: correct and
    # topically-adjacent-incorrect score distributions overlap, so no threshold
    # separates them. If this assertion ever fails, V5 has genuinely improved
    # and paper Sections VII-D and VIII must be rewritten before the test is
    # relaxed -- do not simply loosen the bound.
    assert support_hard["hard"]["margin"] < 0.0, (
        "the adjacent-provision boundary appears to have moved; re-run the "
        "stress test and update the paper before changing this assertion")


def test_deployed_operating_point_never_flags_a_correct_claim(support_hard):
    # Invariant 4: tau is chosen as the highest threshold keeping FLAGGING
    # PRECISION at 1.000, trading recall for the guarantee that no correct
    # claim is ever flagged.
    dep = support_hard["deployed"]
    assert dep["tau"] == 0.20
    assert dep["fp"] == 0
    assert dep["precision"] == 1.0


def test_deployed_recall_is_the_reported_number(support_hard):
    dep = support_hard["deployed"]
    assert dep["tp"] == 9 and dep["fn"] == 2
    assert abs(dep["recall"] - 0.818) < 0.001
