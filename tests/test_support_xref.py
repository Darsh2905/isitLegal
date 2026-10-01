"""
V5 x citation graph: support-via-reference on non-self-contained provisions.

The integration was specified as a binary rescue -- suppress the flag when a
one-hop reference supports the claim. Measuring the stress set BEFORE
implementing showed that costs flagging precision, because the edge that
rescues a correct citation also rescues a topically-adjacent wrong one. It is
therefore implemented as a THIRD verdict that annotates rather than suppresses.
"""
from verify.support import SupportScreen


def test_probe_set_shape(support_xref):
    c = support_xref["xref"]["plain"]
    assert c["tp"] + c["fp"] + c["tn"] + c["fn"] == 11


def test_v5_alone_false_alarms_on_this_class(support_xref):
    # The motivating defect, asserted so it cannot silently disappear: on
    # correct citations to non-self-contained provisions, V5 by itself flags
    # every one of them. Its precision here is 0.500, not the 1.000 measured
    # on the probes authored for it.
    plain = support_xref["xref"]["plain"]
    assert plain["fp"] == 5
    assert plain["precision"] == 0.5


def test_graph_removes_false_alarms(support_xref):
    plain, linked = support_xref["xref"]["plain"], support_xref["xref"]["linked"]
    assert support_xref["false_alarms_removed"] == 3
    assert linked["fp"] < plain["fp"]
    assert linked["precision"] > plain["precision"]


def test_no_precision_loss_on_the_sets_v5_was_tuned_against(support_xref):
    # Invariant 4's deployment property: no correct claim is ever hard-flagged.
    # The graph must not break it on the existing probe sets.
    assert support_xref["existing"]["linked"]["precision"] == 1.0
    assert support_xref["existing"]["linked"]["fp"] == 0


def test_recall_cost_is_measured_not_hidden(support_xref):
    # One wrong attribution (h-bad-3) moves out of the hard-flag bucket. It is
    # NOT silently passed -- it becomes a support-via-reference annotation --
    # but the recall cost is real and is asserted here so it stays visible.
    ep = support_xref["existing"]["plain"]
    el = support_xref["existing"]["linked"]
    assert ep["tp"] == 9 and el["tp"] == 8
    moved = support_xref["moved"]
    assert [m[0] for m in moved] == ["h-bad-3"]


def test_via_reference_never_strips(support_xref):
    """The new verdict is advisory, exactly as WEAK_SUPPORT is."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingest.metadata_db import build_seed_db
    from retrieval.xref import build_graph
    from verify.verifier import CitationVerifier, Chunk, Violation
    from eval.verifier_probes import SUPPORT_XREF

    db = build_seed_db()
    ver = CitationVerifier(db, support=True, graph=build_graph(db))
    seen = False
    for pid, claim, key, bad, note in SUPPORT_XREF:
        p = db.get(key)
        # Attribute by marker, as a generator would: V5 screens provisions
        # named textually OR resolved through a marker, and these probe claims
        # carry no textual citation.
        # The marker must sit INSIDE the sentence: appended after the full
        # stop it becomes a sentence of its own, which carries no substantive
        # terms and so trips the minimum-evidence abstention instead.
        attributed = claim.rstrip(".") + " [1]."
        out, rep = ver.verify(attributed, [Chunk(1, key, p.text, p.verbatim)])
        if rep.by_type(Violation.SUPPORT_VIA_REFERENCE):
            seen = True
            assert rep.coverage == 1.0, f"{pid}: via-reference stripped a claim"
            assert rep.sentences_stripped == 0, pid
    assert seen, "no probe produced a via-reference finding; invariant untested"


def test_verdict_is_three_valued(support_xref):
    assert {SupportScreen.SUPPORTED, SupportScreen.VIA_REFERENCE,
            SupportScreen.WEAK} == {"supported", "via_reference", "weak"}
