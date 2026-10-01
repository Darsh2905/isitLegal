"""
Support screening on NON-SELF-CONTAINED provisions (contributions C1c x C5).

Reports the ablation that motivated the integration: V5 alone, versus V5 with
the statutory citation graph supplying support-via-reference. Both the recall
gain and the precision cost are reported, because the same graph edge that
rescues a correct citation also rescues a topically-adjacent wrong one.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import build_seed_db
from retrieval.xref import build_graph
from verify.support import SupportScreen
from eval.verifier_probes import SUPPORT_PROBES, SUPPORT_HARD, SUPPORT_XREF

TAU = 0.20


def _counts(rows):
    """rows: (pid, is_wrong, hard_flagged)."""
    tp = sum(1 for _, bad, f in rows if f and bad)
    fp = sum(1 for _, bad, f in rows if f and not bad)
    fn = sum(1 for _, bad, f in rows if not f and bad)
    tn = sum(1 for _, bad, f in rows if not f and not bad)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": p, "recall": r}


def report():
    db = build_seed_db()
    g = build_graph(db)
    plain = SupportScreen(db, TAU)                 # V5 alone
    linked = SupportScreen(db, TAU, graph=g)       # V5 + citation graph

    print("SUPPORT ON NON-SELF-CONTAINED PROVISIONS")
    print("=" * 82)
    print(f"{'probe':9}{'provision':11}{'score':>6}{'V5 alone':>11}{'V5+graph':>16}  via")
    print("-" * 82)
    rows_plain, rows_linked = [], []
    for pid, claim, key, bad, note in SUPPORT_XREF:
        s = plain.score(claim, key)
        v_p, _ = plain.resolve(claim, key)
        v_l, sl = linked.resolve(claim, key)
        # Only the WEAK verdict is a flag; via_reference is a weaker annotation.
        f_p = v_p == SupportScreen.WEAK
        f_l = v_l == SupportScreen.WEAK
        rows_plain.append((pid, bad, f_p))
        rows_linked.append((pid, bad, f_l))
        print(f"{pid:9}{key:11}{s.score:>6.2f}"
              f"{('FLAG' if f_p else 'pass'):>11}"
              f"{v_l:>16}  {sl.via_key or '-'}")
    print("-" * 82)

    cp, cl = _counts(rows_plain), _counts(rows_linked)
    print(f"{'':9}{'flagging precision':>28}{'recall':>10}{'false alarms':>15}")
    print(f"{'V5 alone':9}{cp['precision']:>28.3f}{cp['recall']:>10.3f}{cp['fp']:>15}")
    print(f"{'V5+graph':9}{cl['precision']:>28.3f}{cl['recall']:>10.3f}{cl['fp']:>15}")
    print(f"\nFalse alarms on correct citations removed: {cp['fp'] - cl['fp']}")
    print(f"Wrong attributions no longer hard-flagged : {cp['tp'] - cl['tp']}"
          f"   (they are annotated as support-via-reference, not passed)")

    # ---- regression on the sets V5 was tuned against --------------------
    print("\nREGRESSION on the existing support probe sets")
    print("-" * 82)
    combined = list(SUPPORT_PROBES) + list(SUPPORT_HARD)
    rp, rl = [], []
    moved = []
    for pid, claim, key, bad, note in combined:
        v_p, _ = plain.resolve(claim, key)
        v_l, sl = linked.resolve(claim, key)
        rp.append((pid, bad, v_p == SupportScreen.WEAK))
        rl.append((pid, bad, v_l == SupportScreen.WEAK))
        if v_p != v_l:
            moved.append((pid, key, bad, v_p, v_l, sl.via_key))
    ep, el = _counts(rp), _counts(rl)
    print(f"{'V5 alone':9}  precision={ep['precision']:.3f}  recall={ep['recall']:.3f}"
          f"  ({ep['tp']}/{ep['tp']+ep['fn']} wrong attributions flagged)")
    print(f"{'V5+graph':9}  precision={el['precision']:.3f}  recall={el['recall']:.3f}"
          f"  ({el['tp']}/{el['tp']+el['fn']} wrong attributions flagged)")
    if moved:
        print("\n  verdict changes:")
        for pid, key, bad, a, b, via in moved:
            print(f"    {pid:9}{key:11}{a} -> {b} via {via}"
                  f"   [{'WRONG attribution' if bad else 'correct claim'}]")

    return {
        "tau": TAU,
        "xref": {"plain": cp, "linked": cl},
        "existing": {"plain": ep, "linked": el},
        "moved": moved,
        "false_alarms_removed": cp["fp"] - cl["fp"],
        "hard_flags_downgraded": cp["tp"] - cl["tp"],
    }


if __name__ == "__main__":
    report()
