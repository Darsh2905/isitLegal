import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import build_seed_db
from verify.support import SupportScreen
from eval.verifier_probes import SUPPORT_PROBES, SUPPORT_HARD

# The threshold V5 is actually deployed at (verify/verifier.py, tau_support).
DEPLOYED_TAU = 0.20


def _confusion(rows, tau):
    tp=sum(1 for _,s,b in rows if s<tau and b)
    fp=sum(1 for _,s,b in rows if s<tau and not b)
    fn=sum(1 for _,s,b in rows if s>=tau and b)
    tn=sum(1 for _,s,b in rows if s>=tau and not b)
    p=tp/(tp+fp) if tp+fp else 0.0
    r=tp/(tp+fn) if tp+fn else 0.0
    f=2*p*r/(p+r) if p+r else 0.0
    return {"tp":tp,"fp":fp,"tn":tn,"fn":fn,"precision":p,"recall":r,"f1":f}


def report():
    db=build_seed_db(); sc=SupportScreen(db)

    def section(name, probes):
        print(f"\n{name}")
        print("-"*76)
        rows=[]
        for pid, claim, key, bad, note in probes:
            s=sc.score(claim,key); rows.append((pid,s.score,bad))
            print(f"  {pid:10}{key:11}{s.score:>6.2f}  "
                  f"{'WRONG' if bad else 'right':6} {note[:34]}")
        sup=[r[1] for r in rows if not r[2]]; uns=[r[1] for r in rows if r[2]]
        print(f"  right  : min={min(sup):.2f} mean={sum(sup)/len(sup):.2f}")
        print(f"  wrong  : max={max(uns):.2f} mean={sum(uns)/len(uns):.2f}")
        sep=min(sup)-max(uns)
        print(f"  margin : {sep:+.2f}  {'SEPARABLE' if sep>0 else 'OVERLAPPING -- no threshold works'}")
        return rows, sep

    print("="*76); print("SUPPORT SCREEN UNDER INCREASING DIFFICULTY"); print("="*76)
    easy, easy_margin = section("EASY: unrelated subject matter", SUPPORT_PROBES)
    hard, hard_margin = section("HARD: topically adjacent provisions", SUPPORT_HARD)

    print("\nBEST ACHIEVABLE THRESHOLD ON HARD SET")
    print("-"*76)
    best=(0,0,0,0)
    for i in range(0,41):
        tau=i/40
        c=_confusion(hard,tau)
        if c["f1"]>best[1]: best=(tau,c["f1"],c["precision"],c["recall"])
    print(f"  best F1={best[1]:.2f} at tau={best[0]:.2f} (P={best[2]:.2f} R={best[3]:.2f})")

    print("\nCombined (easy + hard):")
    allr=easy+hard
    best2=(0,0,0,0)
    for i in range(0,41):
        tau=i/40
        c=_confusion(allr,tau)
        if c["f1"]>best2[1]: best2=(tau,c["f1"],c["precision"],c["recall"])
    print(f"  best F1={best2[1]:.2f} at tau={best2[0]:.2f} (P={best2[2]:.2f} R={best2[3]:.2f})")

    # ---- DEPLOYED OPERATING POINT --------------------------------------
    # V5 is advisory-only and is deployed at tau=0.20, chosen as the highest
    # threshold that keeps FLAGGING PRECISION at 1.000 -- i.e. no correct
    # claim is ever flagged -- accepting the recall cost. The paper and README
    # quote this operating point, so it is printed here rather than left to be
    # re-derived by hand (see CLAUDE.md invariant 6).
    dep = _confusion(allr, DEPLOYED_TAU)
    print(f"\nDEPLOYED OPERATING POINT (tau={DEPLOYED_TAU:.2f}, advisory-only, combined set)")
    print("-"*76)
    print(f"  flagged {dep['tp']}/{dep['tp']+dep['fn']} wrong attributions "
          f"(recall {dep['recall']:.3f}), "
          f"{dep['fp']} correct claims flagged (precision {dep['precision']:.3f})")

    # The band of thresholds over which flagging precision stays at 1.000.
    band=[i/40 for i in range(1,41)
          if _confusion(allr,i/40)["fp"]==0 and _confusion(allr,i/40)["tp"]>0]
    if band:
        print(f"  precision stays 1.000 for tau in [{min(band):.3f}, {max(band):.3f}]; "
              f"recall plateaus at {dep['recall']:.3f}")
    print("  V5 NEVER STRIPS -- these flags annotate only (CLAUDE.md invariant 4).")

    return {
        "easy": {"rows": easy, "margin": easy_margin},
        "hard": {"rows": hard, "margin": hard_margin},
        "best_hard": {"tau": best[0], "f1": best[1], "precision": best[2], "recall": best[3]},
        "best_combined": {"tau": best2[0], "f1": best2[1], "precision": best2[2], "recall": best2[3]},
        "deployed": dict(dep, tau=DEPLOYED_TAU),
        "precision_1_band": (min(band), max(band)) if band else None,
    }


if __name__ == "__main__":
    report()
