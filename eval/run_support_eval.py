import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import build_seed_db
from verify.support import SupportScreen
from eval.verifier_probes import SUPPORT_PROBES


def report():
    db = build_seed_db(); sc = SupportScreen(db)

    print("SUPPORT SCORES (V1-V4 pass on every probe below)")
    print("="*76)
    print(f"{'probe':10}{'provision':11}{'score':>7}  {'label':11} matched terms")
    print("-"*76)
    rows=[]
    for pid, claim, key, bad, note in SUPPORT_PROBES:
        s = sc.score(claim, key)
        rows.append((pid, key, s.score, bad))
        print(f"{pid:10}{key:11}{s.score:>7.2f}  {'UNSUPPORTED' if bad else 'supported ':11} "
              f"{', '.join(s.matched[:4]) or '-'}")

    print("\nTHRESHOLD CALIBRATION")
    print("-"*76)
    print(f"{'tau':>6}{'TP':>5}{'FP':>5}{'TN':>5}{'FN':>5}{'prec':>7}{'rec':>7}{'F1':>7}")
    best=None
    calibration={}
    for i in range(0,21):
        tau=i/20
        tp=fp=tn=fn=0
        for pid,key,score,bad in rows:
            flag = score < tau
            if flag and bad: tp+=1
            elif flag and not bad: fp+=1
            elif not flag and not bad: tn+=1
            else: fn+=1
        p = tp/(tp+fp) if tp+fp else 0.0
        r = tp/(tp+fn) if tp+fn else 0.0
        f = 2*p*r/(p+r) if p+r else 0.0
        calibration[round(tau,3)]={"tp":tp,"fp":fp,"tn":tn,"fn":fn,
                                   "precision":p,"recall":r,"f1":f}
        if tau in (0.1,0.2,0.25,0.3,0.35,0.4,0.5,0.6):
            print(f"{tau:>6.2f}{tp:>5}{fp:>5}{tn:>5}{fn:>5}{p:>7.2f}{r:>7.2f}{f:>7.2f}")
        if best is None or f>best[1]: best=(tau,f,p,r)
    print("-"*76)
    print(f"Best F1 = {best[1]:.2f} at tau = {best[0]:.2f} (precision {best[2]:.2f}, recall {best[3]:.2f})")

    sup=[r[2] for r in rows if not r[3]]; uns=[r[2] for r in rows if r[3]]
    margin = min(sup)-max(uns)
    print(f"\nSupported   scores: min={min(sup):.2f} max={max(sup):.2f} mean={sum(sup)/len(sup):.2f}")
    print(f"Unsupported scores: min={min(uns):.2f} max={max(uns):.2f} mean={sum(uns)/len(uns):.2f}")
    print(f"Separation margin : {margin:+.2f}  "
          f"({'separable' if margin>0 else 'OVERLAPPING'})")

    return {
        "rows": rows,
        "calibration": calibration,
        "best": {"tau": best[0], "f1": best[1], "precision": best[2], "recall": best[3]},
        "supported": {"min": min(sup), "max": max(sup), "mean": sum(sup)/len(sup)},
        "unsupported": {"min": min(uns), "max": max(uns), "mean": sum(uns)/len(uns)},
        "margin": margin,
    }


if __name__ == "__main__":
    report()
