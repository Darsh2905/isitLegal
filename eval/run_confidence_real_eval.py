"""
Confidence gate / calibrated abstention (C3) -- RECALIBRATED ON THE REAL CORPUS.

WHY THIS SCRIPT EXISTS, SEPARATELY FROM run_confidence_eval.py. The deployed
gate (retrieval/confidence.py, tau=0.305) was calibrated only against the
35-provision toy corpus, and that script's own docstring says so explicitly:
"these numbers demonstrate that the method is sound, not that the gate
performs at this level." That caveat turned out to be load-bearing, not
decorative -- an ad-hoc smoke test of the deployed threshold against the real
3,263-provision corpus produced a false positive on "How is agricultural
income taxed?", a query the toy-corpus gate correctly refuses. The real
corpus's much larger, lexically competitive vocabulary changes the score
distribution the threshold was tuned against, so a threshold tuned on one
does not transfer to the other -- which is exactly why app.py's Live
Generation tab (Tab 6) has been skipping the gate entirely rather than
silently misapply an out-of-domain threshold.

METHODOLOGY. Identical discipline to run_confidence_eval.py: the threshold
(and the signal itself) is tuned on a development split and reported on a
held-out test split, never the reverse. ANSWERABLE reuses eval/gold_qa.py's
16 queries unchanged (their gold keys exist in the real corpus, a superset of
the toy one). UNANSWERABLE reuses eval/unanswerable_qa.py's 16 queries
unchanged -- the real corpus still covers the same ten Acts, so a query about
tax, IP, family law, company law, labour law, or limitation is still
genuinely out of scope, not merely out of scope for the toy subset.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.india_code import build_real_db
from retrieval.bm25_baseline import BM25Retriever
from retrieval.confidence import ConfidenceGate
from eval.gold_qa import GOLD
from eval.unanswerable_qa import UNANSWERABLE_DEV, UNANSWERABLE_TEST

ANSWERABLE = [g[1] for g in GOLD]
ANSWERABLE_DEV = ANSWERABLE[0::2]
ANSWERABLE_TEST = ANSWERABLE[1::2]


def _signals(db, r, gate):
    def raw_top(q):
        res = r.search(q, 2)
        return res[0][1] if res else 0.0
    def margin(q):
        res = r.search(q, 2)
        return (res[0][1] - res[1][1]) if len(res) > 1 else 0.0
    def normalised(q):
        return gate.confidence(q).score
    # Bounds are swept empirically below (see _autoscale) rather than reused
    # from the toy-corpus script: raw BM25 magnitudes shift with corpus size
    # and IDF distribution, so a bound tuned for 35 provisions is not known
    # to be valid for 3,263 without checking.
    return [("raw top-1", raw_top), ("top1-top2 margin", margin),
           ("normalised (ours)", normalised)]


def _autoscale(fn, queries, pad=1.3):
    vals = [fn(q) for q in queries]
    return max(vals) * pad if vals else 1.0


def _tune(fn, hi):
    a = [fn(q) for q in ANSWERABLE_DEV]
    u = [fn(q) for q in UNANSWERABLE_DEV]
    steps = 400
    best, rows = -1.0, []
    for i in range(steps + 1):
        tau = hi * i / steps
        cov = sum(1 for x in a if x >= tau) / len(a)
        abst = 1 - sum(1 for x in u if x >= tau) / len(u)
        bal = (cov + abst) / 2
        rows.append((tau, bal))
        best = max(best, bal)
    top = [t for t, b in rows if b == best]
    return round((min(top) + max(top)) / 2, 4), best, (min(top), max(top))


def _eval_split(fn, tau, answerable, unanswerable):
    ans_ok = sum(1 for q in answerable if fn(q) >= tau)
    bad = sum(1 for q in unanswerable if fn(q) >= tau)
    cov = ans_ok / len(answerable)
    abst = 1 - bad / len(unanswerable)
    return {"coverage": cov, "correct_abstention": abst,
            "balanced_accuracy": (cov + abst) / 2,
            "wrongly_answered": bad,
            "wrongly_abstained": len(answerable) - ans_ok}


def report():
    db = build_real_db()
    # The gate's signal is calibrated on UNSTEMMED BM25: stemming, though it
    # improves retrieval, raises lexical overlap for out-of-scope questions too
    # and cut held-out correct abstention 1.000 -> 0.625 (retrieval/text.py).
    r = BM25Retriever(db, stem=False)
    gate = ConfidenceGate(db, r)
    signals = _signals(db, r, gate)

    print("CONFIDENCE GATE / CALIBRATED ABSTENTION (C3) -- REAL CORPUS")
    print("=" * 78)
    print(f"corpus: {db.count()} provisions (vs. 35 in the original calibration)")
    print(f"answerable: {len(ANSWERABLE_DEV)} dev / {len(ANSWERABLE_TEST)} test   "
          f"unanswerable: {len(UNANSWERABLE_DEV)} dev / {len(UNANSWERABLE_TEST)} test")
    print("Every threshold below is tuned on DEV and reported on TEST.")

    print("\nSIGNAL COMPARISON  (tuned on dev, reported on held-out test)")
    print("-" * 78)
    print(f"{'signal':20}{'dev sep.':>10}{'tau*':>9}{'dev bal.':>10}"
          f"{'TEST bal.':>11}{'test errors':>13}")
    results = {}
    for name, fn in signals:
        a = [fn(q) for q in ANSWERABLE_DEV]
        u = [fn(q) for q in UNANSWERABLE_DEV]
        dev_sep = min(a) - max(u)
        hi = _autoscale(fn, ANSWERABLE_DEV + UNANSWERABLE_DEV)
        tau, dev_bal, interval = _tune(fn, hi)
        dv = _eval_split(fn, tau, ANSWERABLE_DEV, UNANSWERABLE_DEV)
        te = _eval_split(fn, tau, ANSWERABLE_TEST, UNANSWERABLE_TEST)
        results[name] = {"dev_separation": dev_sep, "tau": tau,
                         "dev": dv, "test": te, "interval": interval, "hi": hi}
        errs = te["wrongly_answered"] + te["wrongly_abstained"]
        print(f"{name:20}{dev_sep:>+10.3f}{tau:>9.3f}{dv['balanced_accuracy']:>10.3f}"
              f"{te['balanced_accuracy']:>11.3f}{errs:>13}")
    print("-" * 78)

    chosen = "normalised (ours)"
    dev_pick = max(results, key=lambda k: results[k]["dev"]["balanced_accuracy"])
    test_pick = max(results, key=lambda k: results[k]["test"]["balanced_accuracy"])
    print(f"Signal with best DEV balanced accuracy : {dev_pick}")
    print(f"Signal with best TEST balanced accuracy: {test_pick}")
    print(f"Deployed signal: {chosen}")

    res = results[chosen]
    tau_star = res["tau"]
    print(f"\nDEPLOYED GATE (real corpus)  tau* = {tau_star:.3f} "
          f"(midpoint of the dev-optimal interval "
          f"[{res['interval'][0]:.3f}, {res['interval'][1]:.3f}])")
    print(f"  (deployed toy-corpus threshold was tau=0.305 -- not assumed "
          f"equal to this one)")
    print("-" * 78)
    for label, m, n_a, n_u in (("dev", res["dev"], len(ANSWERABLE_DEV), len(UNANSWERABLE_DEV)),
                               ("test", res["test"], len(ANSWERABLE_TEST), len(UNANSWERABLE_TEST))):
        print(f"  {label:5} coverage={m['coverage']:.3f} "
              f"({n_a - m['wrongly_abstained']}/{n_a})"
              f"   correct abstention={m['correct_abstention']:.3f} "
              f"({n_u - m['wrongly_answered']}/{n_u})"
              f"   balanced={m['balanced_accuracy']:.3f}")
    print(f"\n  Unanswerable queries wrongly answered on test: "
          f"{res['test']['wrongly_answered']}")
    print(f"  Answerable queries wrongly refused on test    : "
          f"{res['test']['wrongly_abstained']}")

    print("\nPER-QUERY CONFIDENCE (test split, deployed signal)")
    print("-" * 78)
    fn = dict((n, f) for n, f in signals)[chosen]
    for q in ANSWERABLE_TEST:
        v = fn(q)
        print(f"  {'ANSWER' if v >= tau_star else 'abstain':8}{v:>7.3f}  "
              f"answerable    {q[:44]}")
    for q in UNANSWERABLE_TEST:
        v = fn(q)
        print(f"  {'ANSWER' if v >= tau_star else 'abstain':8}{v:>7.3f}  "
              f"UNANSWERABLE  {q[:44]}")

    # Regression check specifically for the false positive that motivated
    # this script: the toy-corpus tau=0.305 applied to the real corpus
    # answered "How is agricultural income taxed?" when it should abstain.
    agri_q = "How is agricultural income taxed?"
    agri_conf = gate.confidence(agri_q).score
    print(f"\nMotivating regression check: {agri_q!r}")
    print(f"  confidence={agri_conf:.3f}  vs tau*={tau_star:.3f}  "
          f"-> {'ANSWER (still fails)' if agri_conf >= tau_star else 'abstain (fixed)'}")
    print(f"  (toy-corpus tau=0.305 would have answered this: "
          f"{'yes -- reproduces the bug' if agri_conf >= 0.305 else 'no'})")

    return {"corpus_size": db.count(), "tau": tau_star, "chosen": chosen,
            "dev_pick": dev_pick, "test_pick": test_pick, "signals": results,
            "dev": res["dev"], "test": res["test"],
            "agri_confidence": agri_conf,
            "agri_fixed": agri_conf < tau_star,
            "n": {"ans_dev": len(ANSWERABLE_DEV), "ans_test": len(ANSWERABLE_TEST),
                  "un_dev": len(UNANSWERABLE_DEV), "un_test": len(UNANSWERABLE_TEST)}}


if __name__ == "__main__":
    report()
