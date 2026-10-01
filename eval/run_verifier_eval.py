import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.metadata_db import build_seed_db
from verify.verifier import CitationVerifier, Chunk, Violation
from eval.verifier_probes import PROBES, CONTEXT


def report():
    db = build_seed_db()
    ver = CitationVerifier(db, theta=0.95)

    chunks = []
    for idx, key in CONTEXT:
        p = db.get(key)
        chunks.append(Chunk(index=idx, key=key, text=p.text, verbatim=p.verbatim))

    tp = fp = tn = fn = 0
    rows = []
    detail_fail = []

    for pid, answer, has_fault, ftypes, note in PROBES:
        out, rep = ver.verify(answer, chunks)
        flagged = len(rep.findings) > 0
        if flagged and has_fault:   tp += 1; outcome = "TP"
        elif flagged and not has_fault: fp += 1; outcome = "FP"; detail_fail.append((pid, "false alarm", rep))
        elif not flagged and not has_fault: tn += 1; outcome = "TN"
        else: fn += 1; outcome = "FN"; detail_fail.append((pid, "missed fault", rep))

        kinds = sorted({f.violation.value for f in rep.findings})
        rows.append((pid, outcome, ",".join(ftypes) or "-",
                     ";".join(kinds) or "-",
                     f"{rep.coverage:.2f}"))

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

    print("=" * 74)
    print("VERIFIER AS FAULT DETECTOR")
    print("=" * 74)
    print(f"{'probe':10} {'result':7} {'planted':14} {'detected':34} {'cov':>5}")
    print("-" * 74)
    for r in rows:
        print(f"{r[0]:10} {r[1]:7} {r[2]:14} {r[3]:34} {r[4]:>5}")
    print("-" * 74)
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}   (n={len(PROBES)})")
    print(f"Detection precision : {prec:.3f}")
    print(f"Detection recall    : {rec:.3f}")
    print(f"Detection F1        : {f1:.3f}")

    if detail_fail:
        print("\n-- CASES NEEDING ATTENTION --")
        for pid, why, rep in detail_fail:
            print(f"  {pid}: {why}")
            for f in rep.findings:
                print(f"      {f.violation.value}: {f.detail}")

    # --- coverage cost on clean vs faulty probes ---
    clean_cov = [ver.verify(a, chunks)[1].coverage
                 for _, a, hf, _, _ in PROBES if not hf]
    faulty_cov = [ver.verify(a, chunks)[1].coverage
                  for _, a, hf, _, _ in PROBES if hf]
    mean_clean = sum(clean_cov)/len(clean_cov)
    mean_faulty = sum(faulty_cov)/len(faulty_cov)
    print(f"\nMean coverage, clean probes : {mean_clean:.3f}")
    print(f"Mean coverage, faulty probes: {mean_faulty:.3f}")

    # --- worked example ---
    print("\n" + "=" * 74)
    print("WORKED EXAMPLE (probe mix-a)")
    print("=" * 74)
    ans = [p for p in PROBES if p[0] == "mix-a"][0][1]
    out, rep = ver.verify(ans, chunks)
    print("PRE-VERIFIER:\n  " + ans)
    print("\nPOST-VERIFIER:\n  " + out.replace("\n", "\n  "))
    print(f"\nFindings ({len(rep.findings)}):")
    for f in rep.findings:
        print(f"  [{f.violation.value}] {f.detail}")
    print(f"\nquote checks performed={rep.quote_checks_performed} "
          f"skipped={rep.quote_checks_skipped}")

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": len(PROBES),
        "precision": prec, "recall": rec, "f1": f1,
        "mean_coverage_clean": mean_clean,
        "mean_coverage_faulty": mean_faulty,
        "rows": rows,
        "worked_example_findings": [f.violation.value for f in rep.findings],
    }


if __name__ == "__main__":
    report()
