import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import build_seed_db
from verify.verifier import CitationVerifier, Chunk, Violation
from eval.verifier_probes import QUOTE_PROBES, CONTEXT


def report():
    db = build_seed_db()
    chunks = []
    for idx, key in CONTEXT:
        p = db.get(key)
        chunks.append(Chunk(idx, key, p.text, p.verbatim))

    print("QUOTATION CHECK (V4)  theta sweep")
    print("="*72)
    sweep = {}
    for theta in (0.80, 0.90, 0.95, 0.98):
        ver = CitationVerifier(db, theta=theta)
        tp=fp=tn=fn=0
        for pid, ans, hf, ft, note in QUOTE_PROBES:
            _, rep = ver.verify(ans, chunks)
            flagged = len(rep.by_type(Violation.UNSUPPORTED_QUOTE))>0
            if flagged and hf: tp+=1
            elif flagged and not hf: fp+=1
            elif not flagged and not hf: tn+=1
            else: fn+=1
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec  = tp/(tp+fn) if tp+fn else 0.0
        sweep[theta] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                        "precision": prec, "recall": rec}
        print(f"theta={theta:.2f}  TP={tp} FP={fp} TN={tn} FN={fn}  P={prec:.2f} R={rec:.2f}")

    print("\nPer-probe detail at theta=0.95")
    print("-"*72)
    ver = CitationVerifier(db, theta=0.95)
    per_probe = {}
    checks_performed = checks_skipped = 0
    for pid, ans, hf, ft, note in QUOTE_PROBES:
        out, rep = ver.verify(ans, chunks)
        q = rep.by_type(Violation.UNSUPPORTED_QUOTE)
        status = "FLAGGED" if q else "passed "
        exp = "fault" if hf else "clean"
        mark = "OK " if (bool(q)==hf) else "MISS"
        per_probe[pid] = {"expected_fault": hf, "flagged": bool(q),
                          "correct": bool(q) == hf,
                          "details": [f.detail for f in q]}
        checks_performed += rep.quote_checks_performed
        checks_skipped += rep.quote_checks_skipped
        print(f"{mark} {pid:11} expect={exp:5} {status}  {note}")
        for f in q: print(f"        -> {f.detail}")
        print(f"        checks performed={rep.quote_checks_performed} skipped={rep.quote_checks_skipped}")

    return {"sweep": sweep, "per_probe": per_probe,
            "quote_checks_performed": checks_performed,
            "quote_checks_skipped": checks_skipped}


if __name__ == "__main__":
    report()
