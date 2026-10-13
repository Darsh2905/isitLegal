"""
V4 quotation check: the toy-corpus theta sweep (the original character-vs-
token negative result) and, since 2026-10, a LENGTH SWEEP on real statutory
text.

Why the length sweep exists. theta = 0.95 on (longest verbatim run / quote
length) was justified by one probe: an 18-token Article 21 quotation with its
LAST word substituted, which scores 17/18 = 0.944 and is rejected. A peer
review pointed out that the same substitution in a 20-token quote scores
19/20 = 0.95 and passes, and in longer quotes the substitution can move
inward and still pass. The toy probe set could not show this: its one
verbatim provision is 18 tokens of quotable text. The sweep below builds the
easy and hard cases explicitly (CLAUDE.md Section 6, Class C): faithful
excerpts of real provisions, which must pass, and every one-word
substitution at four positions, which must fail, across quote lengths and
thresholds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import build_seed_db
from verify.verifier import CitationVerifier, Chunk, Violation, _tokens
from eval.verifier_probes import QUOTE_PROBES, CONTEXT

# Long real provisions, chosen for running prose without sub-clause labels.
SWEEP_KEYS = ["CONST§22", "CRPC§41", "BNSS§35", "CONST§21", "IPC§300",
              "BNS§101", "IEA§65B", "BSA§63", "NIA§138", "ITA§66"]
SWEEP_LENGTHS = (10, 15, 18, 19, 20, 25, 30, 40, 60)
SWEEP_THETAS = (0.90, 0.95, 0.98, 1.00)
POSITIONS = ("first", "second", "middle", "last")


def length_sweep():
    from ingest.india_code import build_real_db
    db = build_real_db()
    rows = {}
    for theta in SWEEP_THETAS:
        ver = CitationVerifier(db, theta=theta, residual=False)
        for n in SWEEP_LENGTHS:
            faithful_ok = faithful_n = sub_pass = sub_n = 0
            for key in SWEEP_KEYS:
                p = db.get(key)
                if p is None:
                    continue
                words = [w for w in p.text.split() if _tokens(w)]
                if len(words) < n + 5:
                    continue
                for off in (0, 3):
                    q = words[off:off + n]
                    ch = [Chunk(1, key, p.text, True)]
                    _, rep = ver.verify(f'It provides that "{" ".join(q)}" [1].', ch)
                    faithful_n += 1
                    faithful_ok += not rep.by_type(Violation.UNSUPPORTED_QUOTE)
                    for pos in POSITIONS:
                        i = {"first": 0, "second": 1, "middle": n // 2,
                             "last": n - 1}[pos]
                        qq = list(q)
                        qq[i] = "statute" if _tokens(q[i]) != ["statute"] else "law"
                        _, rep = ver.verify(f'It provides that "{" ".join(qq)}" [1].', ch)
                        sub_n += 1
                        sub_pass += not rep.by_type(Violation.UNSUPPORTED_QUOTE)
            rows[(theta, n)] = {"faithful_accepted": faithful_ok,
                                "faithful_n": faithful_n,
                                "substitution_accepted": sub_pass,
                                "substitution_n": sub_n}
    return rows


def report():
    db = build_seed_db()
    chunks = []
    for idx, key in CONTEXT:
        p = db.get(key)
        chunks.append(Chunk(idx, key, p.text, p.verbatim))

    print("QUOTATION CHECK (V4)  theta sweep")
    print("="*72)
    sweep = {}
    for theta in (0.80, 0.90, 0.95, 0.98, 1.00):
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

    print("\nPer-probe detail at the deployed theta=1.00 (exact)")
    print("-"*72)
    ver = CitationVerifier(db, theta=1.0)
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

    print("\nLENGTH SWEEP on real provisions: one-word substitutions ACCEPTED")
    print("(must be 0) and faithful excerpts accepted (must be all), by theta")
    print("-"*72)
    ls = length_sweep()
    print("  n tokens " + "".join(f"  theta={t:.2f}  " for t in SWEEP_THETAS))
    for n in SWEEP_LENGTHS:
        cells = []
        for t in SWEEP_THETAS:
            r = ls[(t, n)]
            cells.append(f"{r['substitution_accepted']:>3}/{r['substitution_n']:<3}"
                         f" ok {r['faithful_accepted']}/{r['faithful_n']}")
        print(f"  {n:>8} " + "  ".join(cells))
    return {"sweep": sweep, "per_probe": per_probe,
            "quote_checks_performed": checks_performed,
            "quote_checks_skipped": checks_skipped,
            "length_sweep": ls}


if __name__ == "__main__":
    report()
