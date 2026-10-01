"""
The deterministic verifier on the REAL corpus, probed adversarially.

See eval/verifier_real_probes.py for how the probes are built (matched
fault / legitimate pairs, real statutory text, several constructs taken
from independently generated answers). Reported as a confusion matrix at the
level the guarantee is stated -- is a faulty claim removed, is a correct one
kept -- plus the independent post-hoc audit from eval/run_repair_eval.py run
over every emitted answer, so the result does not rest on the verifier's own
report of itself.

This is a probe set, not a sample of real-world answers: it establishes that
each check fires on the construct it targets and stays silent on the correct
form of that construct. It is not an estimate of how often generators make
these errors.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.india_code import build_real_db
from retrieval.xref import build_graph
from verify.verifier import CitationVerifier, Chunk, Violation
from eval.run_repair_eval import audit
from eval.verifier_real_probes import PROBES, MARKER_PROBES


def _chunks(db, keys):
    return [Chunk(i, k, db.get(k).text, db.get(k).verbatim)
            for i, k in enumerate(keys, start=1)]


def report():
    db = build_real_db()
    ver = CitationVerifier(db, repair=True, support=True, graph=build_graph(db))

    print("VERIFIER ON THE REAL CORPUS: ADVERSARIAL PROBES")
    print("=" * 78)
    tp = fn = tn = fp = 0
    audit_pass = 0
    rows = []
    for pid, construct, sentence, keys, expect in PROBES:
        chunks = _chunks(db, keys)
        out, rep = ver.verify(sentence, chunks)
        stripped = rep.sentences_stripped > 0
        ok_audit, why = audit(out, chunks, db)
        audit_pass += ok_audit
        if expect == "strip":
            tp += stripped
            fn += not stripped
        else:
            tn += not stripped
            fp += stripped
        correct = stripped == (expect == "strip")
        blocking = [f.violation.value for f in rep.findings
                    if f.violation.value in ("unresolved_marker", "nonexistent_provision",
                                             "unknown_act", "unsupported_quote")]
        rows.append({"id": pid, "construct": construct, "expect": expect,
                     "stripped": stripped, "correct": correct, "why": blocking,
                     "elided_accepted": rep.quotes_accepted_with_elision})
        print(f"  {pid} {'ok ' if correct else 'BAD'} expect={expect:5} "
              f"{'stripped' if stripped else 'kept':8} {construct[:44]:44} "
              f"{','.join(blocking)}")

    n_fault = tp + fn
    n_clean = tn + fp
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / n_fault if n_fault else 1.0
    print("-" * 78)
    print(f"  faults stripped       : {tp}/{n_fault}   (recall {recall:.3f})")
    print(f"  correct claims kept   : {tn}/{n_clean}   "
          f"(false alarms: {fp})")
    print(f"  strip precision       : {precision:.3f}")
    print(f"  independent audit     : {audit_pass}/{len(PROBES)} emitted answers pass")

    print("\nMARKER / CITATION CONSISTENCY (advisory + repair R3)")
    print("-" * 78)
    marker_ok = 0
    marker_rows = []
    for pid, construct, sentence, keys, expect in MARKER_PROBES:
        out, rep = ver.verify(sentence, _chunks(db, keys))
        repointed = any(r.rule == "R3" for r in rep.repair_log.repairs)
        flagged = bool(rep.by_type(Violation.MARKER_MISMATCH))
        got = "repointed" if repointed else "flagged" if flagged else "untouched"
        kept = rep.sentences_stripped == 0
        ok = got == expect and kept
        marker_ok += ok
        marker_rows.append({"id": pid, "expect": expect, "got": got, "kept": kept})
        print(f"  {pid} {'ok ' if ok else 'BAD'} expect={expect:9} got={got:9} "
              f"{construct}")
        if repointed:
            print(f"        -> {out.splitlines()[0]}")
    print(f"  {marker_ok}/{len(MARKER_PROBES)} as expected (none stripped)")

    return {"n": len(PROBES), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "precision": precision, "recall": recall,
            "audit_pass": audit_pass, "rows": rows,
            "marker_ok": marker_ok, "marker_n": len(MARKER_PROBES),
            "marker_rows": marker_rows}


if __name__ == "__main__":
    report()
