"""
The pre/post-verifier hallucination comparison, run against a REAL,
INDEPENDENTLY INVOKED generator (Tier 2, item 10's live half) -- as opposed
to eval/run_generation_eval.py, which analyses answers composed by this
project's own model.

See eval/live_generation_answers.py's module docstring for exactly what this
round establishes and what it still does not: real, independent generation,
over a query set that is still project-authored. Not a substitute for the
paper's full promised ablation on an externally-sourced adversarial set.

WHAT IS RE-DERIVED FRESH VS. FROZEN. Only the raw Groq completion text is
frozen (eval/live_generation_answers.py) -- the one part of this pipeline
that is neither free nor deterministic to reproduce. The confidence gate
decision, the retrieved chunks, and the verification are all recomputed here
from the live corpus and the current code, exactly as
eval/run_generation_eval.py already re-derives retrieval fresh rather than
trusting a cached snapshot (CLAUDE.md invariant 6).

METRICS, same definitions as run_generation_eval.py:
  - gate coverage: fraction of the 16 known-answerable gold queries the
    real-corpus gate actually let through to generation.
  - pre-verifier / post-verifier citation resolution accuracy: independently
    re-derived (mirrors run_repair_eval.py's audit() pattern), not read from
    the verifier's own report.
  - coverage: fraction of sentences surviving verification.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.india_code import build_real_db
from ingest.normalize import extract_citations, extract_unresolved_act_refs
from retrieval.bm25_baseline import BM25Retriever
from retrieval.confidence import ConfidenceGate
from verify.verifier import CitationVerifier, Chunk
from ingest.case_citations import build_case_index
from eval.citation_audit import citation_stats, strip_annotations
from eval.stats import fmt
from eval.gold_qa import GOLD
from eval.unanswerable_qa import UNANSWERABLE_DEV, UNANSWERABLE_TEST
from eval.live_generation_answers import LIVE_ANSWERABLE, LIVE_UNANSWERABLE_ALL_ABSTAINED

REAL_CORPUS_TAU = 1.077


def _citation_stats(answer, chunks, db, cases):
    """Shared independent audit (eval/citation_audit.py): every citation
    form the verifier checks, re-derived from the text alone."""
    total, bad, unmapped, _ = citation_stats(answer, chunks, db, cases)
    return total, bad, unmapped


def report():
    db = build_real_db()
    cases = build_case_index()
    ver = CitationVerifier(db, repair=True, support=True, cases=cases)
    by_key = {p.key: p for p in db.all_provisions()}

    print("LIVE PRE/POST-VERIFIER COMPARISON (independently invoked generator)")
    print("=" * 78)
    print("Model: openai/gpt-oss-20b via Groq. See eval/live_generation_answers.py")
    print("for the full methodological disclosure -- real generation, still a")
    print("project-authored query set.")

    # --- Gate behaviour, re-derived fresh (deterministic, free) ---
    # The gate's signal is calibrated on UNSTEMMED BM25: stemming, though it
    # improves retrieval, raises lexical overlap for out-of-scope questions too
    # and cut held-out correct abstention 1.000 -> 0.625 (retrieval/text.py).
    bm25 = BM25Retriever(db, stem=False)
    gate = ConfidenceGate(db, bm25, tau=REAL_CORPUS_TAU)

    print("\nCONFIDENCE GATE ON THE GOLD SET, END TO END")
    print("-" * 78)
    gate_pass = gate_abstain = 0
    for qid, q, rel, in_force in GOLD:
        answerable, conf = gate.gate(q)
        frozen = LIVE_ANSWERABLE[qid]
        # Re-derived decision must match what was actually captured -- this
        # is the reproducibility check invariant 6 exists for.
        assert answerable == (frozen["outcome"] == "generated"), (
            f"{qid}: gate decision does not reproduce the captured run "
            f"(recomputed answerable={answerable}, captured outcome="
            f"{frozen['outcome']!r}) -- code has changed since capture")
        if answerable:
            gate_pass += 1
        else:
            gate_abstain += 1
            print(f"  {qid:5} ABSTAIN (score={conf.score:.3f})  {q[:50]}")
    gate_coverage = gate_pass / len(GOLD)
    print(f"\n  Gate coverage on the full gold set: {gate_coverage:.3f} "
          f"({gate_pass}/{len(GOLD)})")
    print(f"  This is NOT the held-out test-split coverage reported in "
          f"Section~VII-N (0.875, 7/8) -- it is the deployed gate applied "
          f"end to end to every gold query, dev-split queries included, "
          f"which is what a real user of the system actually experiences.")

    un_pass = 0
    for q in UNANSWERABLE_DEV + UNANSWERABLE_TEST:
        answerable, _ = gate.gate(q)
        un_pass += answerable
    print(f"\n  Unanswerable queries wrongly answered by the gate: {un_pass}/"
          f"{len(UNANSWERABLE_DEV) + len(UNANSWERABLE_TEST)}")
    assert un_pass == 0, "reproducibility check: capture recorded 0/16 wrongly answered"

    # --- Citation verification on every query the gate actually generated for ---
    print("\nCITATION VERIFICATION ON GENERATED ANSWERS")
    print("-" * 78)
    rows = []
    for qid, q, rel, in_force in GOLD:
        frozen = LIVE_ANSWERABLE[qid]
        if frozen["outcome"] != "generated":
            continue
        answer = frozen["raw_text"]
        chunk_keys = frozen["chunk_keys"]
        chunks = [Chunk(i, k, by_key[k].text, by_key[k].verbatim)
                 for i, k in enumerate(chunk_keys, start=1)]

        pre_total, pre_bad, pre_unmapped = _citation_stats(answer, chunks, db, cases)
        out, rep = ver.verify(answer, chunks)
        post_total, post_bad, post_unmapped = _citation_stats(
            strip_annotations(out), chunks, db, cases)

        rows.append({"qid": qid, "pre_total": pre_total, "pre_bad": pre_bad,
                    "pre_unmapped": pre_unmapped, "post_total": post_total,
                    "post_bad": post_bad, "post_unmapped": post_unmapped,
                    "coverage": rep.coverage, "findings": len(rep.findings),
                    "by_type": [f.violation.value for f in rep.findings],
                    "elided_accepted": rep.quotes_accepted_with_elision})
        print(f"  {qid:5} pre: {pre_total-pre_bad}/{pre_total} resolve "
              f"({pre_unmapped} unmapped-repealed)   "
              f"post: {post_total-post_bad}/{post_total} resolve "
              f"({post_unmapped} unmapped-repealed)   "
              f"coverage={rep.coverage:.2f}   findings={len(rep.findings)}")

    pre_total = sum(r["pre_total"] for r in rows)
    pre_bad = sum(r["pre_bad"] for r in rows)
    post_total = sum(r["post_total"] for r in rows)
    post_bad = sum(r["post_bad"] for r in rows)
    post_unmapped = sum(r["post_unmapped"] for r in rows)
    pre_acc = 1 - pre_bad / pre_total if pre_total else 1.0
    post_acc = 1 - post_bad / post_total if post_total else 1.0
    mean_cov = sum(r["coverage"] for r in rows) / len(rows) if rows else 0.0

    print(f"\n  PRE-verifier resolution accuracy : {pre_acc:.3f}  "
          f"({pre_total-pre_bad}/{pre_total})")
    print(f"  POST-verifier resolution accuracy: {post_acc:.3f}  "
          f"({post_total-post_bad}/{post_total})  <- the core guarantee (V1/V2/V4)")
    print(f"  Repealed citations annotated but not stripped (no recorded "
          f"successor, V3 by design): {post_unmapped}")
    print(f"  Mean coverage: {mean_cov:.3f}")
    from collections import Counter
    by_type = Counter(v for r in rows for v in r["by_type"])
    elided = sum(r["elided_accepted"] for r in rows)
    print(f"  Findings by type: {dict(sorted(by_type.items()))}")
    print(f"  Quotations accepted as faithful elisions (each segment verbatim,"
          f" in order, no negation/exception omitted): {elided}")

    return {
        "model": "openai/gpt-oss-20b",
        "gate_coverage_full_gold": gate_coverage,
        "gate_pass": gate_pass, "gate_abstain": gate_abstain,
        "gate_n": len(GOLD),
        "unanswerable_wrongly_answered": un_pass,
        "unanswerable_n": len(UNANSWERABLE_DEV) + len(UNANSWERABLE_TEST),
        "rows": rows,
        "pre_accuracy": pre_acc, "post_accuracy": post_acc,
        "pre_total": pre_total, "pre_bad": pre_bad,
        "post_total": post_total, "post_bad": post_bad,
        "post_unmapped_repealed": post_unmapped,
        "mean_coverage": mean_cov,
        "n_generated": len(rows),
        "findings_by_type": dict(by_type),
        "elided_quotes_accepted": elided,
    }


if __name__ == "__main__":
    report()
