"""
The pre/post-verifier hallucination comparison (Tier 2, item 10 -- the
paper's headline ablation, run for the first time).

Uses the recorded answers in eval/generation_answers.py -- see that module's
docstring for exactly what these numbers are and, more importantly, are NOT:
a real generation happened, composed by this project's own generating model
rather than an independently invoked API call, which is a genuine limitation
on how far these specific numbers generalise, stated there in full and not
repeated here.

METRICS. For each condition (ungrounded, grounded):
  - citations_total / citations_bad: every citation found in the PRE-verifier
    answer, and how many are wrong (nonexistent, unresolved marker, or
    superseded without a mapping).
  - pre-verifier citation accuracy: 1 - citations_bad/citations_total.
  - post-verifier citation accuracy: always 1.000 by construction (the
    verifier's core guarantee) -- reported to make the point explicit, not
    because it is a discovery.
  - coverage: fraction of sentences surviving verification.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.india_code import build_real_db
from ingest.normalize import extract_citations, extract_unresolved_act_refs
from retrieval.bm25_baseline import BM25Retriever
from retrieval.dense import DenseEncoder, DenseRetriever, MODEL_IDS
from retrieval.fusion import FusedRetriever
from retrieval.rerank import Reranker
from verify.verifier import CitationVerifier, Chunk
from ingest.case_citations import build_case_index
from eval.citation_audit import citation_stats, strip_annotations
from eval.generation_answers import UNGROUNDED_ANSWERS, GROUNDED_ANSWERS
from eval.gold_qa import GOLD


def _retrieve_chunks_for_gold(db):
    """
    Re-derives the exact top-5 chunks each gold query retrieves through the
    real pipeline (BM25 + dense, RRF-fused, cross-encoder reranked) fresh,
    rather than reading a cached snapshot -- CLAUDE.md invariant 6: every
    number must be reproducible by running a script in eval/. Retrieval is
    deterministic (fixed BM25 index, fixed embedding model, fixed RRF
    constant, fixed cross-encoder), so this reproduces exactly the chunks the
    GROUNDED_ANSWERS in eval/generation_answers.py were composed against.
    """
    # PINNED to the retrieval configuration GROUNDED_ANSWERS were composed
    # against (unstemmed BM25, whole-provision dense vectors and reranking).
    # Those answers'
    # [n] markers index THAT top-5; re-deriving it with a later retriever
    # would move the markers under frozen text and change the scores for a
    # reason unrelated to answer quality.
    bm25 = BM25Retriever(db, stem=False)
    enc = DenseEncoder(MODEL_IDS["general"])
    dense = DenseRetriever(db, enc, passages=False)
    fused = FusedRetriever([("bm25", bm25), ("dense", dense)], candidate_k=20)
    reranker = Reranker(db, passages=False)
    by_key = {p.key: p for p in db.all_provisions()}

    chunks_by_qid = {}
    for qid, q, rel, _ in GOLD:
        cands = fused.search(q, 20)
        top = reranker.rerank(q, cands, top_k=5)
        chunks_by_qid[qid] = [Chunk(i, key, by_key[key].text, True)
                              for i, (key, _score) in enumerate(top, start=1)]
    return chunks_by_qid


def _citation_stats(answer, chunks, db, ver):
    """
    Independently re-derived citation counts -- the shared audit in
    eval/citation_audit.py, which re-derives every citation form the verifier
    checks (statutory, unknown-act, bare, case, marker, unreadable) from the
    text alone rather than trusting the verifier's own report. Two DIFFERENT
    things are counted on purpose:

      bad        -- violates the core guarantee (V1/V2/V4): a citation in
                    any form that does not resolve to the corpus. This is
                    what "resolution 1.0 by construction" promises, and it
                    must be exactly zero after verification.
      unmapped_repealed -- cites real, existing, but REPEALED law with no
                    recorded successor. V3 annotates these by design
                    (CLAUDE.md invariant 2); counting them as "bad" would
                    measure mapping coverage, not the verifier.
    """
    total, bad, unmapped, _ = citation_stats(answer, chunks, db, ver.cases)
    return total, bad, unmapped


def _run_condition(name, answers, chunks_by_qid, db, ver):
    print(f"\n{name.upper()}")
    print("-" * 78)
    rows = []
    for qid, answer in answers.items():
        chunks = chunks_by_qid.get(qid, [])
        total, bad, unmapped = _citation_stats(answer, chunks, db, ver)
        out, rep = ver.verify(answer, chunks)
        post_total, post_bad, post_unmapped = _citation_stats(
            strip_annotations(out), chunks, db, ver)
        rows.append({"qid": qid, "pre_total": total, "pre_bad": bad,
                    "pre_unmapped": unmapped,
                    "post_total": post_total, "post_bad": post_bad,
                    "post_unmapped": post_unmapped,
                    "coverage": rep.coverage, "findings": len(rep.findings)})
        print(f"  {qid}  pre: {total-bad}/{total} resolve ({unmapped} unmapped-repealed)   "
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
    print(f"\n  PRE-verifier resolution accuracy : {pre_acc:.3f}  "
          f"({pre_total-pre_bad}/{pre_total})")
    print(f"  POST-verifier resolution accuracy: {post_acc:.3f}  "
          f"({post_total-post_bad}/{post_total})  <- the core guarantee (V1/V2/V4)")
    print(f"  Repealed citations annotated but not stripped (no recorded "
          f"successor, V3 by design): {post_unmapped}")
    mean_cov = sum(r["coverage"] for r in rows) / len(rows)
    print(f"  Mean coverage          : {mean_cov:.3f}")
    return {"rows": rows, "pre_accuracy": pre_acc, "post_accuracy": post_acc,
            "pre_total": pre_total, "pre_bad": pre_bad,
            "post_total": post_total, "post_bad": post_bad,
            "post_unmapped_repealed": post_unmapped,
            "mean_coverage": mean_cov}


def report():
    db = build_real_db()
    ver = CitationVerifier(db, repair=True, support=False,
                           cases=build_case_index())

    print("PRE/POST-VERIFIER HALLUCINATION COMPARISON")
    print("=" * 78)
    print("See eval/generation_answers.py for the full methodological caveat:")
    print("these are real, unscripted answers, generated by this project's own")
    print("model rather than an independently invoked API call.")

    results = {}
    results["ungrounded"] = _run_condition("ungrounded", UNGROUNDED_ANSWERS, {}, db, ver)

    chunks_by_qid = _retrieve_chunks_for_gold(db)
    results["grounded"] = _run_condition("grounded", GROUNDED_ANSWERS,
                                        chunks_by_qid, db, ver)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("-" * 78)
    for cond in ("ungrounded", "grounded"):
        r = results[cond]
        print(f"  {cond:12} pre={r['pre_accuracy']:.3f}  "
              f"post={r['post_accuracy']:.3f}  coverage={r['mean_coverage']:.3f}")

    return results


if __name__ == "__main__":
    report()
