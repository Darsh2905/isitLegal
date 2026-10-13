"""
Second live generation round: three independently invoked models, 66 queries
(50 externally authored AILA 2019 fact situations + the 16 gold questions),
grounded and ungrounded. See eval/capture_live_round2.py for what was captured
and why; this script re-derives everything from the frozen completions.

REPORTED, per model and condition:
  - citation resolution before and after verification, from the shared
    independent audit (eval/citation_audit.py), with 95% Wilson intervals;
  - coverage (fraction of sentences kept);
  - what was stripped, by finding type;
  - the fail-closed residual scan's trigger rate, with every instance listed
    (a scan that fires on harmless text costs coverage; the instances are
    printed so that cost can be read, not inferred);
  - the same figures restricted to queries the deployed gate lets through.

SUPPORT, as distinct from RESOLUTION, is measured separately on a labelled
sample of this round's output: eval/run_support_labels_eval.py.
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "live_round2_answers.json")
REAL_CORPUS_TAU = 1.077


def retrieval_stack(db):
    """The deployed retrieval configuration (fusion of stemmed BM25 and
    passage-window MiniLM; the caller reranks and applies in-force
    preference)."""
    from retrieval.bm25_baseline import BM25Retriever
    from retrieval.dense import DenseEncoder, DenseRetriever, MODEL_IDS
    from retrieval.fusion import FusedRetriever
    bm25 = BM25Retriever(db)
    dense = DenseRetriever(db, DenseEncoder(MODEL_IDS["general"]))
    return FusedRetriever([("bm25", bm25), ("dense", dense)], candidate_k=20)


def gate_for(db):
    from retrieval.bm25_baseline import BM25Retriever
    from retrieval.confidence import ConfidenceGate
    return ConfidenceGate(db, BM25Retriever(db, stem=False), tau=REAL_CORPUS_TAU)


ADVISORY = {"repealed_provision", "invalidated_provision", "weak_support",
            "support_via_reference", "marker_mismatch"}


def load():
    with open(FROZEN, encoding="utf-8") as f:
        return json.load(f)


def verify_all(data=None):
    """Verify every frozen answer. Returns one row per answer."""
    from ingest.india_code import build_real_db
    from ingest.case_citations import build_case_index
    from verify.verifier import CitationVerifier, Chunk
    from eval.citation_audit import citation_stats, strip_annotations
    data = data or load()
    db = build_real_db()
    cases = build_case_index()
    ver = CitationVerifier(db, repair=True, support=True, cases=cases)
    rows = []
    for k, a in sorted(data["answers"].items()):
        if a["text"] is None:
            continue
        retr = data["retrieval"][a["qid"]]
        if a["condition"] == "grounded":
            chunks = [Chunk(i, ck, db.get(ck).text, db.get(ck).verbatim)
                      for i, ck in enumerate(retr["chunk_keys"], 1)]
        else:
            chunks = []
        text = a["text"]
        pre_t, pre_b, _, pre_forms = citation_stats(text, chunks, db, cases)
        out, rep = ver.verify(text, chunks)
        post_t, post_b, _, _ = citation_stats(strip_annotations(out), chunks,
                                              db, cases)
        rows.append({
            "key": k, "model": a["model"], "condition": a["condition"],
            "qid": a["qid"], "source": a["source"],
            "gate_pass": retr["gate_pass"],
            "truncated": a["finish_reason"] == "length",
            "pre_total": pre_t, "pre_bad": pre_b, "pre_forms": pre_forms,
            "post_total": post_t, "post_bad": post_b,
            "coverage": rep.coverage,
            "sentences": rep.sentences_total,
            "stripped": rep.sentences_stripped,
            "findings": [(f.violation.value, f.span, f.detail)
                         for f in rep.findings],
            "residual": [(f.span, f.detail) for f in rep.findings
                         if f.violation.value == "unparsed_reference"],
            "output": out,
        })
    return rows


def summarise(rows):
    from eval.stats import wilson
    pre_t = sum(r["pre_total"] for r in rows)
    pre_b = sum(r["pre_bad"] for r in rows)
    post_t = sum(r["post_total"] for r in rows)
    post_b = sum(r["post_bad"] for r in rows)
    sent = sum(r["sentences"] for r in rows)
    strp = sum(r["stripped"] for r in rows)
    resid_sent = sum(1 for r in rows for _ in r["residual"])
    stripped_by = Counter(f[0] for r in rows for f in r["findings"]
                          if f[0] not in ADVISORY)
    advisory_by = Counter(f[0] for r in rows for f in r["findings"]
                          if f[0] in ADVISORY)
    return {
        "answers": len(rows),
        "pre_total": pre_t, "pre_bad": pre_b,
        "pre_resolution": 1 - pre_b / pre_t if pre_t else 1.0,
        "pre_ci": wilson(pre_t - pre_b, pre_t),
        "post_total": post_t, "post_bad": post_b,
        "post_resolution": 1 - post_b / post_t if post_t else 1.0,
        "post_ci": wilson(post_t - post_b, post_t),
        "sentences": sent, "sentences_stripped": strp,
        "coverage": 1 - strp / sent if sent else 1.0,
        "mean_answer_coverage": (sum(r["coverage"] for r in rows) / len(rows)
                                 if rows else 1.0),
        "residual_hits": resid_sent,
        "residual_rate": resid_sent / sent if sent else 0.0,
        "stripped_by": dict(stripped_by), "advisory_by": dict(advisory_by),
        "truncated": sum(r["truncated"] for r in rows),
    }


def report():
    data = load()
    rows = verify_all(data)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"])].append(r)
        groups[("ALL", r["condition"])].append(r)

    print("LIVE ROUND 2: three models, 66 queries (50 AILA + 16 gold)")
    print("=" * 78)
    missing = [k for k, a in data["answers"].items() if a["text"] is None]
    print(f"  answers captured: {len(rows)}   failed/too large: {len(missing)}")
    res = {}
    for key in sorted(groups):
        s = summarise(groups[key])
        res[key] = s
        m, c = key
        print(f"\n  {m}  [{c}]  answers={s['answers']}  truncated={s['truncated']}")
        print(f"    resolution  pre  {s['pre_resolution']:.3f} "
              f"({s['pre_total']-s['pre_bad']}/{s['pre_total']}, CI "
              f"{s['pre_ci'][0]:.3f}-{s['pre_ci'][1]:.3f})")
        print(f"                post {s['post_resolution']:.3f} "
              f"({s['post_total']-s['post_bad']}/{s['post_total']}, CI "
              f"{s['post_ci'][0]:.3f}-{s['post_ci'][1]:.3f})")
        print(f"    coverage    {s['coverage']:.3f} of {s['sentences']} sentences "
              f"(mean per answer {s['mean_answer_coverage']:.3f})")
        print(f"    stripped by {s['stripped_by']}")
        print(f"    advisory    {s['advisory_by']}")
        print(f"    residual scan fired on {s['residual_hits']} sentence(s) "
              f"({s['residual_rate']:.4f} of sentences)")

    gated = [r for r in rows if r["condition"] == "grounded" and r["gate_pass"]]
    res[("ALL", "grounded-gate-pass")] = summarise(gated)
    s = res[("ALL", "grounded-gate-pass")]
    print(f"\n  GROUNDED, only queries the deployed gate passes: answers="
          f"{s['answers']}  resolution {s['pre_resolution']:.3f} -> "
          f"{s['post_resolution']:.3f}  coverage {s['coverage']:.3f}")

    print("\n  EVERY RESIDUAL-SCAN INSTANCE (fail-closed strips):")
    for r in rows:
        for span, detail in r["residual"]:
            print(f"    {r['key'][:48]:48} {span!r}")
    return {"summary": res, "rows": rows, "missing": len(missing)}


if __name__ == "__main__":
    report()
