"""
Dense retrieval ablation: InLegalBERT (domain-adapted) vs a general-purpose
embedding model (sentence-transformers/all-MiniLM-L6-v2), against BM25.

Promised in the paper's architecture section as "a reportable experiment in
its own right" -- this is that experiment. Run on the REAL corpus
(ingest/india_code.py), not the 35-provision toy corpus: at toy-corpus size
every retriever saturates and the comparison is uninformative, exactly the
reason the toy corpus was flagged as unsuitable for retrieval measurement in
the first place.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.india_code import build_real_db
from retrieval.bm25_baseline import BM25Retriever
from retrieval.dense import DenseEncoder, DenseRetriever, MODEL_IDS
from eval.gold_qa import GOLD


def _eval(retriever, K_list=(1, 3, 5, 10)):
    out = {}
    for K in K_list:
        hits = 0; rr = 0.0
        for qid, q, rel, _ in GOLD:
            res = [k for k, _ in retriever.search(q, K)]
            hit = any(k in rel for k in res)
            hits += hit
            rank = next((i + 1 for i, k in enumerate(res) if k in rel), None)
            rr += 1.0 / rank if rank else 0.0
        out[K] = hits / len(GOLD)
    _, mrr_rows = None, None
    rr_total = 0.0
    for qid, q, rel, _ in GOLD:
        res = [k for k, _ in retriever.search(q, 10)]
        rank = next((i + 1 for i, k in enumerate(res) if k in rel), None)
        rr_total += 1.0 / rank if rank else 0.0
    out["mrr@10"] = rr_total / len(GOLD)
    return out


def report():
    print("DENSE RETRIEVAL ABLATION (real corpus)")
    print("=" * 78)
    db = build_real_db()
    print(f"corpus: {db.count()} provisions\n")

    print("Building retrievers (embedding the corpus once per arm)...")
    t0 = time.time()
    results = {"bm25": _eval(BM25Retriever(db, stem=False)),
               "bm25_stemmed": _eval(BM25Retriever(db, stem=True))}
    print(f"  BM25    ready in {time.time()-t0:.1f}s")

    # Each encoder is measured twice: whole-provision vectors (the original
    # arm, truncated at 256 tokens -- 21% of provisions are longer) and
    # max-pooled passage windows (retrieval/dense.py, now the default).
    for arm, model_id in MODEL_IDS.items():
        enc = DenseEncoder(model_id)
        for passages, suffix in ((False, ""), (True, "_passages")):
            t0 = time.time()
            dr = DenseRetriever(db, enc, passages=passages)
            build_t = time.time() - t0
            results[arm + suffix] = _eval(dr)
            results[arm + suffix]["build_seconds"] = build_t
            print(f"  {arm + suffix:18}({model_id}) ready in {build_t:.1f}s")

    print("\n" + "-" * 78)
    print(f"{'arm':20}{'Recall@1':>10}{'Recall@3':>10}{'Recall@5':>10}"
          f"{'Recall@10':>11}{'MRR@10':>9}")
    print("-" * 78)
    for arm in ("bm25", "bm25_stemmed", "domain", "domain_passages",
                "general", "general_passages"):
        r = results[arm]
        print(f"{arm:20}{r[1]:>10.3f}{r[3]:>10.3f}{r[5]:>10.3f}"
              f"{r[10]:>11.3f}{r['mrr@10']:>9.3f}")

    return {"corpus_size": db.count(), "results": {
        k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}}


if __name__ == "__main__":
    report()
