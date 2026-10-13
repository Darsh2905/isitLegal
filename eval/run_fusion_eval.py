"""RRF fusion of BM25 + dense retrieval, on the real corpus."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.india_code import build_real_db
from retrieval.bm25_baseline import BM25Retriever
from retrieval.dense import DenseEncoder, DenseRetriever, MODEL_IDS
from retrieval.fusion import FusedRetriever
from eval.gold_qa import GOLD


def _eval(retriever):
    out = {}
    for K in (1, 3, 5, 10):
        hits = 0
        for _, q, rel, _ in GOLD:
            res = [k for k, _ in retriever.search(q, K)]
            hits += any(k in rel for k in res)
        out[K] = hits / len(GOLD)

    rr_total = 0.0
    for _, q, rel, _ in GOLD:
        res = [k for k, _ in retriever.search(q, 10)]
        rank = next((i + 1 for i, k in enumerate(res) if k in rel), None)
        rr_total += 1.0 / rank if rank else 0.0
    out["mrr@10"] = rr_total / len(GOLD)
    return out


def report():
    print("RRF FUSION: BM25 + DENSE (real corpus)")
    print("=" * 78)
    db = build_real_db()
    bm25 = BM25Retriever(db)
    enc = DenseEncoder(MODEL_IDS["general"])
    dense = DenseRetriever(db, enc)
    # candidate_k=20: the DEPLOYED depth (pipeline, app, rerank eval). This
    # table previously fused each arm's top 50 (the class default) while the
    # reranking table fused each arm's top 20, so the paper's two "RRF fusion"
    # rows disagreed (MRR@10 0.690 vs 0.676) with no stated reason -- a
    # reviewer caught it. Both now measure the one deployed configuration.
    fused = FusedRetriever([("bm25", bm25), ("dense", dense)], candidate_k=20)

    results = {"bm25": _eval(bm25), "dense": _eval(dense), "bm25+rrf+dense": _eval(fused)}

    print(f"{'arm':16}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'R@10':>8}{'MRR@10':>9}")
    print("-" * 60)
    for arm, r in results.items():
        print(f"{arm:16}{r[1]:>7.3f}{r[3]:>7.3f}{r[5]:>7.3f}{r[10]:>8.3f}{r['mrr@10']:>9.3f}")

    return {"corpus_size": db.count(), "results": results}


if __name__ == "__main__":
    report()
