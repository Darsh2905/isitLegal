"""
Cross-encoder reranking, on top of RRF-fused BM25+dense candidates.

Reports BOTH MRR and nDCG@10, per the paper's explicit commitment (following
Rustamov et al.'s finding that reranking can improve one while degrading the
other) -- never report just whichever looks better.
"""
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.india_code import build_real_db
from retrieval.bm25_baseline import BM25Retriever
from retrieval.dense import DenseEncoder, DenseRetriever, MODEL_IDS
from retrieval.fusion import FusedRetriever
from retrieval.inforce import prefer_in_force
from retrieval.rerank import Reranker
from eval.gold_qa import GOLD


def _mrr(rankings_by_query, gold_by_query):
    total = 0.0
    for qid, ranking in rankings_by_query.items():
        rel = gold_by_query[qid]
        rank = next((i + 1 for i, k in enumerate(ranking) if k in rel), None)
        total += 1.0 / rank if rank else 0.0
    return total / len(rankings_by_query)


def _ndcg_at_k(ranking, rel, k=10):
    dcg = 0.0
    for i, key in enumerate(ranking[:k]):
        if key in rel:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def report():
    print("CROSS-ENCODER RERANKING (real corpus)")
    print("=" * 78)
    db = build_real_db()
    bm25 = BM25Retriever(db)
    enc = DenseEncoder(MODEL_IDS["general"])
    dense = DenseRetriever(db, enc)
    fused = FusedRetriever([("bm25", bm25), ("dense", dense)], candidate_k=20)
    reranker = Reranker(db)  # passage windows (deployed)
    reranker_whole = Reranker(db, passages=False)  # original arm

    fused_rank, whole_rank, rerank_rank, inforce_rank = {}, {}, {}, {}
    gold_by_query = {}
    for qid, q, rel, _ in GOLD:
        gold_by_query[qid] = rel
        cands = fused.search(q, 20)
        fused_rank[qid] = [k for k, _ in cands]
        whole_rank[qid] = [k for k, _ in reranker_whole.rerank(q, cands, top_k=10)]
        reranked = reranker.rerank(q, cands, top_k=10)
        rerank_rank[qid] = [k for k, _ in reranked]
        # The deployed stack's last retrieval step (retrieval/inforce.py).
        inforce_rank[qid] = [k for k, _ in prefer_in_force(reranked, db)]

    results = {}
    for label, ranks in (("bm25+rrf+dense", fused_rank),
                         ("+ rerank (whole)", whole_rank),
                         ("+ rerank", rerank_rank),
                         ("+ in-force pref.", inforce_rank)):
        mrr = _mrr(ranks, gold_by_query)
        ndcgs = [_ndcg_at_k(ranks[qid], gold_by_query[qid]) for qid in ranks]
        ndcg = sum(ndcgs) / len(ndcgs)
        hits1 = sum(1 for qid in ranks if ranks[qid] and ranks[qid][0] in gold_by_query[qid])
        results[label] = {"mrr@10": mrr, "ndcg@10": ndcg, "recall@1": hits1 / len(ranks)}
        print(f"  {label:16} MRR@10={mrr:.3f}  nDCG@10={ndcg:.3f}  "
              f"Recall@1={hits1/len(ranks):.3f}")

    d_mrr = results["+ rerank"]["mrr@10"] - results["bm25+rrf+dense"]["mrr@10"]
    d_ndcg = results["+ rerank"]["ndcg@10"] - results["bm25+rrf+dense"]["ndcg@10"]
    print(f"\n  delta MRR@10:  {d_mrr:+.3f}")
    print(f"  delta nDCG@10: {d_ndcg:+.3f}")
    if (d_mrr > 0) != (d_ndcg > 0):
        print("  MRR and nDCG moved in OPPOSITE directions -- reporting both "
              "was not optional here.")

    return {"corpus_size": db.count(), "results": results}


if __name__ == "__main__":
    report()
