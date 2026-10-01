import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import build_seed_db
from retrieval.bm25_baseline import BM25Retriever
from eval.gold_qa import GOLD


def report():
    db = build_seed_db(); r = BM25Retriever(db)
    print(f"Corpus: {db.count()} provisions   Queries: {len(GOLD)}\n")

    def ev(K):
        hits=0; rr=0.0; per=[]
        for qid,q,rel,_ in GOLD:
            res=[k for k,_ in r.search(q,K)]
            hit=any(k in rel for k in res)
            hits+=hit
            rank=next((i+1 for i,k in enumerate(res) if k in rel), None)
            rr += 1.0/rank if rank else 0.0
            per.append((qid,hit,rank))
        return hits/len(GOLD), rr/len(GOLD), per

    print(f"{'metric':16} {'value':>7}")
    print("-"*25)
    recall = {}
    for K in (1,3,5,10):
        rec,mrr,_ = ev(K)
        recall[K] = rec
        print(f"Recall@{K:<9} {rec:>7.3f}")
    _,mrr,per = ev(10)
    print(f"{'MRR@10':16} {mrr:>7.3f}")

    print("\nPer-query (K=10):")
    print("-"*46)
    for qid,hit,rank in per:
        q = next(g[1] for g in GOLD if g[0]==qid)
        print(f"  {qid} {'HIT ' if hit else 'MISS'} rank={str(rank):>4}  {q[:44]}")

    print("\nMisses in detail:")
    misses = []
    for qid,hit,rank in per:
        if not hit:
            g = next(x for x in GOLD if x[0]==qid)
            misses.append(qid)
            print(f"  {qid}: {g[1]}")
            print(f"     gold={g[2]}")
            print(f"     got ={[k for k,_ in r.search(g[1],5)]}")

    return {"corpus_size": db.count(), "n_queries": len(GOLD),
            "recall": recall, "mrr@10": mrr, "misses": misses,
            "per_query": per}


if __name__ == "__main__":
    report()
