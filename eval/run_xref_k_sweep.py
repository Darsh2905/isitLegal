import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import build_seed_db
from retrieval.bm25_baseline import BM25Retriever
from retrieval.xref import build_graph
from eval.xref_queries import XREF_GOLD
from eval.gold_qa import GOLD


def report():
    db = build_seed_db(); r = BM25Retriever(db); g = build_graph(db)
    N = db.count()

    def base(q,K):  return [k for k,_ in r.search(q,K)]
    def aug(q,K,hops=1):
        seeds = base(q,K); out=list(seeds)
        for k,d,v in g.expand(seeds,hops=hops,direction="in"):
            if k not in out: out.append(k)
        return out

    def recall(gold_set, fn, K):
        h=0
        for item in gold_set:
            q, rel = item[1], item[2]
            if any(k in rel for k in fn(q,K)): h+=1
        return h/len(gold_set)

    print(f"Corpus = {N} provisions.  At K=10 the retriever returns "
          f"{100*10/N:.0f}% of the whole corpus, so only small K is informative.\n")
    print("CROSS-REFERENCE QUERIES (n=5)        ORDINARY QUERIES (n=16)")
    print("="*74)
    print(f"{'K':>3} {'BM25':>8}{'+graph':>9}{'gain':>7}   |{'BM25':>9}{'+graph':>9}{'ctx':>7}")
    print("-"*74)
    sweep = {}
    for K in (1,2,3,5):
        xb, xa = recall(XREF_GOLD,base,K), recall(XREF_GOLD,aug,K)
        ob, oa = recall(GOLD,base,K),      recall(GOLD,aug,K)
        ctx = sum(len(aug(q,K)) for _,q,_,_ in XREF_GOLD)/len(XREF_GOLD)
        sweep[K] = {"xref_bm25": xb, "xref_graph": xa, "gain": xa-xb,
                    "ordinary_bm25": ob, "ordinary_graph": oa, "ctx": ctx}
        print(f"{K:>3} {xb:>8.3f}{xa:>9.3f}{xa-xb:>+7.3f}   |{ob:>9.3f}{oa:>9.3f}"
              f"{ctx:>7.1f}")
    print("-"*74)
    print("gain = Recall@K improvement on provisions whose own text is too thin")
    print("       to be retrieved;  ctx = mean candidate-set size after expansion")

    return {"corpus_size": N, "sweep": sweep}


if __name__ == "__main__":
    report()
