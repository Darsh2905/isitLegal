import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.metadata_db import build_seed_db
from retrieval.bm25_baseline import BM25Retriever
from retrieval.xref import build_graph
from eval.xref_queries import XREF_GOLD
from eval.gold_qa import GOLD


def report():
    db = build_seed_db(); r = BM25Retriever(db); g = build_graph(db)

    def baseline(q, K):
        return [k for k,_ in r.search(q, K)]

    def graph_augmented(q, K, hops=1):
        """First-stage retrieval, then admit inbound graph neighbours."""
        seeds = [k for k,_ in r.search(q, K)]
        expanded = list(seeds)
        for key, dist, via in g.expand(seeds, hops=hops, direction="in"):
            if key not in expanded:
                expanded.append(key)
        return expanded

    print("CROSS-REFERENCE RECOVERY  (gold = non-self-contained provisions)")
    print("="*78)
    print(f"{'qid':5}{'query':46}{'BM25':>8}{'+graph':>9}")
    print("-"*78)
    b_hits = ga_hits = 0
    b_ranks, ga_ranks = [], []
    for qid,q,gold,desc in XREF_GOLD:
        B, G = baseline(q,10), graph_augmented(q,10)
        bh = any(k in gold for k in B); gh = any(k in gold for k in G)
        b_hits += bh; ga_hits += gh
        br = next((i+1 for i,k in enumerate(B) if k in gold), None)
        gr = next((i+1 for i,k in enumerate(G) if k in gold), None)
        b_ranks.append(1/br if br else 0.0)
        ga_ranks.append(1/gr if gr else 0.0)
        print(f"{qid:5}{q[:44]:46}{('HIT@'+str(br)) if bh else 'MISS':>8}"
              f"{('HIT@'+str(gr)) if gh else 'MISS':>9}")
    print("-"*78)
    n=len(XREF_GOLD)
    b_recall, g_recall = b_hits/n, ga_hits/n
    b_mrr, g_mrr = sum(b_ranks)/n, sum(ga_ranks)/n
    print(f"Recall@10   BM25 = {b_recall:.3f}   BM25+graph = {g_recall:.3f}")
    print(f"MRR@10      BM25 = {b_mrr:.3f}   BM25+graph = {g_mrr:.3f}")

    # Regression check: does expansion hurt the ordinary queries?
    def rec(fn,K):
        h=0
        for _,q,rel,_ in GOLD:
            if any(k in rel for k in fn(q,K)): h+=1
        return h/len(GOLD)
    ord_b, ord_g = rec(baseline,10), rec(graph_augmented,10)
    print("\nREGRESSION on 16 ordinary gold queries (expansion must not hurt):")
    print(f"  Recall@10  BM25 = {ord_b:.3f}   BM25+graph = {ord_g:.3f}")

    avg_b = sum(len(baseline(q,10)) for _,q,_,_ in XREF_GOLD)/n
    avg_g = sum(len(graph_augmented(q,10)) for _,q,_,_ in XREF_GOLD)/n
    print(f"\nCandidate set size: {avg_b:.1f} -> {avg_g:.1f} "
          f"(+{100*(avg_g-avg_b)/avg_b:.0f}% context cost)")

    return {
        "n_xref_queries": n,
        "xref_recall@10": {"bm25": b_recall, "graph": g_recall},
        "xref_mrr@10": {"bm25": b_mrr, "graph": g_mrr},
        "ordinary_recall@10": {"bm25": ord_b, "graph": ord_g},
        "candidate_set_size": {"bm25": avg_b, "graph": avg_g},
        "graph_stats": g.stats(),
    }


if __name__ == "__main__":
    report()
