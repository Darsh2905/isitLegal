"""
Naive fixed-window chunking vs. paragraph-level parent-document chunking,
measured against the real judgment sample (ingest/scrape_judgments.py) and
the gold query set in eval/judgment_queries.py.

This is CLAUDE.md Tier 1 item 6's "unlocks testing the RAPTOR-style
long-document design decision that's currently justified only by citation
to other papers, not by our own measurement" -- run for the first time
here. A "hit" for either strategy is the gold paragraph falling inside the
matched chunk's paragraph range (exact-paragraph equality for
parent-document chunks, since each chunk covers exactly one paragraph;
range membership for naive chunks, since each covers `window` paragraphs).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.scrape_judgments import scrape_all
from retrieval.judgment_retrieval import JudgmentRetriever
from eval.judgment_queries import GOLD


def _evaluate(retriever, K):
    hits = 0
    rr = 0.0
    per = []
    for query, doc_id, para_idx, case_name in GOLD:
        results = retriever.search(query, K)
        rank = None
        for i, (chunk, _score) in enumerate(results):
            if chunk.doc_id == doc_id and chunk.para_start <= para_idx <= chunk.para_end:
                rank = i + 1
                break
        hit = rank is not None
        hits += hit
        rr += 1.0 / rank if rank else 0.0
        per.append((query, doc_id, case_name, hit, rank))
    return hits / len(GOLD), rr / len(GOLD), per


def report():
    judgments = scrape_all()
    print(f"Judgment corpus: {len(judgments)} real Supreme Court judgments "
          f"(ingest/scrape_judgments.py)   Queries: {len(GOLD)}\n")

    strategies = {}
    for strategy in ("naive", "parent_doc"):
        retriever = JudgmentRetriever(judgments, strategy=strategy)
        n_chunks = len(retriever.chunks)
        recall = {}
        per = None
        for K in (1, 3, 5, 10):
            rec, mrr, per_k = _evaluate(retriever, K)
            recall[K] = rec
            if K == 10:
                per = per_k
                mrr10 = mrr
        strategies[strategy] = {
            "n_chunks": n_chunks, "recall": recall, "mrr@10": mrr10, "per": per,
        }

        label = "NAIVE (fixed 4-paragraph windows)" if strategy == "naive" \
            else "PARENT-DOCUMENT (one paragraph per chunk)"
        print(f"{label}")
        print(f"  chunks indexed: {n_chunks}")
        for K in (1, 3, 5, 10):
            print(f"  Recall@{K:<2} {recall[K]:.3f}")
        print(f"  MRR@10   {mrr10:.3f}")
        print()

    naive_r1 = strategies["naive"]["recall"][1]
    pd_r1 = strategies["parent_doc"]["recall"][1]
    print(f"Recall@1 delta (parent-document minus naive): {pd_r1 - naive_r1:+.3f}")
    print(f"Chunk-count ratio (naive / parent-document): "
          f"{strategies['naive']['n_chunks']} / {strategies['parent_doc']['n_chunks']} = "
          f"{strategies['naive']['n_chunks'] / strategies['parent_doc']['n_chunks']:.3f}")

    print("\nPer-query misses (parent-document strategy):")
    for query, doc_id, case_name, hit, rank in strategies["parent_doc"]["per"]:
        if not hit:
            print(f"  MISS  {case_name[:50]:50} doc={doc_id}  query={query[:50]}")

    return {
        "n_judgments": len(judgments),
        "n_queries": len(GOLD),
        "naive": {"n_chunks": strategies["naive"]["n_chunks"],
                  "recall": strategies["naive"]["recall"],
                  "mrr@10": strategies["naive"]["mrr@10"]},
        "parent_doc": {"n_chunks": strategies["parent_doc"]["n_chunks"],
                       "recall": strategies["parent_doc"]["recall"],
                       "mrr@10": strategies["parent_doc"]["mrr@10"]},
        "recall1_delta": pd_r1 - naive_r1,
    }


if __name__ == "__main__":
    report()
