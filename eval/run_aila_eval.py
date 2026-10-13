"""
AILA 2019 Task 2 (statute retrieval) with the official relevance judgments --
the first evaluation in this project on a community benchmark rather than on
a query set it wrote itself (contribution C4, previously undelivered; raised
by the 2026-10 peer review).

DATA. FIRE 2019 AILA track (Bhattacharya et al., 2019), Zenodo record
4063986, unpacked in data/aila/: 50 queries (fact situations written by the
organisers from Supreme Court judgments), 197 statutes (title + description),
and binary relevance judgments for every query-statute pair, derived from the
statutes the lawyers in each case actually cited. AILA_Q1-Q10 were released as
training data; the official evaluation is over AILA_Q11-Q50, and that split is
used here. Nothing is tuned on these queries: every component runs with the
configuration fixed on this project's own corpus.

MEASURES. As trec_eval 8.1 (the track's evaluation tool) computes them, over
the full ranking of all 197 statutes: MAP, BPREF, reciprocal rank, P@10; plus
Recall@10 and nDCG@10 for comparison with this project's other tables.

COMPARISON. The track's best Task 2 run reached MAP 0.1566 (thuir_legal_2,
query summarisation + language model), reciprocal rank 0.281 and P@10 0.0975
(overview paper, Table 2). Those runs were unsupervised and so are ours; the
comparison is like for like in data, split and measure, but the track's runs
are from 2019 and pre-date the pretrained encoders used here.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AILA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "aila")
TEST = [f"AILA_Q{i}" for i in range(11, 51)]
TRACK_BEST = {"run": "thuir_legal_2", "map": 0.1566, "bpref": 0.0961,
              "recip_rank": 0.281, "p10": 0.0975}


def load():
    from ingest.metadata_db import MetadataDB
    statutes = []
    sdir = os.path.join(AILA, "Object_statutes")
    for fn in sorted(os.listdir(sdir), key=lambda f: int(f[1:-4])):
        lines = open(os.path.join(sdir, fn), encoding="utf-8",
                     errors="replace").read().splitlines()
        title = lines[0].split(":", 1)[1].strip() if lines else ""
        desc = lines[1].split(":", 1)[1].strip() if len(lines) > 1 else ""
        # act "AILA", section "S12": the key is AILA§S12.
        statutes.append(("AILA", fn[:-4], title, desc, None, True))
    db = MetadataDB()
    db.load_seed(statutes, [])
    queries = {}
    for line in open(os.path.join(AILA, "Query_doc.txt"), encoding="utf-8",
                     errors="replace"):
        if "||" in line:
            q, t = line.rstrip("\n").split("||", 1)
            queries[q.strip()] = t.strip()
    qrels = {}
    for line in open(os.path.join(AILA, "relevance_judgments_statutes.txt")):
        q, _, d, r = line.split()
        qrels.setdefault(q, {})[f"AILA§{d}"] = int(r)
    return db, queries, qrels


def trec_measures(ranking, judged):
    """ranking: list of keys, best first. judged: key -> 0/1."""
    rel = {k for k, v in judged.items() if v > 0}
    R, N = len(rel), sum(1 for v in judged.values() if v == 0)
    hits, ap, rr, nonrel_above, bpref = 0, 0.0, 0.0, 0, 0.0
    for i, k in enumerate(ranking, 1):
        if k in rel:
            hits += 1
            ap += hits / i
            if rr == 0.0:
                rr = 1.0 / i
            if R:
                bpref += 1 - min(nonrel_above, R) / min(R, N) if N else 1.0
        elif k in judged:
            nonrel_above += 1
    p10 = sum(1 for k in ranking[:10] if k in rel) / 10
    r10 = sum(1 for k in ranking[:10] if k in rel) / R if R else 0.0
    dcg = sum(1 / math.log2(i + 1) for i, k in enumerate(ranking[:10], 1) if k in rel)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(R, 10) + 1))
    return {"map": ap / R if R else 0.0, "bpref": bpref / R if R else 0.0,
            "recip_rank": rr, "p10": p10, "r10": r10,
            "ndcg10": dcg / idcg if idcg else 0.0}


def _mean(rows):
    return {m: sum(r[m] for r in rows) / len(rows) for m in rows[0]}


def systems(db):
    from retrieval.bm25_baseline import BM25Retriever
    from retrieval.dense import DenseEncoder, DenseRetriever, MODEL_IDS
    from retrieval.fusion import FusedRetriever
    from retrieval.rerank import Reranker
    n = db.count()
    bm25_u = BM25Retriever(db, stem=False)
    bm25_s = BM25Retriever(db)
    mini = DenseRetriever(db, DenseEncoder(MODEL_IDS["general"]))
    legal = DenseRetriever(db, DenseEncoder(MODEL_IDS["domain"]))
    fused = FusedRetriever([("bm25", bm25_s), ("dense", mini)], candidate_k=n)
    rer = Reranker(db)

    def full(r):
        return lambda q: [k for k, _ in r.search(q, n)]

    def fused_reranked(q):
        f = [k for k, _ in fused.search(q, n)]
        top = [k for k, _ in rer.rerank(q, [(k, 0.0) for k in f[:20]], top_k=20)]
        return top + [k for k in f if k not in top]

    return [("BM25", full(bm25_u)), ("BM25, stemmed", full(bm25_s)),
            ("InLegalBERT, passage windows", full(legal)),
            ("MiniLM, passage windows", full(mini)),
            ("RRF (stemmed BM25 + MiniLM)", full(fused)),
            ("RRF + passage rerank (top 20)", fused_reranked)]


def report():
    db, queries, qrels = load()
    print("AILA 2019 TASK 2 -- statute retrieval, official qrels, test queries "
          "AILA_Q11-Q50")
    print("=" * 78)
    print(f"  statutes {db.count()}   test queries {len(TEST)}   relevant pairs "
          f"{sum(sum(qrels[q].values()) for q in TEST)}")
    print(f"  {'system':32} {'MAP':>7} {'BPREF':>7} {'RR':>7} {'P@10':>7} "
          f"{'R@10':>7} {'nDCG10':>7}")
    out = {}
    for name, rank in systems(db):
        rows = [trec_measures(rank(queries[q]), qrels[q]) for q in TEST]
        m = _mean(rows)
        out[name] = m
        print(f"  {name:32} {m['map']:7.4f} {m['bpref']:7.4f} {m['recip_rank']:7.4f} "
              f"{m['p10']:7.4f} {m['r10']:7.4f} {m['ndcg10']:7.4f}")
    b = TRACK_BEST
    print(f"  {'track best (' + b['run'] + ')':32} {b['map']:7.4f} {b['bpref']:7.4f} "
          f"{b['recip_rank']:7.4f} {b['p10']:7.4f}")
    return {"systems": out, "track_best": TRACK_BEST, "n_test": len(TEST)}


if __name__ == "__main__":
    report()
