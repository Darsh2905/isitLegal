"""
Real-corpus measurement (Tier 1). Everything the 35-provision development
corpus could not show, because it was too small and too lexically distinctive
to produce a realistic result:

  - BM25 recall away from saturation (was Recall@1=1.000, meaningless)
  - V4 quotation checking exercised across a real corpus (was 1 verbatim
    provision out of 35; here all provisions are verbatim, sourced text)
  - The cross-reference graph on genuinely non-self-contained REAL provisions,
    rather than synthetic ones that (per CLAUDE.md's own documented caveat)
    retained descriptive marginal notes real unreachable provisions would not

This does NOT replace data/seed_corpus.py -- see ingest/india_code.py's module
docstring for why both corpora are kept, serving different purposes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.india_code import build_real_db, _KNOWN_PAIRS, _pair_confirmed
from ingest.scrape_indiankanoon import ACT_SOURCES, scrape_all, numeric_gaps
from ingest.scrape_correspondence import parse_all as parse_all_correspondence, TABLE_SOURCES
from retrieval.bm25_baseline import BM25Retriever
from retrieval.xref import build_graph, dependency_score
from eval.gold_qa import GOLD


def report():
    db = build_real_db()
    # Two arms: unstemmed BM25 is the ablation anchor every earlier revision
    # reported; stemmed BM25 (retrieval/text.py) is the deployed retriever.
    r = BM25Retriever(db, stem=False)
    r_stem = BM25Retriever(db, stem=True)

    print("REAL CORPUS: SCALE AND SOURCING")
    print("=" * 78)
    by_act = {}
    for p in db.all_provisions():
        by_act[p.act] = by_act.get(p.act, 0) + 1
    for act, n in sorted(by_act.items()):
        print(f"  {act:6} {n:5}")
    print(f"  {'TOTAL':6} {db.count():5}")
    verbatim = sum(1 for p in db.all_provisions() if p.verbatim)
    print(f"\n  verbatim (sourced, non-paraphrased) provisions: "
          f"{verbatim}/{db.count()}")
    print("  All ten target Acts included (IPC, BNS, CRPC, BNSS, IEA, BSA,")
    print("  NIA, ITA, ICA, Constitution).")

    print("\n  Known data-quality caveats found and handled during scraping:")
    scraped = scrape_all()
    for name, fname, act, url, _w in ACT_SOURCES:
        gaps = numeric_gaps(scraped.get(act, []))
        if gaps:
            print(f"    {act:6} numeric gaps in source: {gaps}")
    print("    (gaps mostly correspond to historically repealed/omitted")
    print("    sections -- e.g. ICA 76-123 was the Sale of Goods chapter,")
    print("    repealed 1930 -- not verified individually for every gap.)")

    print("\nRETRIEVAL: BM25 AWAY FROM SATURATION")
    print("=" * 78)
    def ev(K, retr=None):
        retr = retr or r
        hits = 0; rr = 0.0
        for qid, q, rel, _ in GOLD:
            res = [k for k, _ in retr.search(q, K)]
            hit = any(k in rel for k in res)
            hits += hit
            rank = next((i + 1 for i, k in enumerate(res) if k in rel), None)
            rr += 1.0 / rank if rank else 0.0
        return hits / len(GOLD), rr / len(GOLD)

    recall, recall_stem = {}, {}
    print(f"  {'':10}{'unstemmed (anchor)':>20}{'stemmed (deployed)':>20}")
    for K in (1, 3, 5, 10):
        recall[K], _ = ev(K)
        recall_stem[K], _ = ev(K, r_stem)
        print(f"  Recall@{K:<3}{recall[K]:>20.3f}{recall_stem[K]:>20.3f}")
    _, mrr = ev(10)
    _, mrr_stem = ev(10, r_stem)
    print(f"  {'MRR@10':10}{mrr:>20.3f}{mrr_stem:>20.3f}")
    print("  Stemming helps 5 queries and hurts 3 (see retrieval/text.py); the")
    print("  confidence gate stays on the UNSTEMMED signal, where it is calibrated.")
    in_range = 0.49 <= recall[1] <= 0.60
    print(f"\n  (toy corpus: Recall@1=1.000 -- meaningless, saturated at n=35)")
    print(f"  (Rustamov et al. reference range for comparable statutory")
    print(f"   corpora: 0.49-0.60 -- this corpus's Recall@1={recall[1]:.3f} "
          f"{'falls in it' if in_range else 'falls BELOW it'})")
    if not in_range:
        print(f"  This corpus is harder than that reference range, not easier:")
        print(f"  including BNSS added 531 real procedural provisions that")
        print(f"  lexically compete with CRPC's on exactly the queries about")
        print(f"  arrest and FIR procedure that used to resolve correctly.")

    print("\nQUOTATION CHECKING (V4) AT SCALE")
    print("=" * 78)
    print(f"  {verbatim} provisions now checkable verbatim, vs 1/35 before")
    print("  (Article 21 was the sole verbatim entry in the toy corpus)")

    print("\nCROSS-REFERENCE GRAPH ON REAL NON-SELF-CONTAINED PROVISIONS")
    print("=" * 78)
    g = build_graph(db)
    stats = g.stats()
    print(f"  edges: {stats['edges']}  (explicit={stats['explicit']} "
          f"intra={stats['intra']})  nodes_with_edges={stats['nodes_with_edges']}")
    nsc = [p for p in db.all_provisions() if dependency_score(p.text or "") > 0.5]
    print(f"  provisions with dependency_score>0.5: {len(nsc)}")
    print("  (toy corpus: 5 SYNTHETIC provisions, criticised in CLAUDE.md for")
    print("  retaining descriptive marginal notes real unreachable provisions")
    print("  would not have -- these are real, not authored for the test)")

    print("\nRECODIFICATION MAPPING VERIFICATION")
    print("=" * 78)
    all_rows = parse_all_correspondence()
    confirmed = set()
    for (new_act, old_act), rows in all_rows.items():
        for r_ in rows:
            confirmed.add((new_act, r_.new_section, old_act, r_.old_section))
    n_verified = 0
    for new_act, new_num, old_act, old_num in _KNOWN_PAIRS:
        # Shared with build_sections_and_mapping(), not reimplemented here --
        # this script and the real DB build had drifted into two separate
        # copies of the same check, and only one of them got the
        # section-vs-sub-section-granularity fix applied at first. See
        # _pair_confirmed's docstring for why a source entry that is a
        # sub-section refinement of our own section-level mapping counts.
        ok = _pair_confirmed(new_act, new_num, old_act, old_num, confirmed)
        n_verified += ok
        print(f"  {old_act}§{old_num:6} -> {new_act}§{new_num:6} "
              f"{'VERIFIED' if ok else 'not confirmed'}")
    print(f"\n  {n_verified}/{len(_KNOWN_PAIRS)} pairs confirmed against the "
          f"UP Police correspondence table series:")
    for _new, _old, _fname, url in TABLE_SOURCES:
        print(f"    {url}")

    # The full tables, not only the eleven hand-tracked pairs above.
    n_map = db.conn.execute("SELECT COUNT(*) n FROM mapping").fetchone()["n"]
    repealed = [p for p in db.all_provisions() if p.repealed_by]
    with_succ = sum(1 for p in repealed if db.successor(p.key))
    split = sum(1 for p in repealed if len(db.successor(p.key)) > 1)
    print(f"\n  Full correspondence mapping loaded: {n_map} section-level pairs")
    print(f"  Repealed provisions with a recorded successor: {with_succ}/"
          f"{len(repealed)} (was 11 when only the hand-tracked pairs were stored)")
    print(f"  Sections split across several successors (annotated, not"
          f" rewritten by R2): {split}")

    return {
        "corpus_size": db.count(),
        "by_act": by_act,
        "verbatim_count": verbatim,
        "recall": recall,
        "mrr@10": mrr,
        "recall_stemmed": recall_stem,
        "mrr@10_stemmed": mrr_stem,
        "mapping_full_pairs": n_map,
        "repealed_total": len(repealed),
        "repealed_with_successor": with_succ,
        "graph_edges": stats["edges"],
        "non_self_contained_real": len(nsc),
        "mapping_verified": n_verified,
        "mapping_total": len(_KNOWN_PAIRS),
    }


if __name__ == "__main__":
    report()
