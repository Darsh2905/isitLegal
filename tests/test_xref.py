"""Cross-reference graph C5 (paper VII-F)."""


def test_graph_is_built(xref):
    stats = xref["graph_stats"]
    assert stats["edges"] == 7
    assert stats["nodes_with_edges"] == 12


def test_graph_expansion_recovers_non_self_contained_provisions(xref_sweep):
    # The whole point of C5: at K=1 the graph reaches provisions whose own text
    # is too thin for lexical retrieval.
    assert xref_sweep["sweep"][1]["gain"] >= 0.20 - 1e-9  # 1.0 - 0.8 in floating point


def test_expansion_never_degrades_ordinary_retrieval(xref, xref_sweep):
    # Regression guard: buying xref recall by hurting the ordinary queries
    # would be a bad trade that a single-number report would hide.
    assert (xref["ordinary_recall@10"]["graph"]
            >= xref["ordinary_recall@10"]["bm25"])
    for K, row in xref_sweep["sweep"].items():
        assert row["ordinary_graph"] >= row["ordinary_bm25"], f"regression at K={K}"


def test_context_cost_stays_bounded(xref):
    # Expansion must not balloon the generator's context.
    cost = xref["candidate_set_size"]
    assert cost["graph"] <= cost["bm25"] * 1.5
