"""
BM25 baseline (paper VII-I).

NOTE: Recall@1 = 1.000 on a 35-provision corpus is SATURATION, not a result
(CLAUDE.md Section 5). These assertions exist to detect a retrieval regression,
not to certify retrieval quality. Do not cite them as accuracy.
"""


def test_corpus_and_query_set_are_the_reported_ones(retrieval):
    assert retrieval["corpus_size"] == 35
    assert retrieval["n_queries"] == 16


def test_no_regression_on_saturated_benchmark(retrieval):
    assert retrieval["recall"][1] == 1.0, retrieval["misses"]
    assert retrieval["mrr@10"] == 1.0


def test_recall_is_monotone_in_k(retrieval):
    ks = sorted(retrieval["recall"])
    vals = [retrieval["recall"][k] for k in ks]
    assert vals == sorted(vals)
