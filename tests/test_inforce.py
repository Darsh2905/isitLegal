"""Recodification diagnostic (paper VII-I)."""


def test_every_repealed_citation_is_annotated_with_its_successor(inforce):
    # V3's guarantee: the user is not misled by a repealed hit regardless of
    # where it ranks, because the annotation always fires.
    assert inforce["annotated"] == inforce["n_pairs"]
    assert inforce["annotated_rate"] == 1.0


def test_lexical_retrieval_cannot_discriminate_in_force_law(inforce):
    # A DOCUMENTED WEAKNESS, asserted so it cannot silently change. BM25 ranks
    # the in-force provision above its repealed predecessor in only 3 of 9
    # cases; this is the motivation for contribution C2. If this improves,
    # paper Section VII/VIII must be updated rather than the number quietly
    # changing.
    assert inforce["n_pairs"] == 9
    assert inforce["inforce_above_repealed"] == 3
