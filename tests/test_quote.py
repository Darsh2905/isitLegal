"""
Quotation check V4, including the character-vs-token negative result
(character-level fuzzy matching scored a meaning-inverting one-word
substitution at 95.9%) and its sequel: token-level scoring at theta=0.95
was itself length-dependent -- a substitution at either end of a 20+ token
quote scores (n-1)/n >= 0.95 and passed. The deployed rule is now exact
(theta=1.0); the length sweep on real provisions locks that in.
"""


def test_quote_checks_actually_ran(quote):
    # Class-B guard. Every V4 metric below is vacuous if the checker never
    # executed -- which is exactly how the unterminated-quotation bug hid.
    assert quote["quote_checks_performed"] > 0


def test_perfect_separation_at_deployed_theta(quote):
    at100 = quote["sweep"][1.0]
    assert at100["fp"] == 0 and at100["fn"] == 0
    assert at100["precision"] == 1.0 and at100["recall"] == 1.0


def test_deployed_theta_is_exact():
    from verify.verifier import CitationVerifier
    from ingest.metadata_db import build_seed_db
    assert CitationVerifier(build_seed_db()).theta == 1.0


def test_exact_matching_rejects_every_substitution_at_every_length(quote):
    for (theta, n), r in quote["length_sweep"].items():
        if theta == 1.0:
            assert r["substitution_accepted"] == 0, (n, r)
            assert r["faithful_accepted"] == r["faithful_n"], (n, r)


def test_theta_095_fails_on_long_quotations(quote):
    # The reviewer's point, measured: at 0.95 an edge substitution passes
    # once the quote reaches 20 tokens (and at 40, the second word too).
    ls = quote["length_sweep"]
    assert ls[(0.95, 19)]["substitution_accepted"] == 0
    assert ls[(0.95, 20)]["substitution_accepted"] > 0
    assert (ls[(0.95, 40)]["substitution_accepted"]
            > ls[(0.95, 20)]["substitution_accepted"])
    assert ls[(0.98, 60)]["substitution_accepted"] > 0


def test_near_miss_substitution_is_caught_at_095(quote):
    # 'statute' substituted for 'law' in an 18-word Article 21 quotation.
    assert quote["per_probe"]["q-f4-c"]["flagged"] is True


def test_lower_thresholds_miss_the_near_miss(quote):
    # This is the empirical justification for theta=0.95: at 0.80 and 0.90 the
    # meaning-inverting substitution passes. If this ever stops holding, the
    # paper's justification for the threshold has changed and must be rewritten.
    assert quote["sweep"][0.80]["recall"] < 1.0
    assert quote["sweep"][0.90]["recall"] < 1.0


def test_clean_quotations_are_never_flagged(quote):
    for pid in ("q-clean-1", "q-clean-2"):
        assert quote["per_probe"][pid]["flagged"] is False
