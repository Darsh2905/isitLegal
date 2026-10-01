"""
Quotation check V4 (paper VII-C), including the character-vs-token negative
result: character-level fuzzy matching scored a meaning-inverting one-word
substitution at 95.9%. Token-level scoring is what makes theta=0.95 an
empirical rather than arbitrary choice -- these tests lock that in.
"""


def test_quote_checks_actually_ran(quote):
    # Class-B guard. Every V4 metric below is vacuous if the checker never
    # executed -- which is exactly how the unterminated-quotation bug hid.
    assert quote["quote_checks_performed"] > 0


def test_perfect_separation_at_deployed_theta(quote):
    at95 = quote["sweep"][0.95]
    assert at95["fp"] == 0 and at95["fn"] == 0
    assert at95["precision"] == 1.0 and at95["recall"] == 1.0


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
