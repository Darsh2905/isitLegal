"""Normaliser accuracy (paper VII-A). Claimed: 48/48 surface forms, 0 FPs."""


def test_probe_set_has_not_shrunk(normalizer):
    # Guards the Class-B failure mode: a metric can only be trusted if the
    # probe set it was measured on is still the one that was reported.
    assert normalizer["positives_total"] == 48
    assert normalizer["negatives_total"] == 8


def test_all_positive_surface_forms_normalise(normalizer):
    assert normalizer["positives_correct"] == 48, normalizer["failures"]
    assert normalizer["positive_accuracy"] == 1.0


def test_no_false_positive_citations(normalizer):
    assert normalizer["false_positives"] == 0, normalizer["false_positive_examples"]
