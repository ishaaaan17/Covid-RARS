from covid_rars.labels import normalize_label


def test_normalize_positive_labels():
    assert normalize_label("positive") == "positive"
    assert normalize_label("COVID positive") == "positive"
    assert normalize_label("tested positive") == "positive"


def test_normalize_negative_labels():
    assert normalize_label("negative") == "negative"
    assert normalize_label("healthy") == "negative"
    assert normalize_label("normal") == "negative"


def test_unknown_label_is_preserved():
    assert normalize_label("") == "unknown"
    assert normalize_label(None) == "unknown"
    assert normalize_label("recovered") == "unknown"
