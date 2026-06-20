from stt_eval.normalize import normalize_text


def test_lowercases_and_strips_punctuation():
    assert normalize_text("Hello, World!") == "hello world"


def test_collapses_whitespace():
    assert normalize_text("  a   b\tc \n") == "a b c"


def test_russian_yo_folds_to_e():
    assert normalize_text("ещё Ещё", language="ru") == "еще еще"


def test_empty():
    assert normalize_text("") == ""
