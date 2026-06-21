from stt_eval.metrics import score


def test_identical_is_perfect():
    s = score("the quick brown fox", "the quick brown fox", "en")
    assert s.wer == 0.0
    assert s.word_accuracy_pct == 100.0
    assert s.similarity_pct == 100.0


def test_punctuation_and_case_ignored_by_normalization():
    s = score("Hello, world.", "hello world", "en")
    assert s.wer == 0.0
    assert s.similarity_pct == 100.0


def test_one_word_wrong():
    s = score("the quick brown fox", "the quick green fox", "en")
    assert abs(s.wer - 0.25) < 1e-6
    assert abs(s.word_accuracy_pct - 75.0) < 1e-6


def test_empty_reference_guard():
    s = score("", "anything", "en")
    assert s.wer == 1.0
