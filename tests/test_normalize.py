from supportpilot.kb.normalize import detect_language, normalize, query_terms


def test_unifies_arabic_letters_and_digits():
    assert normalize("كيف ١٢۳") == "کیف 123"


def test_zwnj_becomes_space():
    assert normalize("می‌خواهم") == "می خواهم"


def test_removes_diacritics_and_tatweel():
    assert normalize("مُحَمّـد") == "محمد"


def test_query_terms_drop_stopwords_and_dedupe():
    assert query_terms("How do I cancel my cancel plan?") == ["how", "cancel", "plan"]
    assert "پرداخت" in query_terms("چرا پرداخت من ناموفق شد؟")
    assert "من" not in query_terms("چرا پرداخت من ناموفق شد؟")


def test_query_terms_are_tsquery_safe():
    # injection-ish characters must never survive into the tsquery string
    assert query_terms("a' | !b & (c) :*") == []
    assert all(t.isalnum() or "_" in t for t in query_terms("refund'; DROP TABLE x; --"))


def test_detect_language():
    assert detect_language("I want a refund") == "en"
    assert detect_language("می‌خواهم پولم را پس بگیرم") == "fa"
    assert detect_language("پرداخت من failed شد و payment نرفت") == "mixed"
