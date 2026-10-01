"""Text normalization for Persian/English search.

Applied identically to documents (at ingestion) and queries (at search time).
"""

import re
import unicodedata

_CHAR_MAP = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",  # Arabic yeh / alef maksura -> Persian yeh
        "ك": "ک",  # Arabic kaf -> Persian kaf
        "ة": "ه",
        "ۀ": "ه",
        "\u200c": " ",  # ZWNJ -> space (so "می‌خواهم" -> "می خواهم")
        "\u200f": "",
        "\u200e": "",
        "\u0640": "",  # RLM, LRM, tatweel
    }
)
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_DIACRITICS = re.compile("[\u064b-\u065f\u0670]")
_SPACES = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+", re.UNICODE)

_STOP = {
    "en": "the a an and or of to in on for is are was were be i my me we you your it this that "
    "with at by from as not have has do does can could would please".split(),
    "fa": "و در به از که را این آن با برای است هست بود شد شده می من ما شما تا یا هم اما "
    "چرا چه چطور آیا نمی باید کرد کنم کنید".split(),
}
STOPWORDS = set(_STOP["en"]) | set(_STOP["fa"])


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CHAR_MAP).translate(_DIGITS)
    text = _DIACRITICS.sub("", text).lower()
    return _SPACES.sub(" ", text).strip()


def query_terms(text: str) -> list[str]:
    """Distinct content-bearing tokens of a normalized query (for the OR full-text query)."""
    seen: dict[str, None] = {}
    for tok in _TOKEN.findall(normalize(text)):
        if len(tok) > 1 and tok not in STOPWORDS:
            seen.setdefault(tok)
    return list(seen)


def detect_language(text: str) -> str:
    fa = sum(1 for ch in text if "\u0600" <= ch <= "\u06ff")
    en = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if fa and en:
        return "mixed" if min(fa, en) / max(fa, en) > 0.2 else ("fa" if fa > en else "en")
    return "fa" if fa else "en"
