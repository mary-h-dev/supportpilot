import math
import re

_SRC = re.compile(r"^[^/]+/(?P<slug>[^#]+?)\.md(#\d+)?$")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion. With n of 10-20 it is WIDE: always report it."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def proportion(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "value": (k / n) if n else None, "lo": lo, "hi": hi}


def doc_slug(source_id: str) -> str:
    """'en/refund-policy.md#0' -> 'refund-policy' (gold labels are language-agnostic)."""
    m = _SRC.match(source_id)
    return m.group("slug") if m else source_id


def ranked_docs(source_ids: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for s in source_ids:
        seen.setdefault(doc_slug(s))
    return list(seen)


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> bool:
    return any(d in gold for d in ranked[:k])


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def median(values: list[float]) -> float | None:
    return percentile(values, 50)
