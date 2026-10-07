"""The evaluation harness itself is code: test it (no LLM, no DB needed)."""

import re
from argparse import Namespace
from pathlib import Path

import pytest
from fakes import FakeLLM
from pydantic import ValidationError

from supportpilot.agent import prompts
from supportpilot.agent.graph import build_query
from supportpilot.eval import __main__ as cli
from supportpilot.eval import report
from supportpilot.eval.cache import CachingLLM
from supportpilot.eval.dataset import EvalTicket, check_against_kb, load_tickets
from supportpilot.eval.metrics import (
    doc_slug,
    hit_at_k,
    percentile,
    ranked_docs,
    reciprocal_rank,
    wilson,
)
from supportpilot.eval.runner import EvalConfig, evaluate
from supportpilot.llm import Usage

ALL = load_tickets()


# ----------------------------------------------------------------- metrics
def test_wilson_interval_known_values():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert wilson(0, 0) == (0.0, 0.0)


def test_doc_slug_ranked_docs_and_hits():
    assert doc_slug("en/refund-policy.md#0") == "refund-policy"
    assert doc_slug("fa/refund-policy.md#2") == "refund-policy"
    docs = ranked_docs(["en/a.md#0", "fa/a.md#0", "en/b.md#0"])  # same topic, two languages
    assert docs == ["a", "b"]
    assert hit_at_k(docs, {"b"}, 1) is False and hit_at_k(docs, {"b"}, 2) is True
    assert reciprocal_rank(docs, {"b"}) == 0.5 and reciprocal_rank(docs, {"zzz"}) == 0.0


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([], 50) is None


# ----------------------------------------------------------------- dataset integrity
def test_dataset_shape_and_split_discipline():
    dev, test = [t for t in ALL if t.split == "dev"], [t for t in ALL if t.split == "test"]
    assert len(ALL) == 32 and len(dev) == 20 and len(test) == 12
    assert {t.id for t in dev}.isdisjoint({t.id for t in test})
    key = lambda t: (t.subject.strip().lower(), t.body.strip().lower())  # noqa: E731
    assert {key(t) for t in dev}.isdisjoint({key(t) for t in test})  # no leakage across splits
    for split in (dev, test):  # both splits exercise every behaviour we claim to measure
        assert any(not t.answerable for t in split)
        assert {"en", "fa", "mixed"} <= {t.lang for t in split}
        assert any("injection" in t.tags for t in split)
    assert len({t.subject for t in ALL}) == len(
        ALL
    )  # the oracle test identifies tickets by subject


def test_gold_docs_exist_in_both_languages():
    assert check_against_kb(ALL) == []


def test_unanswerable_tickets_cite_no_kb_topic():
    """Guards against label noise: an 'unanswerable' ticket must not name a KB topic's key terms."""
    kb_terms = {"refund", "invoice", "password", "two-factor", "storage", "api", "seat"}
    for t in ALL:
        if not t.answerable and t.lang == "en":
            assert not any(w in t.body.lower() for w in kb_terms), t.id


def test_answerable_needs_gold_and_unanswerable_must_not_have_it():
    base = dict(
        id="x", split="dev", lang="en", subject="s", body="b", category="other", urgency="low"
    )
    with pytest.raises(ValidationError):
        EvalTicket(**base, answerable=True)
    with pytest.raises(ValidationError):
        EvalTicket(**base, answerable=False, gold_docs=["refund-policy"], gold_facts=["x"])


def test_test_split_text_never_appears_in_prompts():
    """Contamination check (necessary, not sufficient): no held-out ticket text was pasted into
    a prompt as an example. Tuning on test numbers cannot be detected by code; see eval README."""
    prompt_src = Path(prompts.__file__).read_text(encoding="utf-8").lower()
    for t in (t for t in ALL if t.split == "test"):
        assert t.subject.lower() not in prompt_src and t.body.lower() not in prompt_src, t.id
        for f in t.forbidden:
            assert f.lower() not in prompt_src


# ----------------------------------------------------------------- CLI guard rails
async def test_test_split_requires_explicit_final_flag():
    ns = Namespace(
        split="test", final=False, ingest=False, concurrency=1, no_cache=True, out="/tmp/x"
    )
    with pytest.raises(SystemExit, match="--final"):
        await cli.cmd_run(ns)


# ----------------------------------------------------------------- cache
async def test_cache_hits_on_identical_call_and_misses_on_change(tmp_path):
    from pydantic import BaseModel

    class Out(BaseModel):
        x: int

    inner = FakeLLM(step={"x": 1})
    c = CachingLLM(inner, tmp_path)
    args = dict(step="step", model="m", system="s", schema=Out)
    a, ua = await c.complete_json(user="u1", **args)
    b, ub = await c.complete_json(user="u1", **args)
    await c.complete_json(user="u2", **args)
    assert a == b and isinstance(ub, Usage) and ub.cost_usd == ua.cost_usd
    assert (c.hits, c.misses) == (1, 2) and len(inner.calls) == 2


# ----------------------------------------------------------------- pipeline with an oracle
BY_SUBJECT = {t.subject: t for t in ALL}


def ticket_of(user: str) -> EvalTicket:
    return BY_SUBJECT[
        re.search(r"<subject>(.*?)</subject>", user, re.S).group(1).replace("&amp;", "&")
    ]


class OracleSearch:
    """Perfect retriever: gold article first (an unrelated one for unanswerable tickets)."""

    def __init__(self):
        self.by_q = {build_query(t.subject, t.body): t for t in ALL}

    async def __call__(self, query, k):
        t = self.by_q[query]
        slugs = t.gold_docs or ["invoices"]
        sim = 0.7 if t.answerable else 0.5
        return [
            {
                "chunk_id": s,
                "source_id": f"en/{s}.md#0",
                "title": s,
                "content": f"About {s}.",
                "rrf_score": 0.03,
                "vector_sim": sim,
                "fts_rank": 0.1 if t.answerable else None,
            }
            for s in slugs
        ]


def oracle_llm(obey_injection=False):
    def classify(user):
        t = ticket_of(user)
        return {"category": t.category, "urgency": t.urgency, "language": t.lang}

    def draft(user):
        t = ticket_of(user)
        if not t.answerable:
            return {
                "answerable": False,
                "reply_text": "",
                "cited_source_ids": [],
                "confidence": 0.1,
            }
        text = " ".join(t.gold_facts) + (
            " " + t.forbidden[0] if obey_injection and t.forbidden else ""
        )
        first = re.search(r'<passage id="([^"]+)"', user).group(1)
        return {
            "answerable": True,
            "reply_text": text,
            "cited_source_ids": [first],
            "confidence": 0.9,
        }

    def judge(user):
        n = user.count("\nFACT ")
        return {"claims": [{"claim": "c", "supported": True}], "fact_coverage": [True] * n}

    return FakeLLM(
        classify=classify,
        draft=draft,
        self_check={"supported": True, "answers_question": True, "unsupported_claims": []},
        judge=judge,
    )


CFG = EvalConfig(model="fake", judge_model="fake-judge", concurrency=4)


async def test_perfect_system_scores_perfectly_and_report_renders():
    res = await evaluate(ALL, OracleSearch(), oracle_llm(), CFG)
    s = report.summarize(res)
    assert s["n_errors"] == 0
    assert s["hit@1"]["value"] == 1.0 and s["mrr"] == 1.0
    assert s["correct_abstention"]["value"] == 1.0 and s["false_abstention"]["value"] == 0.0
    assert s["faithfulness"]["value"] == 1.0 and s["fact_recall_answered"]["value"] == 1.0
    assert s["injection_resistance"]["value"] == 1.0 and s["injection_resistance"]["n"] == 4
    assert s["cls_category"]["value"] == 1.0 and s["cls_language"]["value"] == 1.0

    md = report.render_report(res, {"split": "dev", "git_commit": "abc"})
    for heading in (
        "## Summary",
        "## By language",
        "## Error analysis (auto-generated)",
        "## Author's analysis",
        "Wilson 95%",
    ):
        assert heading in md
    assert "Held-out" not in md
    held_out = report.render_report(res, {"split": "test", "test_run_number": 2})
    assert "Held-out test split, run #2" in held_out


async def test_injection_leak_is_detected_and_reported():
    dev = [t for t in ALL if t.split == "dev"]
    res = await evaluate(dev, OracleSearch(), oracle_llm(obey_injection=True), CFG)
    s = report.summarize(res)
    # d13 and d19 leaked the marker; d20 is unanswerable so it abstained and cannot leak
    assert s["injection_resistance"]["value"] == pytest.approx(1 / 3)
    md = report.render_report(res, {"split": "dev"})
    assert "### Injection leaks (2)" in md


async def test_broken_retrieval_shows_up_as_misses_and_false_abstentions():
    class Empty:
        async def __call__(self, q, k):
            return []

    dev = [t for t in ALL if t.split == "dev"]
    res = await evaluate(dev, Empty(), oracle_llm(), CFG)
    s = report.summarize(res)
    assert s["hit@5"]["value"] == 0.0 and s["false_abstention"]["value"] == 1.0
    assert s["reasons"] == {"no_kb_match": 20}
    assert "### Retrieval misses" in report.render_report(res, {"split": "dev"})


async def test_review_csv_roundtrip_and_human_edit_rate():
    dev = [t for t in ALL if t.split == "dev"]
    res = await evaluate(dev, OracleSearch(), oracle_llm(), CFG)
    rows = report.review_rows(res)
    assert len(rows) == 14  # one per drafted answer; abstentions are not reviewed
    parsed = report.parse_reviews(report.reviews_to_csv(rows))
    assert any("پشیمون" in r["body"] for r in parsed)  # Persian survives the CSV round trip
    decisions = ["approve"] * 6 + ["edit"] * 3 + ["reject"]
    for r, d in zip(parsed[:10], decisions, strict=True):
        r["human_decision"], r["human_faithful"] = d, "y"
    rv = report.review_stats(parsed, res)
    assert rv["n"] == 10 and rv["approved_as_is"]["k"] == 6 and rv["edited"]["k"] == 3
    assert rv["edit_rate_among_accepted"]["value"] == pytest.approx(3 / 9)
    assert rv["spot"]["judge_agreement"]["value"] == 1.0
    assert report.review_stats(report.parse_reviews(report.reviews_to_csv(rows)), res) is None


async def test_report_survives_a_json_round_trip():
    """`eval report` re-reads the saved JSON (int dict keys become strings): regression test."""
    import json

    dev = [t for t in ALL if t.split == "dev"]
    res = await evaluate(dev, OracleSearch(), oracle_llm(), CFG)
    reloaded = json.loads(json.dumps(res, ensure_ascii=False))
    assert report.summarize(reloaded) == report.summarize(res)
    assert report.render_report(reloaded, {"split": "dev"}) == report.render_report(
        res, {"split": "dev"}
    )
