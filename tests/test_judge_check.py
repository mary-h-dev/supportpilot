"""The judge sensitivity check: the cases themselves, and the harness around them."""

import html
import re
from argparse import Namespace
from pathlib import Path

import pytest
from fakes import FakeLLM

from supportpilot.eval import __main__ as cli
from supportpilot.eval import judge_check as jc

CASES = jc.load_cases()
BY_DRAFT = {html.escape(c.draft, quote=False): c for c in CASES}
POISONED = [c for c in CASES if not c.expect_faithful]
CONTROLS = [c for c in CASES if c.expect_faithful]
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+", text.translate(_DIGITS)))


def case_of(user: str) -> jc.CalCase:
    return next(c for d, c in BY_DRAFT.items() if d in user)


def judge_returning(decide):
    """Scripted judge: decide(case) -> True if it should call the draft faithful."""

    def respond(user):
        ok = decide(case_of(user))
        return {"claims": [{"claim": "the draft's claim", "supported": ok}], "fact_coverage": []}

    return FakeLLM(judge=respond)


# ----------------------------------------------------------------- the cases themselves
def test_case_set_shape():
    assert len(CONTROLS) >= 5 and len(POISONED) >= 10
    assert len(BY_DRAFT) == len(CASES)  # drafts are unique: the scripted judge can identify cases
    assert all(c.error_type for c in POISONED) and not any(c.error_type for c in CONTROLS)
    langs = lambda cs: {c.docs[0].split("/")[0] for c in cs}  # noqa: E731
    assert langs(POISONED) == {"en", "fa"} and langs(CONTROLS) == {"en", "fa"}
    assert len({c.error_type for c in POISONED}) >= 6  # varied defects, not one trick


def test_every_case_points_at_real_articles_and_passages_load():
    for c in CASES:
        ps = jc.passages_for(c)
        assert ps and all(p["content"].strip() for p in ps) and ps[0]["source_id"].endswith("#0")
    refund = jc.passages_for(next(c for c in CASES if c.id == "c1"))
    assert "14 days" in refund[0]["content"]


def test_controls_contain_no_number_that_is_absent_from_their_passages():
    """Mechanical guard against a control accidentally containing an error."""
    for c in CONTROLS:
        known = numbers(" ".join(p["content"] for p in jc.passages_for(c)))
        assert numbers(c.draft) <= known, f"{c.id}: {numbers(c.draft) - known}"


def test_wrong_number_cases_really_contain_a_number_the_articles_do_not():
    for c in (c for c in POISONED if c.error_type == "wrong_number"):
        known = numbers(" ".join(p["content"] for p in jc.passages_for(c)))
        assert numbers(c.draft) - known, f"{c.id} is not actually wrong"


def test_swapped_values_case_uses_only_real_numbers_in_the_wrong_place():
    c = next(c for c in POISONED if c.error_type == "swapped_values")
    known = numbers(" ".join(p["content"] for p in jc.passages_for(c)))
    assert numbers(c.draft) <= known  # the defect is the pairing, not a new number


def test_invalid_case_files_are_rejected(tmp_path):
    bad = tmp_path / "c.json"
    bad.write_text('[{"id":"x","docs":["en/refund-policy"],"draft":"d","expect_faithful":false}]')
    with pytest.raises(ValueError, match="error_type"):
        jc.load_cases(bad)


# ----------------------------------------------------------------- the harness
async def run(llm):
    return await jc.run_check(CASES, llm, "judge-model", concurrency=4)


async def test_a_good_judge_scores_perfectly():
    res = await run(judge_returning(lambda c: c.expect_faithful))
    s = jc.summarize(res)
    assert s["detection"]["value"] == 1.0 and s["false_alarm"]["value"] == 0.0
    assert s["detection"]["n"] == len(POISONED) and s["failed"] == []
    assert "⚠️" not in jc.render_report(res, {"judge_model": "m"})


async def test_a_lenient_judge_is_exposed_and_the_report_warns():
    """This is the failure the whole check exists for: everything is 'faithful'."""
    res = await run(judge_returning(lambda c: True))
    s = jc.summarize(res)
    assert s["detection"]["value"] == 0.0 and len(s["missed"]) == len(POISONED)
    md = jc.render_report(res, {"judge_model": "m"})
    assert "NOT trustworthy" in md and "MISSED" in md


async def test_an_over_strict_judge_shows_false_alarms():
    res = await run(judge_returning(lambda c: False))
    s = jc.summarize(res)
    assert s["detection"]["value"] == 1.0 and s["false_alarm"]["value"] == 1.0
    assert "false alarms on correct drafts" in jc.render_report(res, {"judge_model": "m"})


async def test_judge_failures_are_excluded_not_counted_as_misses_or_catches():
    def flaky(user):
        if case_of(user).id == "p1":
            raise RuntimeError("timeout")
        c = case_of(user)
        return {"claims": [{"claim": "x", "supported": c.expect_faithful}], "fact_coverage": []}

    from supportpilot.llm import LLMError

    def respond(user):
        try:
            return flaky(user)
        except RuntimeError as e:
            raise LLMError(str(e)) from e

    res = await run(FakeLLM(judge=respond))
    s = jc.summarize(res)
    assert s["failed"] == ["p1"] and s["detection"]["n"] == len(POISONED) - 1
    assert s["detection"]["value"] == 1.0


async def test_a_judge_that_finds_no_claims_is_a_failure_not_a_verdict():
    res = await run(FakeLLM(judge={"claims": [], "fact_coverage": []}))
    s = jc.summarize(res)
    assert len(s["failed"]) == len(CASES) and s["detection"]["value"] is None


async def test_cli_writes_report_and_json(tmp_path, monkeypatch):
    import supportpilot.llm as llm_mod

    monkeypatch.setattr(
        llm_mod, "OpenAICompatLLM", lambda *a, **k: judge_returning(lambda c: c.expect_faithful)
    )
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)  # eval/ + kb/ relative paths
    await cli.cmd_judge_check(Namespace(concurrency=4, no_cache=True, out=str(tmp_path)))
    names = {p.name for p in tmp_path.iterdir()}
    assert any(n.startswith("judge-check-") and n.endswith(".md") for n in names)
    assert any(n.endswith(".json") for n in names)
    md = next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert "Deliberately wrong drafts flagged" in md and "p9" in md
