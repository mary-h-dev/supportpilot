"""Is the LLM judge actually able to catch an unfaithful draft?

The dev drafts are all faithful, so "judge agrees with the human on 15/15" proves nothing about
its ability to detect errors. This check feeds the judge hand-written drafts with ONE deliberate
defect each (wrong number, invented policy, contradiction...) plus correct controls, and reports:
  - detection rate: poisoned drafts the judge flagged (should be high)
  - false-alarm rate: correct drafts the judge flagged (should be ~0)
If detection is low, the faithfulness numbers in the main report cannot be trusted.
"""

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel

from ..kb.chunking import chunk_text
from ..kb.ingest import parse_article
from .judge import judge_draft
from .metrics import proportion

DEFAULT_PATH = Path("eval/judge_calibration.json")


class CalCase(BaseModel):
    id: str
    docs: list[str]  # "en/refund-policy": the articles shown to the judge as passages
    draft: str
    expect_faithful: bool
    error_type: str = ""  # required for poisoned cases


def load_cases(path: Path = DEFAULT_PATH) -> list[CalCase]:
    cases = [CalCase(**c) for c in json.loads(Path(path).read_text(encoding="utf-8"))]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("duplicate calibration case ids")
    for c in cases:
        if c.expect_faithful == bool(c.error_type):
            raise ValueError(f"{c.id}: poisoned cases need error_type, controls must not have one")
    return cases


def passages_for(case: CalCase, kb_root: Path = Path("kb")) -> list[dict]:
    out = []
    for ref in case.docs:
        source_uri, title, _lang, body = parse_article(kb_root / f"{ref}.md", kb_root)
        for i, chunk in enumerate(chunk_text(body)):
            out.append({"source_id": f"{source_uri}#{i}", "title": title, "content": chunk})
    return out


async def run_check(
    cases: list[CalCase], llm, judge_model: str, kb_root: Path = Path("kb"), concurrency: int = 2
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def one(c: CalCase) -> dict:
        async with sem:
            rec = {"id": c.id, "error_type": c.error_type, "expect_faithful": c.expect_faithful}
            try:
                j = await judge_draft(llm, judge_model, passages_for(c, kb_root), c.draft, [])
            except Exception as e:
                return rec | {"failed": f"{type(e).__name__}: {e}"}
            if j["no_claims"]:  # nothing to judge: treat as a judge failure, not as a verdict
                return rec | {"failed": "judge returned no claims"}
            return rec | {"judged_faithful": j["faithful"], "unsupported": j["unsupported"]}

    return await asyncio.gather(*(one(c) for c in cases))


def summarize(results: list[dict]) -> dict:
    scored = [r for r in results if "failed" not in r]
    poisoned = [r for r in scored if not r["expect_faithful"]]
    controls = [r for r in scored if r["expect_faithful"]]
    return {
        "detection": proportion(sum(not r["judged_faithful"] for r in poisoned), len(poisoned)),
        "false_alarm": proportion(sum(not r["judged_faithful"] for r in controls), len(controls)),
        "failed": [r["id"] for r in results if "failed" in r],
        "missed": [r["id"] for r in poisoned if r["judged_faithful"]],
        "false_alarms": [r["id"] for r in controls if not r["judged_faithful"]],
    }


def _p(d: dict) -> str:
    if d["value"] is None:
        return "n/a"
    return f"{d['k']}/{d['n']} = {d['value'] * 100:.0f}%  (95% CI {d['lo'] * 100:.0f}–{d['hi'] * 100:.0f}%)"


def render_report(results: list[dict], meta: dict) -> str:
    s = summarize(results)
    L = [f"# Judge sensitivity check: `{meta.get('judge_model', '?')}`", ""]
    L += ["| Run | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in meta.items()]
    L += ["", "## Result", "", "| | |", "|---|---|"]
    L += [f"| Deliberately wrong drafts flagged (detection) | {_p(s['detection'])} |"]
    L += [f"| Correct drafts wrongly flagged (false alarms) | {_p(s['false_alarm'])} |"]
    L += [f"| Judge failures (excluded) | {len(s['failed'])} {s['failed'] or ''} |"]
    det = s["detection"]
    if det["value"] is not None and det["value"] < 0.8:
        L += [
            "",
            f"> ⚠️ **The judge missed {len(s['missed'])} of {det['n']} deliberately wrong drafts "
            f"({', '.join(s['missed'])}). Faithfulness numbers produced with this judge model are "
            "NOT trustworthy.** Try another judge model.",
        ]
    if s["false_alarms"]:
        L += [
            "",
            f"> Note: false alarms on correct drafts: {', '.join(s['false_alarms'])}. "
            "A judge that over-flags would understate faithfulness.",
        ]
    L += [
        "",
        "## Per case",
        "",
        "| id | defect | should be | judge said | outcome | judge's unsupported claims |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        want = "faithful" if r["expect_faithful"] else "unfaithful"
        if "failed" in r:
            L.append(
                f"| {r['id']} | {r['error_type'] or '-'} | {want} | - | judge failed | {r['failed']} |"
            )
            continue
        said = "faithful" if r["judged_faithful"] else "unfaithful"
        outcome = (
            ("ok" if r["judged_faithful"] else "FALSE ALARM")
            if r["expect_faithful"]
            else ("MISSED" if r["judged_faithful"] else "caught")
        )
        claims = " \\| ".join(r["unsupported"]).replace("\n", " ")[:160]
        L.append(
            f"| {r['id']} | {r['error_type'] or '-'} | {want} | {said} | {outcome} | {claims} |"
        )
    L += [
        "",
        "_The cases are hand-written (`eval/judge_calibration.json`) and were not tuned on any model. "
        "They cover one defect type each, so a high score shows the judge catches *these* defect types, "
        "not that it catches every possible one._",
        "",
    ]
    return "\n".join(L)
