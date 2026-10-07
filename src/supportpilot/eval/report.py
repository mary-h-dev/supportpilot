"""Turns raw per-ticket results into the metrics, the markdown report and the review sheet."""

import csv
import io
from collections import Counter

from .metrics import median, percentile, proportion

REVIEW_COLUMNS = [
    "id",
    "lang",
    "answerable",
    "subject",
    "body",
    "draft",
    "sources",
    "judge_faithful",
    "judge_unsupported",
    "human_decision",
    "human_faithful",
    "human_notes",
]


def _ans(rs):
    return [r for r in rs if r["answerable"] and "retrieval" in r]


def _una(rs):
    return [r for r in rs if not r["answerable"] and "retrieval" in r]


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "agent" in r]  # tickets that ran end to end
    ans, una = _ans(ok), _una(ok)
    s: dict = {"n_tickets": len(results), "n_errors": len(results) - len(ok)}

    for k in (1, 3, 5):
        s[f"hit@{k}"] = proportion(sum(r["retrieval"]["hit"][str(k)] for r in ans), len(ans))
    s["mrr"] = sum(r["retrieval"]["rr"] for r in ans) / len(ans) if ans else None

    s["correct_abstention"] = proportion(
        sum(r["class"] == "correct_abstention" for r in una), len(una)
    )
    s["false_abstention"] = proportion(sum(r["class"] == "false_abstention" for r in ans), len(ans))
    s["coverage"] = proportion(sum(r["class"] == "answered" for r in ans), len(ans))

    judged = [r for r in ok if r.get("judge") and "failed" not in r["judge"]]
    scored = [r for r in judged if not r["judge"]["no_claims"]]
    s["faithfulness"] = proportion(sum(r["judge"]["faithful"] for r in scored), len(scored))
    s["claim_support"] = proportion(
        sum(r["judge"]["claims_supported"] for r in scored),
        sum(r["judge"]["claims_total"] for r in scored),
    )
    s["judge_failed"] = sum(1 for r in ok if r.get("judge") and "failed" in r["judge"])
    s["judge_no_claims"] = len(judged) - len(scored)

    answered = [
        r for r in ans if r["class"] == "answered" and r.get("judge") and "failed" not in r["judge"]
    ]
    total_facts = sum(len(r["gold"]["facts"]) for r in ans)
    s["fact_recall_answered"] = proportion(
        sum(sum(r["judge"]["facts_covered"]) for r in answered),
        sum(len(r["gold"]["facts"]) for r in answered),
    )
    s["fact_recall_end_to_end"] = proportion(  # abstaining counts as missing every fact
        sum(sum(r["judge"]["facts_covered"]) for r in answered), total_facts
    )

    inj = [r for r in ok if "injection" in r["tags"]]
    s["injection_resistance"] = proportion(sum(not r["leak"]["final"] for r in inj), len(inj))
    s["injection_near_miss"] = sum(r["leak"]["discarded"] for r in inj)

    for field in ("category", "urgency", "language"):
        have = [r for r in ok if r["agent"]["classification"]]
        gold = (lambda r: r["lang"]) if field == "language" else (lambda r, f=field: r["gold"][f])
        s[f"cls_{field}"] = proportion(
            sum(r["agent"]["classification"][field] == gold(r) for r in have), len(have)
        )

    lat = [r["agent"]["latency_ms"] for r in ok]
    s["latency_p50_ms"], s["latency_p95_ms"] = percentile(lat, 50), percentile(lat, 95)
    s["agent_cost_usd"] = sum(r["agent"]["cost_usd"] for r in ok)
    s["judge_cost_usd"] = sum(r.get("judge", {}).get("cost_usd", 0) for r in ok)
    s["tokens_in"] = sum(r["agent"]["prompt_tokens"] for r in ok)
    s["tokens_out"] = sum(r["agent"]["completion_tokens"] for r in ok)

    s["vec_sim_median"] = {
        "answerable": median(
            [
                r["retrieval"]["top_vec_sim"]
                for r in ans
                if r["retrieval"]["top_vec_sim"] is not None
            ]
        ),
        "unanswerable": median(
            [
                r["retrieval"]["top_vec_sim"]
                for r in una
                if r["retrieval"]["top_vec_sim"] is not None
            ]
        ),
    }
    s["reasons"] = dict(Counter(r["agent"]["reason"] for r in ok if r["agent"]["reason"]))
    s["by_lang"] = {}
    for lang in ("en", "fa", "mixed"):
        a, u = [r for r in ans if r["lang"] == lang], [r for r in una if r["lang"] == lang]
        s["by_lang"][lang] = {
            "answerable": len(a),
            "unanswerable": len(u),
            "hit@3": sum(r["retrieval"]["hit"]["3"] for r in a),
            "answered": sum(r["class"] == "answered" for r in a),
            "correct_abstention": sum(r["class"] == "correct_abstention" for r in u),
        }
    return s


# ------------------------------------------------------------------ human review
def review_rows(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        d = (r.get("agent") or {}).get("draft")
        if not d:
            continue
        j = r.get("judge") or {}
        rows.append(
            {
                "id": r["id"],
                "lang": r["lang"],
                "answerable": r["answerable"],
                "subject": r["subject"],
                "body": r["body"],
                "draft": d["text"],
                "sources": ", ".join(s["source_id"] for s in d["sources"]),
                "judge_faithful": j.get("faithful", ""),
                "judge_unsupported": " | ".join(j.get("unsupported", [])),
                "human_decision": "",
                "human_faithful": "",
                "human_notes": "",
            }
        )
    return rows


def reviews_to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=REVIEW_COLUMNS)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def parse_reviews(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))


def review_stats(rows: list[dict], results: list[dict]) -> dict | None:
    decided = [
        r
        for r in rows
        if r.get("human_decision", "").strip().lower() in {"approve", "edit", "reject"}
    ]
    if not decided:
        return None
    c = Counter(r["human_decision"].strip().lower() for r in decided)
    n = len(decided)
    judge = {r["id"]: (r.get("judge") or {}).get("faithful") for r in results}
    spot = [
        r for r in rows if r.get("human_faithful", "").strip().lower() in {"y", "n", "yes", "no"}
    ]
    human_ok = lambda r: r["human_faithful"].strip().lower() in {"y", "yes"}  # noqa: E731
    agree = sum(1 for r in spot if judge.get(r["id"]) is not None and human_ok(r) == judge[r["id"]])
    return {
        "n": n,
        "approved_as_is": proportion(c["approve"], n),
        "edited": proportion(c["edit"], n),
        "rejected": proportion(c["reject"], n),
        "edit_rate_among_accepted": proportion(c["edit"], c["approve"] + c["edit"]),
        "spot": {
            "n": len(spot),
            "human_faithful": proportion(sum(human_ok(r) for r in spot), len(spot)),
            "judge_agreement": proportion(agree, len(spot)),
        },
    }


# ------------------------------------------------------------------ markdown
def _p(d: dict | None, pct: bool = True) -> str:
    if not d or d["value"] is None:
        return "n/a"
    return f"{d['value'] * 100:.1f}%  (95% CI {d['lo'] * 100:.0f}–{d['hi'] * 100:.0f}%)"


def _cell(x, n: int = 110) -> str:
    x = str(x if x is not None else "").replace("|", "\\|").replace("\n", " ")
    return x if len(x) <= n else x[: n - 1] + "…"


def summary_table(s: dict, rv: dict | None = None) -> str:
    rows = [
        ("Retrieval hit@1", s["hit@1"]),
        ("Retrieval hit@3", s["hit@3"]),
        ("Retrieval hit@5", s["hit@5"]),
        ("Correct abstention (KB has no answer → human)", s["correct_abstention"]),
        ("False abstention (KB has the answer, still → human)", s["false_abstention"]),
        ("Answer coverage (answerable tickets that got a draft)", s["coverage"]),
        ("Faithfulness (drafts fully supported, LLM judge)", s["faithfulness"]),
        ("Claim-level support (LLM judge)", s["claim_support"]),
        ("Fact recall, among drafted answers", s["fact_recall_answered"]),
        ("Fact recall, end-to-end (abstaining = 0)", s["fact_recall_end_to_end"]),
        ("Injection resistance (no forbidden text in final draft)", s["injection_resistance"]),
        ("Classification: category accuracy", s["cls_category"]),
        ("Classification: urgency accuracy", s["cls_urgency"]),
        ("Classification: language accuracy", s["cls_language"]),
    ]
    out = ["| Metric | Result | n |", "|---|---|---|"]
    out += [f"| {name} | {_p(d)} | {d['n']} |" for name, d in rows]
    out.append(
        f"| MRR (retrieval) | {s['mrr']:.3f} | {s['hit@1']['n']} |"
        if s["mrr"] is not None
        else "| MRR | n/a | 0 |"
    )
    if rv:
        out.append(f"| Human: approved as-is | {_p(rv['approved_as_is'])} | {rv['n']} |")
        out.append(f"| Human: edited | {_p(rv['edited'])} | {rv['n']} |")
        out.append(f"| Human: rejected | {_p(rv['rejected'])} | {rv['n']} |")
        out.append(
            f"| **Human-edit rate** (edited / accepted) | {_p(rv['edit_rate_among_accepted'])} | "
            f"{rv['edit_rate_among_accepted']['n']} |"
        )
    else:
        out.append("| Human-edit rate | _pending: fill `human_decision` in the review CSV_ | – |")
    return "\n".join(out)


def _errors(results: list[dict]) -> str:
    ok = [r for r in results if "agent" in r]
    sec: list[str] = []

    def table(title, header, rows):
        sec.append(f"\n### {title} ({len(rows)})\n")
        if not rows:
            sec.append("_None._")
            return
        sec.append("| " + " | ".join(header) + " |")
        sec.append("|" + "---|" * len(header))
        sec.extend("| " + " | ".join(_cell(c) for c in r) + " |" for r in rows)

    table(
        "Pipeline crashes",
        ["id", "error"],
        [(r["id"], r["error"]) for r in results if "error" in r],
    )
    table(
        "Retrieval misses (gold article not in top-5)",
        ["id", "lang", "gold", "retrieved"],
        [
            (
                r["id"],
                r["lang"],
                ", ".join(r["gold"]["docs"]),
                ", ".join(r["retrieval"]["ranked_docs"][:3]),
            )
            for r in _ans(ok)
            if not r["retrieval"]["hit"]["5"]
        ],
    )
    table(
        "False abstentions (answerable, but routed to a human)",
        ["id", "lang", "reason", "discarded draft"],
        [
            (
                r["id"],
                r["lang"],
                r["agent"]["reason"],
                (r["agent"]["discarded_draft"] or {}).get("reply_text", ""),
            )
            for r in ok
            if r["class"] == "false_abstention"
        ],
    )
    table(
        "Answered although the KB has no answer",
        ["id", "lang", "draft", "sources"],
        [
            (
                r["id"],
                r["lang"],
                r["agent"]["draft"]["text"],
                ", ".join(s["source_id"] for s in r["agent"]["draft"]["sources"]),
            )
            for r in ok
            if r["class"] == "hallucinated_answer"
        ],
    )
    table(
        "Unfaithful drafts (judge found unsupported claims)",
        ["id", "unsupported claims"],
        [
            (r["id"], " | ".join(r["judge"]["unsupported"]))
            for r in ok
            if r.get("judge") and "failed" not in r["judge"] and r["judge"]["unsupported"]
        ],
    )
    table(
        "Missing gold facts in answered drafts",
        ["id", "missing facts"],
        [
            (
                r["id"],
                " | ".join(
                    f
                    for f, c in zip(r["gold"]["facts"], r["judge"]["facts_covered"], strict=False)
                    if not c
                ),
            )
            for r in ok
            if r["class"] == "answered"
            and r.get("judge")
            and "failed" not in r["judge"]
            and not all(r["judge"]["facts_covered"])
        ],
    )
    table(
        "Injection leaks",
        ["id", "in final draft", "in discarded draft"],
        [
            (r["id"], r["leak"]["final"], r["leak"]["discarded"])
            for r in ok
            if "injection" in r["tags"] and (r["leak"]["final"] or r["leak"]["discarded"])
        ],
    )
    table(
        "Classification errors",
        ["id", "field", "gold", "predicted"],
        [
            (r["id"], f, g, r["agent"]["classification"][f])
            for r in ok
            if r["agent"]["classification"]
            for f, g in (
                ("category", r["gold"]["category"]),
                ("urgency", r["gold"]["urgency"]),
                ("language", r["lang"]),
            )
            if r["agent"]["classification"][f] != g
        ],
    )
    return "\n".join(sec)


def render_report(results: list[dict], meta: dict, reviews: list[dict] | None = None) -> str:
    s = summarize(results)
    rv = review_stats(reviews, results) if reviews else None
    L: list[str] = [f"# SupportPilot evaluation: `{meta['split']}` split", ""]
    L += ["| Run | |", "|---|---|"] + [f"| {k} | {_cell(v, 200)} |" for k, v in meta.items()]
    if meta["split"] == "test":
        L += [
            "",
            f"> ⚠️ **Held-out test split, run #{meta.get('test_run_number', '?')}.** "
            "Prompts and thresholds must be tuned on `dev` only; never change anything because of these numbers.",
        ]
    L += [
        "",
        f"Tickets: {s['n_tickets']} ({s['n_errors']} pipeline errors). "
        "Intervals are Wilson 95% and are wide at this sample size: treat differences of a few points as noise.",
        "",
    ]
    L += ["## Summary", "", summary_table(s, rv), ""]
    L += [
        "## By language",
        "",
        "| lang | answerable | hit@3 | answered | unanswerable | correct abstention |",
        "|---|---|---|---|---|---|",
    ]
    for lang, b in s["by_lang"].items():
        L.append(
            f"| {lang} | {b['answerable']} | {b['hit@3']} | {b['answered']} | {b['unanswerable']} | {b['correct_abstention']} |"
        )
    L += [
        "",
        "## Cost and latency",
        "",
        f"- Agent latency per ticket: p50 {s['latency_p50_ms'] or 0:.0f} ms, p95 {s['latency_p95_ms'] or 0:.0f} ms (excludes judge; retrieval is memoised, so embedding time is not included)",
        f"- Tokens: {s['tokens_in']} in / {s['tokens_out']} out",
        f"- Estimated cost: agent ${s['agent_cost_usd']:.5f} + judge ${s['judge_cost_usd']:.5f} "
        "(from `LLM_PRICE_*`; includes calls served from cache at their original price)",
        f"- Judge: {s['judge_failed']} failed, {s['judge_no_claims']} drafts had no factual claims (excluded from faithfulness)",
        "",
    ]
    L += [
        "## Diagnostics",
        "",
        f"- Median top-1 vector similarity: answerable **{s['vec_sim_median']['answerable'] or 0:.3f}** vs "
        f"unanswerable **{s['vec_sim_median']['unanswerable'] or 0:.3f}**. If these are close, a similarity "
        "threshold cannot separate 'KB has the answer' from 'KB does not', which is why abstention relies on "
        "the drafter's `answerable` flag, deterministic guards and the fact-check instead.",
        f"- Routed to a human, by reason: {s['reasons'] or 'none'}",
        "",
    ]
    if rv:
        sp = rv["spot"]
        L += [
            "## Human review",
            "",
            f"- Reviewed drafts: {rv['n']}.",
            f"- Spot check of the judge: {sp['n']} drafts labelled by hand; human says faithful {_p(sp['human_faithful'])}; "
            f"judge agrees with the human on {_p(sp['judge_agreement'])}.",
            "",
        ]
    L += ["## Error analysis (auto-generated)", _errors(results), ""]
    L += [
        "## Author's analysis",
        "",
        "_Written by hand after reading the failure tables above: what failed, why, what was changed on `dev`, "
        "and what was deliberately left unfixed._",
        "",
    ]
    return "\n".join(L)
