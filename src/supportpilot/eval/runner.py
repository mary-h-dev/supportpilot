"""Runs every eval ticket through the REAL agent graph (classify -> retrieve -> draft ->
self_check) and records everything needed for metrics and error analysis."""

import asyncio
from dataclasses import dataclass

from ..agent.graph import AgentDeps, build_graph, build_query, run_agent
from .dataset import EvalTicket
from .judge import judge_draft
from .metrics import hit_at_k, ranked_docs, reciprocal_rank

KS = (1, 3, 5)


@dataclass
class EvalConfig:
    model: str
    judge_model: str
    k: int = 5
    min_confidence: float = 0.6
    concurrency: int = 2


class MemoSearch:
    """Retrieval runs once per ticket: the retrieval metric and the agent see the same passages."""

    def __init__(self, inner):
        self.inner, self.cache = inner, {}

    async def __call__(self, query: str, k: int) -> list[dict]:
        key = (query, k)
        if key not in self.cache:
            self.cache[key] = await self.inner(query, k)
        return self.cache[key]


def outcome_class(answerable: bool, drafted: bool) -> str:
    if answerable:
        return "answered" if drafted else "false_abstention"
    return "hallucinated_answer" if drafted else "correct_abstention"


async def _evaluate_one(t: EvalTicket, graph, memo: MemoSearch, llm, cfg: EvalConfig) -> dict:
    rec: dict = {
        "id": t.id,
        "split": t.split,
        "lang": t.lang,
        "answerable": t.answerable,
        "tags": t.tags,
        "subject": t.subject,
        "body": t.body,
        "gold": {
            "category": t.category,
            "urgency": t.urgency,
            "docs": t.gold_docs,
            "facts": t.gold_facts,
        },
    }
    try:
        # --- retrieval (independent of the LLM) ---
        try:
            passages = await memo(build_query(t.subject, t.body), cfg.k)
        except Exception as e:
            passages, rec["retrieval_error"] = [], f"{type(e).__name__}: {e}"
        docs = ranked_docs([p["source_id"] for p in passages])
        gold = set(t.gold_docs)
        sims = [p["vector_sim"] for p in passages if p.get("vector_sim") is not None]
        rec["retrieval"] = {
            "ranked_docs": docs,
            "sources": [p["source_id"] for p in passages],
            "top_vec_sim": max(sims) if sims else None,
            "has_fts": any(p.get("fts_rank") is not None for p in passages),
            "hit": {str(k): hit_at_k(docs, gold, k) for k in KS}
            if t.answerable
            else None,  # str: JSON-safe
            "rr": reciprocal_rank(docs, gold) if t.answerable else None,
        }

        # --- full agent ---
        state = await run_agent(
            graph, {"id": t.id, "subject": t.subject, "body": t.body, "sender_email": ""}
        )
        m = state.get("metrics", [])
        draft = state.get("draft")
        drafted = state.get("outcome") == "draft_ready" and draft is not None
        rec["agent"] = {
            "outcome": state.get("outcome"),
            "reason": state.get("reason", ""),
            "flags": state.get("flags", []),
            "classification": state.get("classification"),
            "draft": draft,
            "discarded_draft": state.get("discarded_draft"),
            "check": state.get("check"),
            "latency_ms": sum(x.get("latency_ms", 0) for x in m),
            "cost_usd": sum(x.get("cost_usd", 0) for x in m),
            "prompt_tokens": sum(x.get("prompt_tokens", 0) for x in m),
            "completion_tokens": sum(x.get("completion_tokens", 0) for x in m),
        }
        rec["class"] = outcome_class(t.answerable, drafted)

        # --- injection leak check (final draft AND the discarded one, separately) ---
        def leaked(text: str | None) -> bool:
            return bool(text) and any(f.lower() in text.lower() for f in t.forbidden)

        rec["leak"] = {
            "final": leaked(draft["text"]) if drafted else False,
            "discarded": leaked((state.get("discarded_draft") or {}).get("reply_text")),
        }

        # --- LLM judge, only on drafts a human would actually see ---
        if drafted:
            try:
                rec["judge"] = await judge_draft(
                    llm, cfg.judge_model, state["passages"], draft["text"], t.gold_facts
                )
            except Exception as e:
                rec["judge"] = {"failed": f"{type(e).__name__}: {e}"}
    except Exception as e:  # one bad ticket must not sink the whole run
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


async def evaluate(tickets: list[EvalTicket], search, llm, cfg: EvalConfig) -> list[dict]:
    memo = MemoSearch(search)
    graph = build_graph(
        AgentDeps(llm=llm, search=memo, model=cfg.model, min_confidence=cfg.min_confidence, k=cfg.k)
    )
    sem = asyncio.Semaphore(cfg.concurrency)

    async def guarded(t: EvalTicket) -> dict:
        async with sem:
            return await _evaluate_one(t, graph, memo, llm, cfg)

    return await asyncio.gather(*(guarded(t) for t in tickets))
