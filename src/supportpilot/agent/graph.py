"""LangGraph workflow:  classify -> retrieve -> draft -> self_check -> END

Any failure or doubt routes to outcome="needs_human" with NO draft (fail-safe). Step 5 inserts a
human-approval interrupt after self_check; nothing here ever sends anything.
"""

import html
import time
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..llm import LLM, LLMError, Usage
from ..logging import log
from ..mcp_client import ToolCallError
from .checks import evaluate_draft
from .prompts import CHECK_SYSTEM, CLASSIFY_SYSTEM, DRAFT_SYSTEM, passages_block
from .schemas import AgentState, Classification, DraftOutput, FaithfulnessCheck
from .search import KBSearch
from .security import detect_injection, ticket_block


@dataclass
class AgentDeps:
    llm: LLM
    search: KBSearch
    model: str
    min_confidence: float = 0.6
    k: int = 5
    # Persistence tail (save_draft -> human interrupt -> send). Leave None to stop at self_check.
    tools: object | None = None  # triage-role MCP client (save_draft)
    sender: object | None = None  # sender-role MCP client (send_reply)


def _metric(step: str, t0: float, usage: Usage | None = None) -> dict:
    m = {"step": step, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    if usage:
        m |= {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost_usd": usage.cost_usd,
            "model": usage.model,
        }
    log.info("agent_step", **m)
    return m


def _human(reason: str, metric: dict, **extra) -> dict:
    return {"outcome": "needs_human", "reason": reason, "metrics": [metric], **extra}


def build_graph(deps: AgentDeps, checkpointer=None):
    async def classify(state: AgentState) -> dict:
        t0 = time.perf_counter()
        t = state["ticket"]
        flags = detect_injection(f"{t['subject']}\n{t['body']}")
        try:
            cls, usage = await deps.llm.complete_json(
                step="classify",
                model=deps.model,
                system=CLASSIFY_SYSTEM,
                user=ticket_block(t["subject"], t["body"]),
                schema=Classification,
            )
        except LLMError as e:
            log.warning("classify_failed", error=str(e))
            return _human("llm_error", _metric("classify", t0), flags=flags)
        return {
            "classification": cls.model_dump(),
            "flags": flags,
            "metrics": [_metric("classify", t0, usage)],
        }

    async def retrieve(state: AgentState) -> dict:
        t0 = time.perf_counter()
        t = state["ticket"]
        query = f"{t['subject']}\n{t['body']}"[:1000]
        try:
            passages = await deps.search(query, deps.k)
        except Exception:
            log.exception("retrieval_failed")
            return _human("retrieval_error", _metric("retrieve", t0))
        m = _metric("retrieve", t0)
        if not passages:
            return _human("no_kb_match", m, passages=[])
        return {"passages": passages, "metrics": [m]}

    async def draft(state: AgentState) -> dict:
        t0 = time.perf_counter()
        t = state["ticket"]
        lang = state["classification"]["language"]
        user = f"{ticket_block(t['subject'], t['body'])}\n\n{passages_block(state['passages'])}"
        try:
            out, usage = await deps.llm.complete_json(
                step="draft",
                model=deps.model,
                system=DRAFT_SYSTEM.format(language=lang),
                user=user,
                schema=DraftOutput,
            )
        except LLMError as e:
            log.warning("draft_failed", error=str(e))
            return _human("llm_error", _metric("draft", t0))
        return {"draft": out.model_dump(), "metrics": [_metric("draft", t0, usage)]}

    async def self_check(state: AgentState) -> dict:
        t0 = time.perf_counter()
        t = state["ticket"]
        d = DraftOutput(**state["draft"])
        reason = evaluate_draft(d, state["passages"], deps.min_confidence)
        if reason:  # deterministic guard failed: no LLM call needed
            return _human(
                reason, _metric("self_check", t0), draft=None, discarded_draft=state["draft"]
            )
        user = (
            f"{ticket_block(t['subject'], t['body'])}\n\n{passages_block(state['passages'])}\n\n"
            f"<draft>\n{html.escape(d.reply_text, quote=False)}\n</draft>"
        )
        try:
            chk, usage = await deps.llm.complete_json(
                step="self_check",
                model=deps.model,
                system=CHECK_SYSTEM,
                user=user,
                schema=FaithfulnessCheck,
            )
        except LLMError as e:
            log.warning("self_check_failed", error=str(e))
            return _human(
                "llm_error", _metric("self_check", t0), draft=None, discarded_draft=state["draft"]
            )
        m = _metric("self_check", t0, usage)
        if not chk.supported:
            return _human(
                "unsupported_claims",
                m,
                draft=None,
                discarded_draft=state["draft"],
                check=chk.model_dump(),
            )
        if not chk.answers_question:
            return _human(
                "off_topic", m, draft=None, discarded_draft=state["draft"], check=chk.model_dump()
            )
        titles = {p["source_id"]: p["title"] for p in state["passages"]}
        final = {
            "text": d.reply_text.strip(),
            "sources": [{"source_id": s, "title": titles[s]} for s in d.cited_source_ids],
            "confidence": d.confidence,
        }
        return {
            "outcome": "draft_ready",
            "reason": "",
            "draft": final,
            "check": chk.model_dump(),
            "metrics": [m],
        }

    async def save_draft(state: AgentState) -> dict:
        t0 = time.perf_counter()
        d = state["draft"]
        try:
            res = await deps.tools.call(
                "save_draft",
                {
                    "ticket_id": state["ticket"]["id"],
                    "text": d["text"],
                    "sources": [s["source_id"] for s in d["sources"]],
                    "confidence": d["confidence"],
                },
            )
        except ToolCallError as e:
            log.warning("save_draft_failed", error=str(e))
            return _human(
                "save_draft_failed", _metric("save_draft", t0), discarded_draft=d, draft=None
            )
        return {"draft_id": res["draft_id"], "metrics": [_metric("save_draft", t0)]}

    async def await_human(state: AgentState) -> dict:
        # Pauses the graph (state is checkpointed in Postgres) until POST /decision resumes it.
        # Keep this node tiny: on resume LangGraph re-runs it from the top.
        decision = interrupt(
            {
                "ticket_id": state["ticket"]["id"],
                "draft_id": state["draft_id"],
                "draft": state["draft"],
                "flags": state.get("flags", []),
                "classification": state.get("classification"),
            }
        )
        return {"decision": decision}

    async def finalize(state: AgentState) -> dict:
        t0 = time.perf_counter()
        if state["decision"]["decision"] == "reject":
            return {
                "outcome": "rejected",
                "reason": "human_rejected",
                "metrics": [_metric("finalize", t0)],
            }
        # Not caught on purpose: if sending fails the node fails, the checkpoint stays just
        # before it, and re-posting the same decision retries from here (see service.decide).
        res = await deps.sender.call(
            "send_reply", {"ticket_id": state["ticket"]["id"], "draft_id": state["draft_id"]}
        )
        return {
            "outcome": "sent",
            "message_id": res["message_id"],
            "metrics": [_metric("finalize", t0)],
        }

    def stop_if_human(next_node: str):
        return lambda state: END if state.get("outcome") == "needs_human" else next_node

    g = StateGraph(AgentState)
    for name, fn in [
        ("classify", classify),
        ("retrieve", retrieve),
        ("draft", draft),
        ("self_check", self_check),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", stop_if_human("retrieve"), ["retrieve", END])
    g.add_conditional_edges("retrieve", stop_if_human("draft"), ["draft", END])
    g.add_conditional_edges("draft", stop_if_human("self_check"), ["self_check", END])
    if deps.tools is not None and deps.sender is not None:
        g.add_node("save_draft", save_draft)
        g.add_node("await_human", await_human)
        g.add_node("finalize", finalize)
        g.add_conditional_edges("self_check", stop_if_human("save_draft"), ["save_draft", END])
        g.add_conditional_edges("save_draft", stop_if_human("await_human"), ["await_human", END])
        g.add_edge("await_human", "finalize")
        g.add_edge("finalize", END)
    else:
        g.add_edge("self_check", END)
    return g.compile(checkpointer=checkpointer)


async def run_agent(graph, ticket: dict, config: dict | None = None) -> AgentState:
    return await graph.ainvoke({"ticket": ticket, "metrics": [], "flags": []}, config)
