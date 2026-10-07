"""Ticket lifecycle around the agent graph: process, record the human decision, resume."""

import json
import uuid
from difflib import SequenceMatcher

from langgraph.types import Command
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .logging import log
from .runtime import AgentRuntime
from .schemas import Decision, DecisionIn


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


def _cfg(ticket_id) -> dict:
    return {"configurable": {"thread_id": str(ticket_id)}}


async def _audit(s: AsyncSession, ticket_id, event: str, payload: dict) -> None:
    await s.execute(
        text("INSERT INTO audit_log (ticket_id, event, payload) VALUES (:t, :e, :p)"),
        {"t": ticket_id, "e": event, "p": json.dumps(payload, ensure_ascii=False, default=str)},
    )


async def process_ticket(rt: AgentRuntime, ticket_id: uuid.UUID) -> None:
    """Run the agent for a ticket. Never raises: any crash ends in needs_human (fail-safe)."""
    async with rt.session_factory() as s:
        t = (
            (
                await s.execute(
                    text("SELECT id, subject, body, sender_email FROM tickets WHERE id=:i"),
                    {"i": ticket_id},
                )
            )
            .mappings()
            .first()
        )
        if not t:
            return
        await s.execute(
            text("UPDATE tickets SET status='processing' WHERE id=:i"), {"i": ticket_id}
        )
        await _audit(s, ticket_id, "processing_started", {})
        await s.commit()

    ticket = {k: str(v) if k == "id" else v for k, v in dict(t).items()}
    try:
        result = await rt.graph.ainvoke(
            {"ticket": ticket, "metrics": [], "flags": []}, _cfg(ticket_id)
        )
        interrupted = "__interrupt__" in result
        outcome, reason = result.get("outcome"), result.get("reason", "")
    except Exception as e:
        log.exception("agent_crashed", ticket_id=str(ticket_id))
        result, interrupted, outcome, reason = {}, False, "needs_human", "agent_crashed"
        result["error"] = type(e).__name__

    metrics = result.get("metrics", [])
    cls = result.get("classification") or {}
    async with rt.session_factory() as s:
        if cls:
            await s.execute(
                text("UPDATE tickets SET language=:l, category=:c, urgency=:u WHERE id=:i"),
                {
                    "l": cls.get("language"),
                    "c": cls.get("category"),
                    "u": cls.get("urgency"),
                    "i": ticket_id,
                },
            )
        if not interrupted:
            await s.execute(
                text("UPDATE tickets SET status='needs_human' WHERE id=:i"), {"i": ticket_id}
            )
        await _audit(
            s,
            ticket_id,
            "agent_completed",
            {
                "outcome": "awaiting_approval" if interrupted else outcome,
                "reason": reason,
                "flags": result.get("flags", []),
                "needs_human": not interrupted,
                "discarded_draft": result.get("discarded_draft"),
                "cost_usd": round(sum(m.get("cost_usd", 0) for m in metrics), 8),
                "latency_ms": round(sum(m.get("latency_ms", 0) for m in metrics), 1),
                "metrics": metrics,
            },
        )
        await s.commit()
    log.info(
        "ticket_processed", ticket_id=str(ticket_id), awaiting_approval=interrupted, reason=reason
    )


async def _record_decision(rt: AgentRuntime, ticket_id, draft_id, body: DecisionIn) -> None:
    """Atomic: succeeds only if the draft is STILL pending, so two reviewers can't both win."""
    approved = body.decision != Decision.REJECT
    async with rt.session_factory() as s:
        row = (
            (
                await s.execute(
                    text(
                        "UPDATE drafts SET status=:st, final_text=:f, decided_at=now(), "
                        "decided_by=:by "
                        "WHERE id=:d AND ticket_id=:t AND status='pending' RETURNING text"
                    ),
                    {
                        "st": "approved" if approved else "rejected",
                        "f": body.edited_text.strip() if body.decision == Decision.EDIT else None,
                        "by": body.reviewer,
                        "d": draft_id,
                        "t": ticket_id,
                    },
                )
            )
            .mappings()
            .first()
        )
        if not row:
            raise Conflict("this draft was already decided")
        payload = {
            "draft_id": str(draft_id),
            "decision": body.decision.value,
            "reviewer": body.reviewer,
            "reason": body.reason,
        }
        if body.decision == Decision.EDIT:  # 0.0 = untouched, 1.0 = fully rewritten
            ratio = SequenceMatcher(None, row["text"], body.edited_text.strip()).ratio()
            payload["edit_ratio"] = round(1 - ratio, 3)
        if not approved:
            await s.execute(
                text("UPDATE tickets SET status='rejected' WHERE id=:i"), {"i": ticket_id}
            )
        await _audit(s, ticket_id, "decision", payload)
        await s.commit()


async def decide(rt: AgentRuntime, ticket_id: uuid.UUID, body: DecisionIn) -> None:
    cfg = _cfg(ticket_id)
    async with rt.session_factory() as s:
        if not (
            await s.execute(text("SELECT 1 FROM tickets WHERE id=:i"), {"i": ticket_id})
        ).first():
            raise NotFound("ticket not found")
        draft = (
            (
                await s.execute(
                    text(
                        "SELECT id, status, final_text FROM drafts WHERE ticket_id=:t "
                        "AND status <> 'superseded' ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"t": ticket_id},
                )
            )
            .mappings()
            .first()
        )
    if not draft:
        raise Conflict("no draft is awaiting a decision for this ticket")

    state = await rt.graph.aget_state(cfg)
    nxt = tuple(state.next)
    if nxt == ("await_human",) and draft["status"] == "pending":
        await _record_decision(rt, ticket_id, draft["id"], body)
        await rt.graph.ainvoke(
            Command(resume={"decision": body.decision.value, "reviewer": body.reviewer}), cfg
        )
    elif nxt == ("finalize",) and draft["status"] == "approved":
        # An earlier attempt recorded the decision but sending failed: retry the same decision.
        stored = state.values["decision"]["decision"]
        same_text = body.decision != Decision.EDIT or (body.edited_text or "").strip() == (
            draft["final_text"] or ""
        )
        if stored != body.decision.value or not same_text:
            raise Conflict("a different decision was already recorded; retry that decision")
        await rt.graph.ainvoke(None, cfg)
    else:
        raise Conflict("ticket is not awaiting a decision")


async def ticket_detail(s: AsyncSession, ticket_id: uuid.UUID) -> dict:
    t = (
        (
            await s.execute(
                text(
                    "SELECT id, sender_email, subject, body, language, category, urgency, status, "
                    "created_at FROM tickets WHERE id=:i"
                ),
                {"i": ticket_id},
            )
        )
        .mappings()
        .first()
    )
    if not t:
        raise NotFound("ticket not found")
    draft = (
        (
            await s.execute(
                text(
                    "SELECT id, text, final_text, sources, confidence, status, "
                    "decided_by, decided_at "
                    "FROM drafts WHERE ticket_id=:i AND status <> 'superseded' "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"i": ticket_id},
            )
        )
        .mappings()
        .first()
    )
    audit = (
        (
            await s.execute(
                text(
                    "SELECT event, payload, created_at FROM audit_log "
                    "WHERE ticket_id=:i ORDER BY id"
                ),
                {"i": ticket_id},
            )
        )
        .mappings()
        .all()
    )
    agent = next((a["payload"] for a in reversed(audit) if a["event"] == "agent_completed"), None)
    return {**t, "draft": draft, "audit": [dict(a) for a in audit], "agent": agent}
