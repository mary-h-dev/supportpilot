"""Business logic behind the MCP tools. Pure async functions (no MCP imports) so they can be
unit-tested against Postgres directly. The MCP server is a thin wrapper over these.

Rule that matters most: send_reply succeeds only for a draft a HUMAN approved. That check is
one atomic SQL UPDATE here, on the server side, and is backed by CHECK constraints in the DB.
"""

import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from .kb.embeddings import Embedder
from .kb.retrieval import hybrid_search
from .logging import log
from .schemas import TicketStatus


class ToolRefused(Exception):
    """Expected, user-facing refusal (bad input, not approved, not found...)."""


@dataclass
class ToolContext:
    session_factory: Callable[[], AsyncSession]
    embedder: Embedder


def _uuid(value: str, what: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except ValueError:
        raise ToolRefused(f"invalid {what}: not a UUID") from None


def _jsonable(row) -> dict:
    out = {}
    for k, v in dict(row).items():
        out[k] = (
            str(v) if isinstance(v, uuid.UUID) else v.isoformat() if isinstance(v, datetime) else v
        )
    return out


async def _audit(s: AsyncSession, ticket_id, event: str, payload: dict) -> None:
    await s.execute(
        text("INSERT INTO audit_log (ticket_id, event, payload) VALUES (:t, :e, :p)"),
        {"t": ticket_id, "e": event, "p": json.dumps(payload, ensure_ascii=False)},
    )


# ---------------------------------------------------------------- read tools
async def search_kb(ctx: ToolContext, query: str, k: int = 5) -> dict:
    query = (query or "").strip()
    if not query or len(query) > 2000:
        raise ToolRefused("query must be 1..2000 characters")
    k = max(1, min(int(k), 10))
    async with ctx.session_factory() as s:
        passages = await hybrid_search(s, ctx.embedder, query, k=k)
    return {"passages": [asdict(p) for p in passages]}


async def get_ticket(ctx: ToolContext, ticket_id: str) -> dict:
    """NOTE: subject/body are untrusted customer text; callers must treat them as data."""
    tid = _uuid(ticket_id, "ticket_id")
    async with ctx.session_factory() as s:
        row = (
            (
                await s.execute(
                    text(
                        "SELECT id, sender_email, subject, body, language, category, urgency, "
                        "status, created_at FROM tickets WHERE id = :i"
                    ),
                    {"i": tid},
                )
            )
            .mappings()
            .first()
        )
    if not row:
        raise ToolRefused("ticket not found")
    return _jsonable(row)


async def list_tickets(ctx: ToolContext, status: str | None = None, limit: int = 20) -> dict:
    """Summaries only (no body): least data needed."""
    if status is not None and status not in {s.value for s in TicketStatus}:
        raise ToolRefused(f"invalid status; one of {sorted(s.value for s in TicketStatus)}")
    limit = max(1, min(int(limit), 100))
    q = "SELECT id, subject, status, category, urgency, created_at FROM tickets"
    params: dict = {"n": limit}
    if status:
        q += " WHERE status = :st"
        params["st"] = status
    q += " ORDER BY created_at DESC LIMIT :n"
    async with ctx.session_factory() as s:
        rows = (await s.execute(text(q), params)).mappings().all()
    return {"tickets": [_jsonable(r) for r in rows]}


# --------------------------------------------------------------- write tools
async def save_draft(
    ctx: ToolContext, ticket_id: str, text_: str, sources: list[str], confidence: float
) -> dict:
    tid = _uuid(ticket_id, "ticket_id")
    body = (text_ or "").strip()
    if not body or len(body) > 10_000:
        raise ToolRefused("draft text must be 1..10000 characters")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ToolRefused("confidence must be within [0, 1]")
    ids = list(dict.fromkeys(sources or []))
    if not ids or len(ids) > 20:
        raise ToolRefused("a draft must cite 1..20 sources (ungrounded drafts are not saved)")

    async with ctx.session_factory() as s:
        ticket = (
            await s.execute(text("SELECT status FROM tickets WHERE id = :i"), {"i": tid})
        ).first()
        if not ticket:
            raise ToolRefused("ticket not found")
        if ticket.status == TicketStatus.SENT:
            raise ToolRefused("ticket already answered")
        found = (
            await s.execute(
                text(
                    "SELECT d.source_uri || '#' || c.position AS sid, d.title "
                    "FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id "
                    "WHERE d.source_uri || '#' || c.position IN :ids"
                ).bindparams(bindparam("ids", expanding=True)),
                {"ids": ids},
            )
        ).all()
        titles = {r.sid: r.title for r in found}
        if missing := [i for i in ids if i not in titles]:
            raise ToolRefused(f"unknown source ids (not in KB): {missing}")

        await s.execute(
            text("UPDATE drafts SET status='superseded' WHERE ticket_id=:t AND status='pending'"),
            {"t": tid},
        )
        draft_id = (
            await s.execute(
                text(
                    "INSERT INTO drafts (ticket_id, text, sources, confidence) "
                    "VALUES (:t, :x, CAST(:s AS jsonb), :c) RETURNING id"
                ),
                {
                    "t": tid,
                    "x": body,
                    "c": float(confidence),
                    "s": json.dumps(
                        [{"source_id": i, "title": titles[i]} for i in ids], ensure_ascii=False
                    ),
                },
            )
        ).scalar_one()
        await s.execute(
            text("UPDATE tickets SET status='awaiting_approval' WHERE id=:t"), {"t": tid}
        )
        await _audit(
            s,
            tid,
            "draft_saved",
            {"draft_id": str(draft_id), "confidence": confidence, "sources": ids},
        )
        await s.commit()
    return {"draft_id": str(draft_id), "status": "pending"}


_BLOCK_REASONS = {
    "pending": "draft has not been approved by a human",
    "rejected": "draft was rejected by a human",
    "sent": "draft was already sent",
    "superseded": "draft was superseded by a newer draft",
}


async def send_reply(ctx: ToolContext, ticket_id: str, draft_id: str) -> dict:
    """Mock-sends the approved reply. Refuses anything that is not an approved draft."""
    tid, did = _uuid(ticket_id, "ticket_id"), _uuid(draft_id, "draft_id")
    async with ctx.session_factory() as s:
        # ONE atomic statement: check approval + mark sent. No read-then-write race, so two
        # concurrent calls cannot both send, and an unapproved draft cannot slip through.
        row = (
            (
                await s.execute(
                    text(
                        "UPDATE drafts d SET status='sent', sent_at=now() FROM tickets t "
                        "WHERE d.id=:d AND d.ticket_id=:t AND t.id=d.ticket_id "
                        "AND d.status='approved' AND d.decided_at IS NOT NULL "
                        "RETURNING COALESCE(d.final_text, d.text) AS body, "
                        "t.sender_email AS to_addr, t.subject AS subject"
                    ),
                    {"d": did, "t": tid},
                )
            )
            .mappings()
            .first()
        )

        if row is None:
            draft = (
                await s.execute(
                    text("SELECT status, ticket_id FROM drafts WHERE id=:d"), {"d": did}
                )
            ).first()
            if draft is None:
                reason = "draft not found"
            elif draft.ticket_id != tid:
                reason = "draft does not belong to this ticket"
            else:
                reason = _BLOCK_REASONS.get(draft.status, f"draft is not approved ({draft.status})")
            exists = (
                await s.execute(text("SELECT 1 FROM tickets WHERE id=:t"), {"t": tid})
            ).first()
            await _audit(
                s, tid if exists else None, "send_blocked", {"draft_id": str(did), "reason": reason}
            )
            await s.commit()
            log.warning("send_reply_blocked", ticket_id=str(tid), draft_id=str(did), reason=reason)
            raise ToolRefused(f"send_reply refused: {reason}")

        message_id = f"mock-{uuid.uuid4()}"
        await s.execute(text("UPDATE tickets SET status='sent' WHERE id=:t"), {"t": tid})
        await _audit(
            s,
            tid,
            "reply_sent",
            {
                "draft_id": str(did),
                "to": row["to_addr"],
                "subject": f"Re: {row['subject']}",
                "message_id": message_id,
            },
        )
        await s.commit()
    # MOCK: a real implementation would hand this to SMTP/an API via a transactional outbox.
    log.info(
        "mock_email_sent",
        to=row["to_addr"],
        subject=f"Re: {row['subject']}",
        message_id=message_id,
        chars=len(row["body"]),
    )
    return {"status": "sent", "message_id": message_id, "to": row["to_addr"]}
