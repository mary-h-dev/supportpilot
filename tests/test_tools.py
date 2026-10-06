"""Tool logic against real Postgres (skipped without TEST_DATABASE_URL)."""

import os
from pathlib import Path

import pytest
from fakes import FakeEmbedder
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from supportpilot import tools
from supportpilot.kb.ingest import ingest_document, parse_article

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
SRC = "en/refund-policy.md#0"


@pytest.fixture
async def ctx():
    engine = create_async_engine(URL)
    sf = async_sessionmaker(engine, expire_on_commit=False)
    async with sf() as s:
        await s.execute(text("DELETE FROM audit_log"))
        await s.execute(text("DELETE FROM tickets"))  # cascades to drafts
        await s.execute(text("DELETE FROM kb_documents"))
        root = Path("kb")
        for f in sorted(root.glob("*/*.md")):
            await ingest_document(s, FakeEmbedder(), *parse_article(f, root))
    yield tools.ToolContext(sf, FakeEmbedder())
    await engine.dispose()


async def new_ticket(ctx, subject="Refund", body="I want my money back"):
    async with ctx.session_factory() as s:
        tid = (
            await s.execute(
                text(
                    "INSERT INTO tickets (sender_email, subject, body) "
                    "VALUES ('c@x.com', :s, :b) RETURNING id"
                ),
                {"s": subject, "b": body},
            )
        ).scalar_one()
        await s.commit()
    return str(tid)


async def approve(ctx, draft_id, final_text=None, status="approved"):
    """Stands in for the step-5 human decision API."""
    async with ctx.session_factory() as s:
        await s.execute(
            text(
                "UPDATE drafts SET status=:st, final_text=:f, decided_at=now(), "
                "decided_by='human@test' WHERE id=:d"
            ),
            {"st": status, "f": final_text, "d": draft_id},
        )
        await s.commit()


async def one(ctx, sql, **p):
    async with ctx.session_factory() as s:
        return (await s.execute(text(sql), p)).first()


async def draft_for(ctx, tid, body="You can request a refund within 14 days."):
    return (await tools.save_draft(ctx, tid, body, [SRC], 0.9))["draft_id"]


# ---------------------------------------------------------------- read tools
async def test_search_kb_returns_source_ids(ctx):
    out = await tools.search_kb(ctx, "I want a refund", k=3)
    assert out["passages"][0]["source_id"].startswith("en/refund-policy.md")
    assert len(out["passages"]) <= 3


async def test_search_kb_validates_input(ctx):
    with pytest.raises(tools.ToolRefused):
        await tools.search_kb(ctx, "   ")
    assert len((await tools.search_kb(ctx, "refund", k=999))["passages"]) <= 10


async def test_get_and_list_tickets(ctx):
    tid = await new_ticket(ctx)
    t = await tools.get_ticket(ctx, tid)
    assert t["id"] == tid and t["status"] == "new"
    listed = (await tools.list_tickets(ctx, "new"))["tickets"]
    assert [x["id"] for x in listed] == [tid] and "body" not in listed[0]
    with pytest.raises(tools.ToolRefused):
        await tools.get_ticket(ctx, "not-a-uuid")
    with pytest.raises(tools.ToolRefused):
        await tools.list_tickets(ctx, "bogus")


# --------------------------------------------------------------- save_draft
async def test_save_draft_creates_pending_and_audits(ctx):
    tid = await new_ticket(ctx)
    did = await draft_for(ctx, tid)
    assert (await one(ctx, "SELECT status FROM drafts WHERE id=:d", d=did)).status == "pending"
    assert (await one(ctx, "SELECT status FROM tickets WHERE id=:t", t=tid)).status == (
        "awaiting_approval"
    )
    assert (await one(ctx, "SELECT count(*) c FROM audit_log WHERE event='draft_saved'")).c == 1


@pytest.mark.parametrize(
    "kw",
    [
        {"text_": "  "},
        {"sources": []},
        {"sources": ["en/made-up.md#9"]},
        {"confidence": 1.5},
    ],
)
async def test_save_draft_rejects_bad_input(ctx, kw):
    tid = await new_ticket(ctx)
    args = {"text_": "ok", "sources": [SRC], "confidence": 0.5} | kw
    with pytest.raises(tools.ToolRefused):
        await tools.save_draft(ctx, tid, **args)
    assert (await one(ctx, "SELECT count(*) c FROM drafts")).c == 0


async def test_save_draft_unknown_ticket(ctx):
    with pytest.raises(tools.ToolRefused, match="not found"):
        await tools.save_draft(ctx, "00000000-0000-0000-0000-000000000000", "x", [SRC], 0.5)


async def test_new_draft_supersedes_old_pending(ctx):
    tid = await new_ticket(ctx)
    d1, d2 = await draft_for(ctx, tid), await draft_for(ctx, tid, "newer text")
    assert (await one(ctx, "SELECT status FROM drafts WHERE id=:d", d=d1)).status == "superseded"
    assert (await one(ctx, "SELECT status FROM drafts WHERE id=:d", d=d2)).status == "pending"


# --------------------------------------------------------------- send_reply
async def test_send_reply_blocked_until_approved(ctx):
    tid = await new_ticket(ctx)
    did = await draft_for(ctx, tid)
    with pytest.raises(tools.ToolRefused, match="not been approved"):
        await tools.send_reply(ctx, tid, did)
    assert (await one(ctx, "SELECT status FROM drafts WHERE id=:d", d=did)).status == "pending"
    assert (await one(ctx, "SELECT count(*) c FROM audit_log WHERE event='send_blocked'")).c == 1
    assert (await one(ctx, "SELECT count(*) c FROM audit_log WHERE event='reply_sent'")).c == 0


async def test_send_reply_blocked_when_rejected(ctx):
    tid = await new_ticket(ctx)
    did = await draft_for(ctx, tid)
    await approve(ctx, did, status="rejected")
    with pytest.raises(tools.ToolRefused, match="rejected"):
        await tools.send_reply(ctx, tid, did)


async def test_send_reply_after_approval_and_only_once(ctx):
    tid = await new_ticket(ctx)
    did = await draft_for(ctx, tid)
    await approve(ctx, did)
    out = await tools.send_reply(ctx, tid, did)
    assert out["status"] == "sent" and out["message_id"].startswith("mock-")
    assert (await one(ctx, "SELECT status FROM tickets WHERE id=:t", t=tid)).status == "sent"
    ev = await one(ctx, "SELECT payload FROM audit_log WHERE event='reply_sent'")
    assert ev.payload["to"] == "c@x.com" and ev.payload["subject"] == "Re: Refund"
    with pytest.raises(tools.ToolRefused, match="already sent"):  # no double send
        await tools.send_reply(ctx, tid, did)


async def test_send_reply_uses_human_edited_text(ctx):
    tid = await new_ticket(ctx)
    did = await draft_for(ctx, tid)
    await approve(ctx, did, final_text="Edited by a human")
    await tools.send_reply(ctx, tid, did)
    row = await one(ctx, "SELECT text, final_text FROM drafts WHERE id=:d", d=did)
    assert row.text != row.final_text == "Edited by a human"  # both kept: edit-rate metric


async def test_send_reply_wrong_ticket_or_unknown_draft(ctx):
    t1, t2 = await new_ticket(ctx), await new_ticket(ctx, "Other")
    did = await draft_for(ctx, t1)
    await approve(ctx, did)
    with pytest.raises(tools.ToolRefused, match="does not belong"):
        await tools.send_reply(ctx, t2, did)
    with pytest.raises(tools.ToolRefused, match="not found"):
        await tools.send_reply(ctx, t1, "00000000-0000-0000-0000-000000000000")
    assert (await one(ctx, "SELECT status FROM drafts WHERE id=:d", d=did)).status == "approved"


async def test_db_constraint_blocks_sent_without_decision(ctx):
    """Even raw SQL (a buggy API, a human with psql) cannot mark an undecided draft sent."""
    tid = await new_ticket(ctx)
    did = await draft_for(ctx, tid)
    async with ctx.session_factory() as s:
        with pytest.raises(IntegrityError):
            await s.execute(
                text("UPDATE drafts SET status='sent', sent_at=now() WHERE id=:d"), {"d": did}
            )
