import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeEmbedder  # noqa: F401  (keeps fakes importable from this module)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from supportpilot.mcp_client import (
    SENDER_TOOLS,
    TRIAGE_TOOLS,
    MCPToolClient,
    ToolCallError,
    ToolNotAllowed,
    _server_env,
    connect_tools,
)


class FakeSession:
    def __init__(self, **result):
        self.calls = []
        self.result = {
            "content": [SimpleNamespace(text='{"a": 1}')],
            "isError": False,
            "structuredContent": None,
        } | result

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return SimpleNamespace(**self.result)


async def test_triage_role_cannot_send_reply_and_server_is_never_called():
    sess = FakeSession()
    client = MCPToolClient(sess, TRIAGE_TOOLS)
    with pytest.raises(ToolNotAllowed):
        await client.call("send_reply", {"ticket_id": "x", "draft_id": "y"})
    assert sess.calls == []  # blocked client-side, request never left the process


async def test_sender_role_cannot_search_or_draft():
    client = MCPToolClient(FakeSession(), SENDER_TOOLS)
    for name in ("search_kb", "save_draft", "get_ticket"):
        with pytest.raises(ToolNotAllowed):
            await client.call(name, {})


async def test_parses_text_and_structured_results_and_errors():
    assert await MCPToolClient(FakeSession(), TRIAGE_TOOLS).call("search_kb", {}) == {"a": 1}
    structured = FakeSession(structuredContent={"b": 2})
    assert await MCPToolClient(structured, TRIAGE_TOOLS).call("search_kb", {}) == {"b": 2}
    err = FakeSession(isError=True, content=[SimpleNamespace(text="refused: nope")])
    with pytest.raises(ToolCallError, match="refused: nope"):
        await MCPToolClient(err, TRIAGE_TOOLS).call("save_draft", {})


def test_tool_server_never_receives_the_llm_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x")
    env = _server_env()
    assert "LLM_API_KEY" not in env and env["DATABASE_URL"] == "postgresql+asyncpg://x"


URL = os.environ.get("TEST_DATABASE_URL")


@pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
async def test_real_mcp_server_over_stdio_end_to_end(monkeypatch):
    """Spawns the real server subprocess. A clean initialize also proves stdout carries only
    protocol messages (logs go to stderr)."""
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setenv("PYTHONPATH", str(Path("src").resolve()))
    engine = create_async_engine(URL)
    async with engine.begin() as c:
        await c.execute(text("DELETE FROM audit_log"))
        await c.execute(text("DELETE FROM tickets"))
        await c.execute(text("DELETE FROM kb_documents"))
        doc = (
            await c.execute(
                text(
                    "INSERT INTO kb_documents (title, source_uri, language) "
                    "VALUES ('Refund policy', 'en/refund-policy.md', 'en') RETURNING id"
                )
            )
        ).scalar_one()
        await c.execute(
            text(
                "INSERT INTO kb_chunks (document_id, position, content, content_norm) "
                "VALUES (:d, 0, 'Full refund within 14 days.', 'full refund within 14 days')"
            ),
            {"d": doc},
        )
        tid = str(
            (
                await c.execute(
                    text(
                        "INSERT INTO tickets (sender_email, subject, body) "
                        "VALUES ('c@x.com', 'Refund', 'money back') RETURNING id"
                    )
                )
            ).scalar_one()
        )

    everything = TRIAGE_TOOLS | SENDER_TOOLS | {"list_tickets"}
    async with connect_tools(everything) as client:
        names = {t.name for t in (await client.session.list_tools()).tools}
        assert names == {"search_kb", "get_ticket", "list_tickets", "save_draft", "send_reply"}

        listed = await client.call("list_tickets", {"status": "new"})
        assert [t["id"] for t in listed["tickets"]] == [tid]

        saved = await client.call(
            "save_draft",
            {
                "ticket_id": tid,
                "text": "You can get a refund.",
                "sources": ["en/refund-policy.md#0"],
                "confidence": 0.9,
            },
        )
        with pytest.raises(ToolCallError, match="not been approved"):
            await client.call("send_reply", {"ticket_id": tid, "draft_id": saved["draft_id"]})

        async with engine.begin() as c:  # the human decision (step 5 will own this)
            await c.execute(
                text(
                    "UPDATE drafts SET status='approved', decided_at=now(), decided_by='h' "
                    "WHERE id=:d"
                ),
                {"d": saved["draft_id"]},
            )
        sent = await client.call("send_reply", {"ticket_id": tid, "draft_id": saved["draft_id"]})
        assert sent["status"] == "sent"
    await engine.dispose()
