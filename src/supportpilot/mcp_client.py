"""MCP client used by the agent, with a per-role TOOL ALLOWLIST.

Least privilege: the triage agent can search/read/save drafts but can NEVER call send_reply.
Only the post-approval sender role can. (The server enforces approval independently: the
allowlist is defense in depth against a bug or a compromised agent, not the only gate.)
"""

import json
import os
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

TRIAGE_TOOLS = frozenset({"search_kb", "get_ticket", "save_draft"})
SENDER_TOOLS = frozenset({"send_reply"})
ALL_TOOLS = TRIAGE_TOOLS | SENDER_TOOLS | {"list_tickets"}

# The tool server gets only what it needs. In particular it never sees LLM_API_KEY.
_SERVER_ENV_KEYS = (
    "DATABASE_URL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "OLLAMA_URL",
    "PATH",
    "PYTHONPATH",
    "SYSTEMROOT",
    "PYTHONUNBUFFERED",
)


class ToolNotAllowed(Exception):
    pass


class ToolCallError(Exception):
    """The server executed the tool and reported an error (e.g. send_reply refused)."""


class MCPToolClient:
    def __init__(self, session: ClientSession, allowed: frozenset[str]):
        self.session, self.allowed = session, allowed

    def restricted(self, allowed: frozenset[str]) -> "MCPToolClient":
        """A view over the SAME connection with a narrower allowlist (never wider)."""
        return MCPToolClient(self.session, self.allowed & allowed)

    async def call(self, name: str, args: dict) -> dict:
        if name not in self.allowed:
            raise ToolNotAllowed(f"tool '{name}' is not allowed for this role")
        res = await self.session.call_tool(name, args)
        text = " ".join(getattr(c, "text", "") for c in res.content).strip()
        if res.isError:
            raise ToolCallError(text or "tool error")
        if res.structuredContent:
            return res.structuredContent
        return json.loads(text)


def _server_env() -> dict[str, str]:
    return {k: os.environ[k] for k in _SERVER_ENV_KEYS if k in os.environ}


@asynccontextmanager
async def connect_tools(allowed: frozenset[str]):
    """Spawn the MCP server as a stdio subprocess and yield a role-restricted client."""
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "supportpilot.mcp_server"], env=_server_env()
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield MCPToolClient(session, allowed)
