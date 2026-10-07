"""Wires the production runtime: LLM + MCP server (stdio subprocess) + Postgres checkpointer."""

from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.ext.asyncio import AsyncSession

from .agent.graph import AgentDeps, build_graph
from .agent.search import MCPKBSearch
from .config import settings
from .db import SessionLocal
from .llm import LLMError, OpenAICompatLLM
from .logging import log
from .mcp_client import ALL_TOOLS, SENDER_TOOLS, TRIAGE_TOOLS, connect_tools


@dataclass
class AgentRuntime:
    graph: object
    session_factory: Callable[[], AsyncSession]


def psycopg_url(sqlalchemy_url: str) -> str:
    """The LangGraph checkpointer uses psycopg, which wants a plain postgresql:// URL."""
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def build_runtime(stack: AsyncExitStack) -> AgentRuntime | None:
    """None (not a crash) when the agent can't start, e.g. no LLM_API_KEY: the API still serves."""
    try:
        llm = OpenAICompatLLM()
    except LLMError as e:
        log.warning("agent_unavailable", reason=str(e))
        return None
    mcp = await stack.enter_async_context(connect_tools(ALL_TOOLS))
    triage = mcp.restricted(TRIAGE_TOOLS)  # one connection, three different privilege levels
    saver = await stack.enter_async_context(
        AsyncPostgresSaver.from_conn_string(psycopg_url(settings.database_url))
    )
    await saver.setup()  # library-managed checkpoint tables
    deps = AgentDeps(
        llm=llm,
        search=MCPKBSearch(triage),
        model=settings.llm_model,
        min_confidence=settings.agent_min_confidence,
        k=settings.agent_retrieval_k,
        tools=triage,
        sender=mcp.restricted(SENDER_TOOLS),
    )
    return AgentRuntime(graph=build_graph(deps, saver), session_factory=SessionLocal)
