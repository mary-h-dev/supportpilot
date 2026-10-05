"""The agent's only KB access point. Step 4 adds an MCP-backed implementation of the same
protocol, so the graph does not change when retrieval moves behind the MCP server."""

from collections.abc import Callable
from dataclasses import asdict
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from ..kb.embeddings import Embedder
from ..kb.retrieval import hybrid_search


class KBSearch(Protocol):
    async def __call__(self, query: str, k: int) -> list[dict]: ...


class DbKBSearch:
    def __init__(self, session_factory: Callable[[], AsyncSession], embedder: Embedder):
        self.session_factory, self.embedder = session_factory, embedder

    async def __call__(self, query: str, k: int) -> list[dict]:
        async with self.session_factory() as session:
            return [asdict(p) for p in await hybrid_search(session, self.embedder, query, k=k)]
