"""Try the agent from the terminal (no DB writes):
python -m supportpilot.agent.cli "subject" "body"            # retrieval via the MCP server
python -m supportpilot.agent.cli --direct "subject" "body"   # retrieval straight from the DB
"""

import asyncio
import json
import sys
from contextlib import AsyncExitStack

from ..config import settings
from ..db import SessionLocal
from ..kb.embeddings import get_embedder
from ..llm import OpenAICompatLLM
from ..logging import setup_logging
from ..mcp_client import TRIAGE_TOOLS, connect_tools
from .graph import AgentDeps, build_graph, run_agent
from .search import DbKBSearch, MCPKBSearch


async def main(subject: str, body: str, direct: bool) -> None:
    setup_logging()
    async with AsyncExitStack() as stack:
        if direct:
            search = DbKBSearch(SessionLocal, get_embedder())
        else:
            search = MCPKBSearch(await stack.enter_async_context(connect_tools(TRIAGE_TOOLS)))
        deps = AgentDeps(
            llm=OpenAICompatLLM(),
            search=search,
            model=settings.llm_model,
            min_confidence=settings.agent_min_confidence,
            k=settings.agent_retrieval_k,
        )
        state = await run_agent(build_graph(deps), {"subject": subject, "body": body})
    state.pop("passages", None)
    print(json.dumps(state, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--direct"]
    if len(args) < 2:
        sys.exit('usage: python -m supportpilot.agent.cli [--direct] "subject" "body"')
    asyncio.run(main(args[0], args[1], "--direct" in sys.argv))
