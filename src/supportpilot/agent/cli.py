"""Try the agent from the terminal (no DB writes):
python -m supportpilot.agent.cli "subject" "body"
"""

import asyncio
import json
import sys

from ..config import settings
from ..db import SessionLocal
from ..kb.embeddings import get_embedder
from ..llm import OpenAICompatLLM
from ..logging import setup_logging
from .graph import AgentDeps, build_graph, run_agent
from .search import DbKBSearch


async def main(subject: str, body: str) -> None:
    setup_logging()
    deps = AgentDeps(
        llm=OpenAICompatLLM(),
        search=DbKBSearch(SessionLocal, get_embedder()),
        model=settings.llm_model,
        min_confidence=settings.agent_min_confidence,
        k=settings.agent_retrieval_k,
    )
    state = await run_agent(build_graph(deps), {"subject": subject, "body": body})
    state.pop("passages", None)
    print(json.dumps(state, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('usage: python -m supportpilot.agent.cli "subject" "body"')
    asyncio.run(main(sys.argv[1], sys.argv[2]))
