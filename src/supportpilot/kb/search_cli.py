"""Try retrieval from the terminal:  python -m supportpilot.kb.search_cli "your question" """

import asyncio
import sys

from ..db import SessionLocal
from .embeddings import get_embedder
from .retrieval import hybrid_search


async def main(query: str) -> None:
    async with SessionLocal() as session:
        for p in await hybrid_search(session, get_embedder(), query, k=5):
            sim = f"{p.vector_sim:.3f}" if p.vector_sim is not None else "  -  "
            fts = f"{p.fts_rank:.3f}" if p.fts_rank is not None else "  -  "
            print(f"rrf={p.rrf_score:.4f} vec={sim} fts={fts}  {p.source_id}")
            print(f"    {p.content[:110].replace(chr(10), ' ')}")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:])))
