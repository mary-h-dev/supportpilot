"""Ingest markdown articles from ./kb/<lang>/*.md. Idempotent: re-running replaces a doc."""

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import SessionLocal
from ..logging import log, setup_logging
from .chunking import chunk_text
from .embeddings import Embedder, get_embedder, to_pgvector
from .normalize import normalize


def parse_article(path: Path, root: Path) -> tuple[str, str, str, str]:
    """-> (source_uri, title, language, body). First '# ' line is the title."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    title = lines[0].lstrip("# ").strip()
    body = "\n".join(lines[1:]).strip()
    return path.relative_to(root).as_posix(), title, path.parent.name, body


async def ingest_document(
    session: AsyncSession, embedder: Embedder, source_uri: str, title: str, lang: str, body: str
) -> int:
    chunks = chunk_text(body)
    # Prefix the title so each chunk carries its topic (helps both BM25-ish and vector match).
    norms = [normalize(f"{title}. {c}") for c in chunks]
    vectors = await embedder.embed(norms)

    await session.execute(text("DELETE FROM kb_documents WHERE source_uri = :u"), {"u": source_uri})
    doc_id = (
        await session.execute(
            text(
                "INSERT INTO kb_documents (title, source_uri, language) "
                "VALUES (:t, :u, :l) RETURNING id"
            ),
            {"t": title, "u": source_uri, "l": lang},
        )
    ).scalar_one()
    for pos, (chunk, norm, vec) in enumerate(zip(chunks, norms, vectors, strict=True)):
        await session.execute(
            text(
                "INSERT INTO kb_chunks (document_id, position, content, content_norm, embedding) "
                "VALUES (:d, :p, :c, :n, CAST(:e AS text)::vector)"
            ),
            {"d": doc_id, "p": pos, "c": chunk, "n": norm, "e": to_pgvector(vec)},
        )
    await session.commit()
    return len(chunks)


async def main(root: Path = Path("kb")) -> None:
    setup_logging()
    embedder = get_embedder()
    files = sorted(root.glob("*/*.md"))
    if not files:
        sys.exit(f"No .md files under {root}/<lang>/")
    async with SessionLocal() as session:
        for f in files:
            n = await ingest_document(session, embedder, *parse_article(f, root))
            log.info("kb_ingested", source=f.as_posix(), chunks=n)


if __name__ == "__main__":
    asyncio.run(main())
