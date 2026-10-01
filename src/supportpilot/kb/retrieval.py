"""Hybrid retrieval: pgvector cosine + Postgres full-text, fused with Reciprocal Rank Fusion."""

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .embeddings import Embedder, to_pgvector
from .normalize import normalize, query_terms

RRF_K = 60

_BASE = (
    "SELECT c.id::text AS id, d.source_uri || '#' || c.position AS source_id, d.title, c.content"
)
_JOIN = "FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id"

VECTOR_SQL = text(
    f"{_BASE}, 1 - (c.embedding <=> CAST(:q AS text)::vector) AS score {_JOIN} "
    "WHERE c.embedding IS NOT NULL "
    "ORDER BY c.embedding <=> CAST(:q AS text)::vector LIMIT :n"
)
FTS_SQL = text(
    f"{_BASE}, ts_rank_cd(c.tsv, q) AS score {_JOIN}, to_tsquery('simple', :tsq) q "
    "WHERE c.tsv @@ q ORDER BY score DESC LIMIT :n"
)


@dataclass
class Passage:
    chunk_id: str
    source_id: str  # e.g. "en/refund-policy.md#0" -- what the LLM cites
    title: str
    content: str
    rrf_score: float
    vector_sim: float | None = None  # cosine similarity, None if not in vector top-N
    fts_rank: float | None = None  # ts_rank_cd, None if no keyword match


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


async def hybrid_search(
    session: AsyncSession, embedder: Embedder, query: str, k: int = 5, pool: int = 20
) -> list[Passage]:
    norm_q = normalize(query)
    [qvec] = await embedder.embed([norm_q])
    vec_rows = (
        (await session.execute(VECTOR_SQL, {"q": to_pgvector(qvec), "n": pool})).mappings().all()
    )
    terms = query_terms(query)
    fts_rows = []
    if terms:
        tsq = " | ".join(terms)  # tokens are \w+ only -> safe to interpolate into tsquery
        fts_rows = (await session.execute(FTS_SQL, {"tsq": tsq, "n": pool})).mappings().all()

    by_id: dict[str, Passage] = {}
    for row in [*vec_rows, *fts_rows]:
        by_id.setdefault(
            row["id"],
            Passage(row["id"], row["source_id"], row["title"], row["content"], 0.0),
        )
    for row in vec_rows:
        by_id[row["id"]].vector_sim = float(row["score"])
    for row in fts_rows:
        by_id[row["id"]].fts_rank = float(row["score"])

    fused = rrf_fuse([[r["id"] for r in vec_rows], [r["id"] for r in fts_rows]])
    for cid, score in fused.items():
        by_id[cid].rrf_score = score
    return sorted(by_id.values(), key=lambda p: p.rrf_score, reverse=True)[:k]
