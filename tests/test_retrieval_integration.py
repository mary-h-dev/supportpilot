"""Needs a real Postgres+pgvector with migrations applied. Skipped unless TEST_DATABASE_URL is set.

Uses a deterministic fake embedder (hashed bag of words) so no Ollama is needed:
it verifies the SQL/plumbing (ingest, FTS, vector, RRF), not embedding quality.
"""

import hashlib
import os
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from supportpilot.kb.ingest import ingest_document, parse_article
from supportpilot.kb.normalize import query_terms
from supportpilot.kb.retrieval import hybrid_search

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


class FakeEmbedder:
    dim = 1024

    async def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in query_terms(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out


@pytest.fixture
async def session():
    engine = create_async_engine(URL)
    async with async_sessionmaker(engine, expire_on_commit=False)() as s:
        await s.execute(text("DELETE FROM kb_documents"))
        root = Path("kb")
        for f in sorted(root.glob("*/*.md")):
            await ingest_document(s, FakeEmbedder(), *parse_article(f, root))
        yield s
        await s.execute(text("DELETE FROM kb_documents"))
        await s.commit()
    await engine.dispose()


async def test_english_query_finds_refund_article(session):
    hits = await hybrid_search(session, FakeEmbedder(), "I want a refund for my payment", k=3)
    assert hits[0].source_id.startswith("en/refund-policy.md")


async def test_persian_query_finds_cancel_article(session):
    hits = await hybrid_search(session, FakeEmbedder(), "چطور اشتراکم را لغو کنم؟", k=3)
    assert hits[0].source_id.startswith("fa/cancel-subscription.md")


async def test_zwnj_and_arabic_letters_still_match(session):
    # Arabic yeh/kaf + no-ZWNJ variants of the same words must hit the same article
    hits = await hybrid_search(session, FakeEmbedder(), "بازيابي رمز عبور", k=3)
    assert hits[0].source_id.startswith("fa/password-reset.md")


async def test_gibberish_has_no_keyword_hits(session):
    hits = await hybrid_search(session, FakeEmbedder(), "qwertyzxcv plmokn", k=3)
    assert all(h.fts_rank is None for h in hits)


async def test_reingest_is_idempotent(session):
    root = Path("kb")
    f = sorted(root.glob("*/*.md"))[0]
    await ingest_document(session, FakeEmbedder(), *parse_article(f, root))
    n = (await session.execute(text("SELECT count(*) FROM kb_documents"))).scalar_one()
    assert n == len(list(root.glob("*/*.md")))
