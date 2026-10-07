"""`python -m supportpilot.eval run` end to end on a real Postgres, with a scripted LLM and a fake
embedder (so no Ollama / API key). Checks the plumbing and the written artifacts, NOT quality."""

import json
import os
from argparse import Namespace

import pytest
from fakes import FakeEmbedder
from test_eval import oracle_llm

from supportpilot.eval import __main__ as cli
from supportpilot.eval import report

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture
def patched(monkeypatch, tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import supportpilot.db as db
    import supportpilot.kb.embeddings as emb
    import supportpilot.kb.ingest as ingest
    import supportpilot.llm as llm_mod

    engine = create_async_engine(URL)
    sf = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(db, "SessionLocal", sf)
    monkeypatch.setattr(ingest, "SessionLocal", sf)
    monkeypatch.setattr(emb, "get_embedder", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr(ingest, "get_embedder", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr(llm_mod, "OpenAICompatLLM", lambda *a, **k: oracle_llm())
    return tmp_path


def ns(split, tmp, **kw):
    base = dict(split=split, final=False, ingest=True, concurrency=4, no_cache=True, out=str(tmp))
    return Namespace(**{**base, **kw})


async def test_dev_run_writes_report_json_and_review_sheet(patched):
    await cli.cmd_run(ns("dev", patched))
    files = {p.name for p in patched.iterdir()}
    assert "latest-dev.md" in files
    assert any(f.startswith("review-dev-") and f.endswith(".csv") for f in files)
    blob = json.loads(next(patched.glob("dev-*.json")).read_text(encoding="utf-8"))
    assert len(blob["results"]) == 20 and blob["meta"]["split"] == "dev"
    assert blob["meta"]["prompt_version"] and len(blob["meta"]["prompts_sha"]) == 10
    s = report.summarize(blob["results"])
    assert s["n_errors"] == 0
    assert s["hit@5"]["value"] > 0.5  # sanity: the real retrieval stack finds gold articles
    md = (patched / "latest-dev.md").read_text(encoding="utf-8")
    assert "## Summary" in md and "pending" in md  # human-edit rate awaits manual review


async def test_test_split_runs_are_counted_and_logged(patched):
    await cli.cmd_run(ns("test", patched, final=True))
    await cli.cmd_run(ns("test", patched, final=True, ingest=False))
    log = (patched / "test_runs.jsonl").read_text().splitlines()
    assert len(log) == 2 and json.loads(log[1])["prompts_sha"]
    assert "run #2" in (patched / "latest-test.md").read_text(encoding="utf-8")


async def test_stale_kb_is_refused_instead_of_silently_evaluated(patched):
    from sqlalchemy import text

    import supportpilot.db as db

    await cli.cmd_run(ns("dev", patched))
    async with db.SessionLocal() as s:
        await s.execute(text("DELETE FROM kb_documents WHERE source_uri LIKE 'fa/%'"))
        await s.commit()
    with pytest.raises(SystemExit, match="--ingest"):
        await cli.cmd_run(ns("dev", patched, ingest=False))
