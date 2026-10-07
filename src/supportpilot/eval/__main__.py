"""python -m supportpilot.eval run --split dev [--ingest]
python -m supportpilot.eval run --split test --final      # held-out set: once, at the end
python -m supportpilot.eval report RESULTS.json [--reviews REVIEW.csv]
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from ..agent import prompts
from ..config import settings
from . import report
from .cache import CachingLLM
from .dataset import DEFAULT_PATH, check_against_kb, load_tickets
from .runner import EvalConfig, evaluate


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:10]


def _commit() -> str:
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown (no git in this environment)"


async def _kb_ready(ingest: bool) -> None:
    from sqlalchemy import text

    from ..db import SessionLocal

    if ingest:
        from ..kb.ingest import main as ingest_main

        await ingest_main()
    expected = len(list(Path("kb").glob("*/*.md")))
    async with SessionLocal() as s:
        have = (await s.execute(text("SELECT count(*) FROM kb_documents"))).scalar_one()
    if have != expected:
        sys.exit(
            f"KB in the database has {have} articles but kb/ has {expected}. Re-run with --ingest."
        )


async def cmd_run(a) -> None:
    if a.split == "test" and not a.final:
        sys.exit("The test split is held out. Run it once, at the very end, with --final.")
    tickets = load_tickets(DEFAULT_PATH, a.split)
    if problems := check_against_kb(tickets):
        sys.exit("Dataset/KB mismatch:\n" + "\n".join(problems))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    await _kb_ready(a.ingest)
    from ..agent.search import DbKBSearch
    from ..db import SessionLocal
    from ..kb.embeddings import get_embedder
    from ..llm import OpenAICompatLLM
    from ..logging import setup_logging

    setup_logging(level=logging.WARNING)
    llm = CachingLLM(OpenAICompatLLM(), Path("eval/.cache"), enabled=not a.no_cache)
    cfg = EvalConfig(
        model=settings.llm_model,
        judge_model=settings.llm_judge_model,
        k=settings.agent_retrieval_k,
        min_confidence=settings.agent_min_confidence,
        concurrency=a.concurrency,
    )

    runs_log = out / "test_runs.jsonl"
    test_run_number = None
    if a.split == "test":
        test_run_number = (len(runs_log.read_text().splitlines()) if runs_log.exists() else 0) + 1
        if test_run_number > 1:
            print(
                f"WARNING: this is test run #{test_run_number}. Results must not drive prompt changes."
            )

    print(
        f"Running {len(tickets)} {a.split} tickets with {cfg.model} (judge: {cfg.judge_model})..."
    )
    results = await evaluate(tickets, DbKBSearch(SessionLocal, get_embedder()), llm, cfg)

    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    meta = {
        "split": a.split,
        "timestamp_utc": ts,
        "git_commit": _commit(),
        "generator_model": cfg.model,
        "judge_model": cfg.judge_model,
        "embedding_model": settings.embedding_model,
        "retrieval_k": cfg.k,
        "min_confidence": cfg.min_confidence,
        "prompt_version": prompts.PROMPT_VERSION,
        "prompts_sha": _sha(Path(prompts.__file__)),
        "dataset_sha": _sha(DEFAULT_PATH),
        "llm_calls_from_cache": f"{llm.hits}/{llm.hits + llm.misses}",
    }
    if test_run_number:
        meta["test_run_number"] = test_run_number
        with runs_log.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": ts,
                        "commit": meta["git_commit"],
                        "prompts_sha": meta["prompts_sha"],
                        "model": cfg.model,
                    }
                )
                + "\n"
            )

    stem = f"{a.split}-{ts}"
    (out / f"{stem}.json").write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (out / f"review-{stem}.csv").write_text(
        report.reviews_to_csv(report.review_rows(results)), encoding="utf-8-sig"
    )
    md = report.render_report(results, meta)
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    (out / f"latest-{a.split}.md").write_text(md, encoding="utf-8")
    print("\n" + report.summary_table(report.summarize(results)))
    print(f"\nWrote {out}/{stem}.md (+ .json, review-{stem}.csv)")
    print("Next: open the review CSV, fill human_decision / human_faithful, then:")
    print(
        f"  python -m supportpilot.eval report {out}/{stem}.json --reviews {out}/review-{stem}.csv"
    )


def cmd_report(a) -> None:
    blob = json.loads(Path(a.results).read_text(encoding="utf-8"))
    reviews = (
        report.parse_reviews(Path(a.reviews).read_text(encoding="utf-8")) if a.reviews else None
    )
    md = report.render_report(blob["results"], blob["meta"], reviews)
    target = Path(a.results).with_suffix(".md")
    target.write_text(md, encoding="utf-8")
    (target.parent / f"latest-{blob['meta']['split']}.md").write_text(md, encoding="utf-8")
    print(f"Wrote {target}")


def main() -> None:
    p = argparse.ArgumentParser(prog="supportpilot.eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=["dev", "test"], default="dev")
    r.add_argument("--final", action="store_true", help="required for the held-out test split")
    r.add_argument("--ingest", action="store_true", help="(re)ingest kb/ first")
    r.add_argument("--concurrency", type=int, default=2)
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--out", default="eval/reports")
    g = sub.add_parser("report")
    g.add_argument("results")
    g.add_argument("--reviews")
    a = p.parse_args()
    asyncio.run(cmd_run(a)) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    main()
