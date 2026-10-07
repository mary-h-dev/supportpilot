"""Disk cache for LLM calls: re-running eval with unchanged prompts/models is free and exactly
repeatable. Any prompt/model/ticket change changes the key, so stale answers are never reused."""

import hashlib
import json
from pathlib import Path

from ..llm import Usage


class CachingLLM:
    def __init__(self, inner, directory: Path = Path("eval/.cache"), enabled: bool = True):
        self.inner, self.dir, self.enabled = inner, Path(directory), enabled
        self.hits = self.misses = 0
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    async def complete_json(self, *, step, model, system, user, schema):
        if not self.enabled:
            self.misses += 1
            return await self.inner.complete_json(
                step=step, model=model, system=system, user=user, schema=schema
            )
        key = hashlib.sha256(
            json.dumps([model, system, user, schema.__name__], ensure_ascii=False).encode()
        ).hexdigest()
        path = self.dir / f"{key}.json"
        if path.exists():
            blob = json.loads(path.read_text(encoding="utf-8"))
            self.hits += 1
            return schema(**blob["data"]), Usage(**blob["usage"])
        self.misses += 1
        out, usage = await self.inner.complete_json(
            step=step, model=model, system=system, user=user, schema=schema
        )
        path.write_text(
            json.dumps({"data": out.model_dump(), "usage": usage.model_dump()}, ensure_ascii=False),
            encoding="utf-8",
        )
        return out, usage
