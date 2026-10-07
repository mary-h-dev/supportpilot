"""Provider-agnostic LLM wrapper.

Anything OpenAI-compatible works (OpenRouter, OpenAI, Ollama, vLLM...): change LLM_BASE_URL,
LLM_API_KEY and LLM_MODEL. We deliberately do NOT rely on provider JSON-mode or tool calling
(free models support them inconsistently): we ask for JSON, parse, validate with Pydantic,
and retry once with the validation error.
"""

import json
import re
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings, settings
from .logging import log

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Transport/auth/rate-limit failure."""


class LLMOutputError(LLMError):
    """The model never produced valid JSON for the schema."""


class Usage(BaseModel):
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            model=self.model or other.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class LLM(Protocol):
    async def complete_json(
        self, *, step: str, model: str, system: str, user: str, schema: type[T]
    ) -> tuple[T, Usage]: ...


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> str:
    """Strip markdown fences / chatter and return the outermost {...} block."""
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class OpenAICompatLLM:
    def __init__(self, cfg: Settings = settings, client=None):
        self.cfg = cfg
        if client is None:
            if not cfg.llm_api_key:
                raise LLMError("LLM_API_KEY is empty. Set it in .env (never in code).")
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                base_url=cfg.llm_base_url, api_key=cfg.llm_api_key, timeout=60, max_retries=4
            )
        self.client = client

    def _cost(self, pt: int, ct: int) -> float:
        c = self.cfg
        return (pt * c.llm_price_in_per_m + ct * c.llm_price_out_per_m) / 1_000_000

    async def _call(self, step: str, model: str, messages: list[dict]) -> tuple[str, Usage]:
        t0 = time.perf_counter()
        try:
            resp = await self.client.chat.completions.create(
                model=model, messages=messages, temperature=0, max_tokens=900
            )
        except Exception as e:  # network, 401, 429 ...
            raise LLMError(f"{type(e).__name__}: {e}") from e
        ms = (time.perf_counter() - t0) * 1000
        u = getattr(resp, "usage", None)
        pt, ct = (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
        usage = Usage(
            model=model,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=round(ms, 1),
            cost_usd=self._cost(pt, ct),
        )
        log.info("llm_call", step=step, **usage.model_dump())
        if not resp.choices or not resp.choices[0].message.content:
            raise LLMError("empty completion")
        return resp.choices[0].message.content, usage

    async def complete_json(
        self, *, step: str, model: str, system: str, user: str, schema: type[T]
    ) -> tuple[T, Usage]:
        sys_full = (
            f"{system}\n\nReturn ONLY one JSON object matching this JSON Schema, "
            f"no prose, no markdown:\n{json.dumps(schema.model_json_schema())}"
        )
        messages = [{"role": "system", "content": sys_full}, {"role": "user", "content": user}]
        total = Usage(model=model)
        last_err = ""
        for attempt in range(2):  # one retry with the validation error
            raw, usage = await self._call(step, model, messages)
            total = total + usage
            try:
                return schema.model_validate_json(extract_json(raw)), total
            except (ValidationError, ValueError) as e:
                last_err = str(e)[:300]
                log.warning("llm_invalid_json", step=step, attempt=attempt, error=last_err)
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Invalid: {last_err}. Return ONLY valid JSON."},
                ]
        raise LLMOutputError(f"{step}: no valid JSON after retry: {last_err}")
