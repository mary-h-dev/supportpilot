from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from supportpilot.config import Settings
from supportpilot.llm import LLMError, LLMOutputError, OpenAICompatLLM, extract_json


class Out(BaseModel):
    x: int


def fake_client(*contents, usage=(100, 50)):
    calls = []
    seq = list(contents)

    async def create(**kw):
        calls.append(kw)
        msg = SimpleNamespace(content=seq.pop(0))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg)],
            usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls


def test_extract_json_strips_fences_and_chatter():
    assert extract_json('```json\n{"x": 1}\n```') == '{"x": 1}'
    assert extract_json('Sure! Here: {"x": 1} hope it helps') == '{"x": 1}'


async def test_valid_json_and_cost_estimate():
    cfg = Settings(llm_api_key="k", llm_price_in_per_m=1.0, llm_price_out_per_m=2.0)
    client, _ = fake_client('{"x": 7}')
    out, usage = await OpenAICompatLLM(cfg, client).complete_json(
        step="t", model="m", system="s", user="u", schema=Out
    )
    assert out.x == 7 and usage.prompt_tokens == 100
    assert usage.cost_usd == pytest.approx((100 * 1.0 + 50 * 2.0) / 1e6)


async def test_retries_once_with_validation_error_and_sums_usage():
    client, calls = fake_client("not json", '{"x": 2}')
    out, usage = await OpenAICompatLLM(Settings(llm_api_key="k"), client).complete_json(
        step="t", model="m", system="s", user="u", schema=Out
    )
    assert out.x == 2 and len(calls) == 2 and usage.prompt_tokens == 200
    assert "Invalid" in calls[1]["messages"][-1]["content"]


async def test_gives_up_after_one_retry():
    client, _ = fake_client("nope", "still nope")
    with pytest.raises(LLMOutputError):
        await OpenAICompatLLM(Settings(llm_api_key="k"), client).complete_json(
            step="t", model="m", system="s", user="u", schema=Out
        )


def test_missing_api_key_fails_loudly():
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        OpenAICompatLLM(Settings(llm_api_key=""))
