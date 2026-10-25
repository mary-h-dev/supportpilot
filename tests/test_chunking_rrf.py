import httpx
import pytest

from supportpilot.config import Settings
from supportpilot.kb.chunking import chunk_text
from supportpilot.kb.embeddings import OllamaEmbedder, get_embedder
from supportpilot.kb.retrieval import rrf_fuse


def test_chunks_respect_max_and_keep_all_text():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(6))
    chunks = chunk_text(body, max_chars=500)
    assert len(chunks) > 1 and all(len(c) <= 500 for c in chunks)
    assert "Paragraph 5" in chunks[-1]


def test_long_paragraph_is_split_on_sentences():
    para = ". ".join(f"Sentence number {i} about billing" for i in range(40)) + "."
    assert all(len(c) <= 300 for c in chunk_text(para, max_chars=300))


def test_rrf_rewards_agreement():
    scores = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] > scores["c"] and scores["b"] > scores["d"]


def test_rrf_single_list_keeps_order():
    scores = rrf_fuse([["x", "y"], []])
    assert scores["x"] > scores["y"]


async def test_ollama_embedder_batches_and_checks_dim():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        texts = json.loads(request.content)["input"]
        calls.append(len(texts))
        return httpx.Response(200, json={"embeddings": [[0.1] * 4 for _ in texts]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ok = OllamaEmbedder("http://x", "bge-m3", 4, client)
    assert len(await ok.embed(["t"] * 20)) == 20 and calls == [16, 4]
    with pytest.raises(ValueError, match="dim"):
        await OllamaEmbedder("http://x", "bge-m3", 1024, client).embed(["t"])


def test_embedder_provider_is_config_driven():
    assert isinstance(get_embedder(Settings(embedding_provider="ollama")), OllamaEmbedder)
    with pytest.raises(ValueError):
        get_embedder(Settings(embedding_provider="nope"))


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _no_sleep(_):
    return None


async def test_ollama_embedder_retries_transient_failures_then_succeeds():
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:  # first two calls: Ollama is (re)loading the model
            return httpx.Response(503, text="model loading")
        return httpx.Response(200, json={"embeddings": [[0.1] * 4]})

    e = OllamaEmbedder("http://x", "bge-m3", 4, _client(handler), retries=2, sleep=_no_sleep)
    assert len(await e.embed(["t"])) == 1 and len(attempts) == 3


async def test_ollama_embedder_gives_actionable_error_when_it_keeps_failing():
    from supportpilot.kb.embeddings import EmbeddingError

    def handler(request):
        raise httpx.ConnectError("connection refused")

    e = OllamaEmbedder(
        "http://host:11434", "bge-m3", 4, _client(handler), retries=1, sleep=_no_sleep
    )
    with pytest.raises(EmbeddingError, match=r"host:11434.*2 attempts.*Is Ollama running"):
        await e.embed(["t"])


async def test_ollama_embedder_does_not_retry_client_errors():
    from supportpilot.kb.embeddings import EmbeddingError

    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404, text="model 'bge-m3' not found")

    e = OllamaEmbedder("http://x", "bge-m3", 4, _client(handler), retries=3, sleep=_no_sleep)
    with pytest.raises(EmbeddingError, match="not found"):
        await e.embed(["t"])
    assert len(calls) == 1  # a wrong request is not fixed by repeating it
