"""Swappable embedding backends. Selected by EMBEDDING_PROVIDER in config."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol

import httpx

from ..config import Settings, settings


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingError(Exception):
    """The embedding backend could not be reached or kept failing."""


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        client: httpx.AsyncClient | None = None,
        retries: int = 2,
        backoff: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.base_url, self.model, self.dim = base_url.rstrip("/"), model, dim
        self._client = client or httpx.AsyncClient(timeout=120)
        self.retries, self.backoff, self.sleep = retries, backoff, sleep

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Transient failures (connection reset, model reloading -> HTTP 5xx) are retried;
        4xx means our request is wrong, so retrying would be pointless."""
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = await self._client.post(
                    f"{self.base_url}/api/embed", json={"model": self.model, "input": batch}
                )
                r.raise_for_status()
                return r.json()["embeddings"]
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                    raise EmbeddingError(
                        f"Ollama rejected the request ({e.response.status_code}) for model "
                        f"'{self.model}': {e.response.text[:200]}"
                    ) from e
                last = e
                if attempt < self.retries:
                    await self.sleep(self.backoff * 3**attempt)
        raise EmbeddingError(
            f"Ollama at {self.base_url} failed after {self.retries + 1} attempts "
            f"({type(last).__name__}: {last}). Is Ollama running and is '{self.model}' pulled?"
        ) from last

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 16):
            out.extend(await self._embed_batch(texts[i : i + 16]))
        if out and len(out[0]) != self.dim:
            raise ValueError(
                f"Embedding dim {len(out[0])} != EMBEDDING_DIM {self.dim}. "
                "Changing model needs a new migration for vector(N)."
            )
        return out


def get_embedder(cfg: Settings = settings) -> Embedder:
    if cfg.embedding_provider == "ollama":
        return OllamaEmbedder(cfg.ollama_url, cfg.embedding_model, cfg.embedding_dim)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {cfg.embedding_provider}")


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"
