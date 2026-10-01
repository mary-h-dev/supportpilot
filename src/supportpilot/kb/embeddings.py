"""Swappable embedding backends. Selected by EMBEDDING_PROVIDER in config."""

from typing import Protocol

import httpx

from ..config import Settings, settings


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(
        self, base_url: str, model: str, dim: int, client: httpx.AsyncClient | None = None
    ):
        self.base_url, self.model, self.dim = base_url.rstrip("/"), model, dim
        self._client = client or httpx.AsyncClient(timeout=120)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 16):
            r = await self._client.post(
                f"{self.base_url}/api/embed", json={"model": self.model, "input": texts[i : i + 16]}
            )
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
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
