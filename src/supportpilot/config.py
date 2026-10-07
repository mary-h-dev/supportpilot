from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration and secrets come from the environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://supportpilot:supportpilot@localhost:5432/supportpilot"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_judge_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    # Optional price per 1M tokens (USD) for the cost estimate. 0 for ':free' models.
    llm_price_in_per_m: float = 0.0
    llm_price_out_per_m: float = 0.0
    agent_min_confidence: float = 0.6
    agent_retrieval_k: int = 5
    # Shared API key for /tickets*. Empty = auth disabled (local dev only).
    api_key: str = ""
    embedding_provider: str = "ollama"
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024
    ollama_url: str = "http://localhost:11434"


settings = Settings()
