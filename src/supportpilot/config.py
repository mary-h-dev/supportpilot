from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration and secrets come from the environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://supportpilot:supportpilot@localhost:5432/supportpilot"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_judge_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    embedding_provider: str = "ollama"
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024
    ollama_url: str = "http://localhost:11434"


settings = Settings()
