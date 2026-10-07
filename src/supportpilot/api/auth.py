import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from ..config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)  # shows "Authorize" in /docs


async def require_api_key(key: str | None = Security(api_key_header)) -> None:
    """Shared-secret auth. Empty API_KEY disables it (local dev). NOT per-user identity."""
    expected = settings.api_key
    if not expected:
        return
    if not key or not secrets.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
