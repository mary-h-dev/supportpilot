from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .api import tickets
from .db import get_session
from .logging import setup_logging

setup_logging()
app = FastAPI(title="SupportPilot", version="0.1.0")
app.include_router(tickets.router)


@app.get("/health")
async def health(session: AsyncSession = Depends(get_session)):
    await session.execute(text("SELECT 1"))
    return {"status": "ok"}
