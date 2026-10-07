from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import Depends, FastAPI, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .api import tickets
from .db import get_session
from .logging import setup_logging
from .runtime import build_runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    async with AsyncExitStack() as stack:
        app.state.runtime = await build_runtime(stack)
        yield


app = FastAPI(title="SupportPilot", version="0.5.0", lifespan=lifespan)
app.include_router(tickets.router)


@app.get("/health")
async def health(request: Request, session: AsyncSession = Depends(get_session)):
    await session.execute(text("SELECT 1"))
    agent = "ready" if getattr(request.app.state, "runtime", None) else "unavailable"
    return {"status": "ok", "agent": agent}
