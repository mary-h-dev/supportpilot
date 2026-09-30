import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from supportpilot.config import settings

config = context.config


def _run(connection):
    context.configure(connection=connection, target_metadata=None)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async():
    engine = create_async_engine(settings.database_url)
    async with engine.connect() as conn:
        await conn.run_sync(_run)
    await engine.dispose()


asyncio.run(_run_async())
