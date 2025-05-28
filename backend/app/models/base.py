# backend/app/models/base.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base # No AsyncAttrs import needed here
from sqlalchemy.ext.asyncio import AsyncAttrs # This is important for your models, not the Base class definition itself
from backend.config.config import settings
from typing import AsyncGenerator # Import AsyncGenerator
from contextlib import asynccontextmanager

SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

async_engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    future=True,
) # Use the 2.0 style

# CORRECTED: Define Base simply using declarative_base(cls=AsyncAttrs)
# AsyncAttrs will be automatically applied to classes that inherit from Base
Base = declarative_base(cls=AsyncAttrs)

@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async dependency to get a database session.
    """
    async with AsyncSession(async_engine, expire_on_commit=False) as session:
        try:
            yield session
        finally:
            await session.close()