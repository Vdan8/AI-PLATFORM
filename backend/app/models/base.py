# app/models/base.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import AsyncAttrs
from app.core.config import settings
from typing import AsyncGenerator, Any  # Import AsyncGenerator

SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

async_engine = create_async_engine(SQLALCHEMY_DATABASE_URL, echo=False)

Base = declarative_base(cls=AsyncAttrs)

async def get_db() -> AsyncGenerator[AsyncSession, None]:  # Use AsyncGenerator
    async with AsyncSession(async_engine, expire_on_commit=False) as session:
        yield session
