# test_db.py
import asyncio
import asyncpg  # Directly import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import sys
print("Runtime Python executable:", sys.executable)

async def test_connection():
    try:
        engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/dbname", echo=True)  # Replace!
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            print(result.scalar_one())
        print("Asyncpg connection successful!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'engine' in locals():
            await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_connection())

