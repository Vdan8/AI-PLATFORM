from dotenv import load_dotenv
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(dotenv_path=dotenv_path)

class Settings(BaseSettings):
    # ==================== Existing Settings ====================
    OPENAI_API_KEY: str
    DEFAULT_GPT_MODEL: str = "gpt-3.5-turbo-1106"
    LOG_FILE_PATH: str = "app/logs/trace_log.jsonl"

    # ==================== New FastAPI 2.0+ Settings ====================
    ENVIRONMENT: str = "dev"  # or "prod"
    CRITICAL_TOOLS: List[str] = []  # Add essential tool names here
    ALLOW_TOOL_RELOAD: bool = False  # For development

    # ==================== Sandbox Settings ====================
    SANDBOX_BASE_DIR: str = "/tmp/tool_sandboxes"  # Base directory for sandbox folders
    SANDBOX_IMAGE: str = "python:3.9-slim-buster"  # Docker image for tool execution
    SANDBOX_MEM_LIMIT: str = "128m"  # Memory limit for containers (e.g., "128m", "512m")
    SANDBOX_CPU_SHARES: int = 256  # CPU shares for containers (relative weight)

    # New auth/database settings (all optional with defaults)
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost/dbname" # Default for local dev
    ALEMBIC_DATABASE_URL: str | None = None  # Optional override for migrations
    SECRET_KEY: str = "temp-secret-change-me"  # Override via .env in prod
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # Add this line

    @property
    def alembic_url(self) -> str:
        """
        Returns a psycopg2-compatible sync database URL for Alembic migrations.
        Falls back to DATABASE_URL with '+asyncpg' stripped.
        """
        if self.ALEMBIC_DATABASE_URL:
            return self.ALEMBIC_DATABASE_URL
        return self.DATABASE_URL.replace("+asyncpg", "")


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Silently ignore extra env vars
        case_sensitive=False  # Optional: allow case-insensitive env vars
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Singleton pattern remains unchanged
settings = Settings()