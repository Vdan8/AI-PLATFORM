from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    # ==================== Existing Settings ====================
    OPENAI_API_KEY: str
    DEFAULT_GPT_MODEL: str = "gpt-3.5-turbo-1106"
    LOG_FILE_PATH: str = "app/logs/trace_log.jsonl"

    # ==================== New FastAPI 2.0+ Settings ====================
    ENVIRONMENT: str = "dev"  # or "prod"
    CRITICAL_TOOLS: List[str] = []  # Add essential tool names here
    ALLOW_TOOL_RELOAD: bool = False  # For development

    # Pydantic v2 config (replaces old Config class)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Silently ignore extra env vars
        case_sensitive=False  # Optional: allow case-insensitive env vars
    )

    # New auth/database settings (all optional with defaults)
    DATABASE_URL: str = "postgresql://user:pass@localhost/dbname"  # Default for local dev
    SECRET_KEY: str = "temp-secret-change-me"  # Override via .env in prod
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # Add this line

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Silently ignore unused env vars
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Singleton pattern remains unchanged
settings = Settings()