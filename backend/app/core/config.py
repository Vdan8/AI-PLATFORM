# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # Existing settings (won't break your code)
    OPENAI_API_KEY: str
    DEFAULT_GPT_MODEL: str = "gpt-3.5-turbo-1106"
    LOG_FILE_PATH: str = "app/logs/trace_log.jsonl"
    
    # New auth/database settings (all optional with defaults)
    DATABASE_URL: str = "postgresql://user:pass@localhost/dbname"  # Default for local dev
    SECRET_KEY: str = "temp-secret-change-me"  # Override via .env in prod
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Silently ignore unused env vars
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()