from dotenv import load_dotenv
import os
from pydantic import Field # <--- NEW: Explicitly import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional # Added Optional for clarity

# --- Load Environment Variables ---
# Determine the path to the .env file.
# This assumes .env is in the project root, one level above 'backend'
# and two levels above 'backend/app'.
# (backend/app/core/config.py -> backend/app/core -> backend/app -> backend -> .env)
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Settings Class Definition ---
class Settings(BaseSettings):
    # ==================== Core Application Settings ====================
    ENVIRONMENT: str = Field("dev", description="Application environment: 'dev', 'test', or 'prod'")

    # OpenAI API Key - REQUIRED
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key (required)")
    LLM_MODEL: str = Field("gpt-4o", description="Default LLM model to use (e.g., 'gpt-4o', 'gpt-3.5-turbo-1106')")

    # Logging & Tracing
    LOG_FILE_PATH: str = Field("app/logs/trace_log.jsonl", description="Path to the JSONL trace log file")

    # Tool Management
    CRITICAL_TOOLS: List[str] = Field(
        default_factory=list,
        description="Comma-separated list of critical tool names required for application startup.",
        # Pydantic automatically parses comma-separated strings from env into a list
    )
    ALLOW_TOOL_RELOAD: bool = Field(
        False,
        description="Allow dynamic tool reloading (useful for development, disable in production)."
    )

    # Database Configuration
    DATABASE_URL: str = Field(
        "postgresql+asyncpg://user:pass@localhost/dbname",
        description="Main asynchronous database URL (e.g., for SQLAlchemy)."
                    " **CRITICAL: Override in production .env**"
    )
    ALEMBIC_DATABASE_URL: Optional[str] = Field(
        None,
        description="Optional: Synchronous database URL for Alembic migrations. "
                    "If not set, DATABASE_URL will be adapted."
    )
    DB_RETRY_ATTEMPTS: int = Field(
        5,
        description="Number of attempts to retry database connection on startup."
    )
    DB_RETRY_DELAY: int = Field(
        3,
        description="Delay in seconds between database connection retries."
    )

    # Authentication & Security
    SECRET_KEY: str = Field(
        "temp-secret-change-me",
        description="Secret key for JWT encoding. **CRITICAL: Generate a strong, random key for production .env**"
    )
    ALGORITHM: str = Field("HS256", description="Algorithm used for JWT token signing.")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, description="Access token expiration time in minutes.")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(7, description="Refresh token expiration time in days.")

    # API Versioning
    API_V1_STR: str = Field("/api/v1", description="Base path for API v1 endpoints.")

    # ==================== Sandbox Settings ====================
    SANDBOX_BASE_DIR: str = Field(
        "/tmp/tool_sandboxes",
        description="Base directory on the host for sandbox execution environments."
    )
    SANDBOX_IMAGE: str = Field(
        "python:3.9-slim-buster",
        description="Docker image to use for sandbox containers."
    )
    SANDBOX_MEM_LIMIT: str = Field(
        "128m",
        description="Memory limit for sandbox containers (e.g., '128m', '512m')."
    )
    SANDBOX_CPU_SHARES: int = Field(
        256,
        description="CPU shares for sandbox containers (relative weight)."
    )
    SANDBOX_TIMEOUT_SECONDS: int = Field(
        30,
        description="Maximum execution time for a tool in the sandbox in seconds."
    )

    # --- Pydantic Settings Configuration ---
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Silently ignore environment variables not defined in Settings
        case_sensitive=False, # Treat environment variable names as case-insensitive
        # env_nested_delimiter="__" # Optional: for nested settings if you use them
    )

    # --- Properties (Non-Field attributes) ---
    @property
    def alembic_url(self) -> str:
        """
        Returns a psycopg2-compatible synchronous database URL for Alembic migrations.
        Falls back to DATABASE_URL with '+asyncpg' stripped.
        """
        if self.ALEMBIC_DATABASE_URL:
            return self.ALEMBIC_DATABASE_URL
        # Remove the '+asyncpg' part to get a sync URL for alembic
        return self.DATABASE_URL.replace("+asyncpg", "")

# --- Singleton Settings Instance ---
# For simplicity and direct access (settings.OPENAI_API_KEY), we'll instantiate it directly.
# If you prefer dependency injection for settings in FastAPI, you would keep get_settings()
# with @lru_cache and remove this global 'settings' instance.
settings = Settings()