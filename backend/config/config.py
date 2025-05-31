import os
from typing import List, Literal, Optional
from pydantic import Field, HttpUrl # HttpUrl if you had any URL fields that need validation
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import logging
from dotenv import load_dotenv


# --- Determine Project Root and .env Path ---
# This setup assumes your .env file is in the project's root directory,
# which is two levels up from this config.py file (backend/app/core/).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOTENV_PATH = os.path.join(PROJECT_ROOT, '.env')

load_dotenv(dotenv_path=DOTENV_PATH)



# Get a module-specific logger for config
logger = logging.getLogger(__name__)

# --- Settings Class Definition ---
class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables or a .env file.
    Environment variables will override values in the .env file.
    Settings are prefixed with 'MCP_' (e.g., MCP_DATABASE_URL).
    """

    # ==================== Pydantic Settings Configuration ====================
    # This configuration tells Pydantic how to load the settings
    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,           # Specify the path to the .env file
        env_file_encoding="utf-8",      # Encoding for the .env file
        extra="ignore",                 # Silently ignore environment variables not defined in Settings
        case_sensitive=False,           # Treat environment variable names as case-insensitive (e.g., OPENAI_API_KEY vs openai_api_key)
        env_prefix='MCP_',              # Prefix for environment variables (e.g., MCP_DATABASE_URL, MCP_OPENAI_API_KEY)
                                        # Without this, it would look for DATABASE_URL.
        # env_nested_delimiter="__"     # Optional: for nested settings if you use them (e.g., DB__HOST)
    )

    # ==================== Core Application Settings ====================
    ENVIRONMENT: Literal["development", "testing", "production"] = Field(
        "development",
        description="Application environment: 'development', 'testing', or 'production'."
    )

    # OpenAI API Key - REQUIRED
    OPENAI_API_KEY: str = Field(..., description="Your OpenAI API key", validation_alias="MCP_OPENAI_API_KEY")


    # NEW: OpenAI Max Retries for Tenacity
    OPENAI_MAX_RETRIES: int = Field(
        3, # Default to 3 retries (total 4 attempts including the first)
        description="Maximum number of retries for OpenAI API calls using tenacity."
    )

    LLM_MODEL: str = Field(
        "gpt-4o",
        description="Default LLM model to use (e.g., 'gpt-4o', 'gpt-3.5-turbo-1106')."
    )
    ORCHESTRATOR_LLM_RESPONSE_MODEL: str = Field(
        "gpt-4o", # Explicitly set this as it was used in agent_orchestrator.py
        description="Specific LLM model used by the Agent Orchestrator for decision making."
    )
    ORCHESTRATOR_MAX_ITERATIONS: int = 5


    # Logging Level
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO",
        description="Minimum logging level for the application."
    )
    LOG_FILE_PATH: str = Field(
        os.path.join(PROJECT_ROOT, "app", "logs", "trace_log.jsonl"), # Robust path
        description="Path to the JSONL trace log file. Defaults to within the project structure."
    )

    # Tool Management
    CRITICAL_TOOLS: List[str] = Field(
        default_factory=list,
        description="Comma-separated list of critical tool names required for application startup. "
                    "Pydantic will automatically parse a comma-separated string from an env var into a list."
    )
    ALLOW_TOOL_RELOAD: bool = Field(
        False,
        description="Allow dynamic tool reloading (useful for development, disable in production)."
    )

    # Database Configuration
    DATABASE_URL: str = Field(
        "postgresql+asyncpg://user:pass@localhost/dbname",
        description="Main asynchronous database URL (e.g., for SQLAlchemy). "
                    "**CRITICAL: Override in production .env for your actual database.**"
    )
    ALEMBIC_DATABASE_URL: Optional[str] = Field(
        None,
        description="Optional: Synchronous database URL for Alembic migrations. "
                    "If not set, DATABASE_URL will be adapted by removing '+asyncpg'."
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
        "temp-secret-change-me-to-a-strong-random-string", # Make this longer and more unique
        description="Secret key for JWT encoding. **CRITICAL: Generate a strong, random key for production .env**"
    )
    ALGORITHM: str = Field("HS256", description="Algorithm used for JWT token signing (e.g., 'HS256').")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        30,
        description="Access token expiration time in minutes."
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        7,
        description="Refresh token expiration time in days."
    )

    # API Versioning
    API_V1_STR: str = Field("/api/v1", description="Base path for API v1 endpoints.")

    # ==================== Sandbox Settings (Optimized) ====================
    SANDBOX_BASE_DIR: str = Field(
        os.path.join(PROJECT_ROOT, "sandbox_data"), # Default to a directory within project root for better organization
        description="Base directory on the host for sandbox execution environments. "
                    "This directory will be created if it doesn't exist."
    )
    SANDBOX_IMAGE: str = Field(
        "python:3.11-slim-bookworm", # IMPORTANT: Customize this to your actual Docker image name!
        description="Docker image to use for sandbox containers. "
                    "Ensure this image is built and available on your Docker host."
    )
    SANDBOX_MEMORY_LIMIT_MB: int = Field(
        256, # Changed from string "128m" to int 256 for easier numeric handling in code
        description="Memory limit for sandbox containers in megabytes (e.g., 256, 512). "
                    "This value will be converted to Docker's 'm' format (e.g., '256m') internally by SandboxService."
    )
    SANDBOX_CPU_SHARES: Optional[int] = Field(
        512, # A common default is 1024. 512 gives half the CPU time slice priority.
        description="CPU shares for sandbox containers (relative weight compared to other containers). "
                    "This is an older Docker setting; for more precise CPU limiting, consider 'cpu_quota' and 'cpu_period' if your Docker version supports it and you have specific needs."
    )
    SANDBOX_MAX_CONTAINER_RUNTIME_SECONDS: int = Field(
        60, # Increased default timeout for potentially longer-running tools
        description="Maximum execution time for a tool in the sandbox in seconds. "
                    "Containers running longer than this will be terminated."
    )

    # --- Properties (Non-Field attributes, computed from other settings) ---
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
@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings class.
    This ensures settings are loaded only once during application lifecycle.
    """
    return Settings()

settings = get_settings()

# --- Optional: Log loaded settings (useful for development/debugging) ---
if settings.ENVIRONMENT == "development":
    logger.info("Application settings loaded:")
    for field_name, value in settings.model_dump().items():
        # Redact sensitive information for logging
        if "KEY" in field_name.upper() or "SECRET" in field_name.upper() or "TOKEN" in field_name.upper() or "DATABASE_URL" in field_name.upper():
            logger.info(f"  {field_name}: ***REDACTED***")
        else:
            logger.info(f"  {field_name}: {value}")