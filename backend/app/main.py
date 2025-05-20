import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List # Added List for type hints

from fastapi import FastAPI, HTTPException, status # Added status for cleaner HTTP error codes
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession # Added AsyncSession
from tenacity import retry, stop_after_attempt, wait_fixed, before_log # Import retry decorators

from app.api import agent # Assuming this is your primary agent router
# Remove auth, routes, job, tools imports unless you have corresponding files and they are fully functional
# from app.api import auth, routes, job, tools

from app.core.config import settings
from app.models.base import Base # For metadata.create_all
from app.models.base import get_db # To get a DB session for startup tasks
from app.utils.logger import trace_logger_service # Corrected logger import
from app.services.sandbox_service import initialize_sandbox, shutdown_sandbox
from app.services.tool_registry import tool_registry_service # Use the service directly

# --- Logger Setup ---
trace_logger = trace_logger_service() # Get the global trace logger instance

# Configure standard Python logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO if settings.ENVIRONMENT == "prod" else logging.DEBUG,
    # Direct logging to console for Uvicorn logs, trace_logger handles file logging
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__) # Get a standard logger for main.py specific logs

# --- Global AsyncEngine (FastAPI app.state is often preferred for managing shared resources) ---
# It's often better to store global state like the engine on app.state
# However, for engine, a global variable is also common. Let's keep it global for now
# but ensure it's properly managed in lifespan.
async_engine: AsyncEngine | None = None

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Initializes critical services and database connections.
    """
    logger.info("🚀 Application startup initiated.")
    # Log startup event to trace
    await trace_logger.log_event(event_type="app_startup", payload={"message": "Application is starting up."})

    global async_engine 

    try:
        # 1. Initialize Database Engine and Perform Warmup Connection
        logger.info("Connecting to database...")
        async_engine = create_async_engine(settings.DATABASE_URL, echo=False)
        # Use a retry mechanism for DB connection for robustness
        await _retry_db_connection(async_engine)
        logger.info("✅ Database connection successful.")

        # 2. Create Database Tables (if not exist)
        logger.info("Ensuring database tables are created...")
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables ensured.")

        # 3. Initialize Sandbox Service
        logger.info("Initializing sandbox service...")
        await initialize_sandbox()
        logger.info("✅ Sandbox service initialized.")

        # 4. Load and Verify Tools from Database
        logger.info("Loading and verifying tools from database...")
        # Get a DB session for lifespan context
        async with get_db() as session:
            tool_registry_service = tool_registry_service(db=session)
            all_loaded_tools = await tool_registry_service.get_all_tool_definitions() # Use the service method

            # Store tool definitions in app.state for easy access by other parts of the app if needed
            # This makes app.state.tools a dict mapping tool_name to its MCPToolDefinition object
            app.state.tools: Dict[str, Any] = {tool.name: tool for tool in all_loaded_tools}

            required_tools: List[str] = getattr(settings, "CRITICAL_TOOLS", []) # Use type hint
            missing_critical_tools = [t_name for t_name in required_tools if t_name not in app.state.tools]

            if missing_critical_tools:
                error_msg = f"🛑 Missing critical tools: {', '.join(missing_critical_tools)}. Application will not start."
                logger.critical(error_msg)
                await trace_logger.log_event(event_type="app_startup_failure", payload={"reason": error_msg})
                raise RuntimeError(error_msg) # Fail fast if critical tools are missing

            logger.info(f"✅ Loaded {len(app.state.tools)} tools from database and verified critical tools.")
            await trace_logger.log_event(
                event_type="app_startup_success",
                payload={"message": "Application startup complete.", "tools_loaded_count": len(app.state.tools)}
            )
            logger.info("✅ Application startup complete.")

            yield  # Application runs here

    except Exception as e:
        error_details = f"🛑 Application startup failed: {e.__class__.__name__}: {e}"
        logger.critical(error_details, exc_info=True) # Log full traceback
        await trace_logger.log_event(event_type="app_startup_failure", payload={"reason": error_details})
        raise # Re-raise the exception to indicate startup failure to Uvicorn/FastAPI

    finally:
        # --- Shutdown Logic ---
        logger.info("👋 Application shutdown initiated.")
        await trace_logger.log_event(event_type="app_shutdown", payload={"message": "Application is shutting down."})

        # 1. Shutdown Sandbox
        logger.info("Shutting down sandbox service...")
        await shutdown_sandbox()
        logger.info("✅ Sandbox service shut down.")

        # 2. Dispose Database Engine
        if async_engine:
            logger.info("Disposing database engine...")
            await async_engine.dispose()
            logger.info("✅ Database engine disposed.")

        logger.info("👋 Application shutdown complete.")


# --- Database Connection Retry Logic ---
@retry(stop=stop_after_attempt(settings.DB_RETRY_ATTEMPTS),
       wait=wait_fixed(settings.DB_RETRY_DELAY),
       before_log=before_log(logger, logging.WARNING),
       reraise=True)
async def _retry_db_connection(engine: AsyncEngine):
    """Retries database connection and table creation for robustness."""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Employee Generator",
    description="A production-ready platform for dynamic AI agent operations, including tool generation and sandboxed execution.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT == "dev" else None, # Only show docs in dev
    redoc_url=None, # Usually not needed in production
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be more restrictive in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Routers ---
# Dynamically include routers based on your actual project structure.
# For now, only 'agent' is confirmed and critical.
# Ensure 'app.api.agent' exists and defines 'agent.router'.
app.include_router(agent.router, prefix="/agent", tags=["Agent Operations"])

# --- Health Check Endpoint ---
@app.get("/", summary="Health Check", response_model=Dict[str, Any])
async def root():
    """
    Returns the status of the backend and the number of tools loaded.
    """
    return {
        "message": "AI Employee Generator Backend is live",
        "status": "operational",
        "tools_loaded": len(getattr(app.state, "tools", {})) # Safely get tools count
    }

# --- Debug Endpoint (Dev Only) ---
@app.get("/tools", summary="List Loaded Tools (Dev Only)", response_model=Dict[str, Any])
async def list_tools():
    """
    Lists the names of all tools loaded from the database (for development environments only).
    """
    if settings.ENVIRONMENT != "dev":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not available in production.")
    return {"tools": list(getattr(app.state, "tools", {}).keys())}