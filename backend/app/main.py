import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from tenacity import retry, stop_after_attempt, wait_fixed

from app.api import agent
from app.core.config import settings
from app.models.base import Base
from app.utils.logger import trace_logger_service
from app.services.sandbox_service import initialize_sandbox_service, shutdown_sandbox_service
from app.services.tool_registry import tool_registry_service

# --- Logger Setup ---
trace_logger = trace_logger_service

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=settings.LOG_LEVEL,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Global Engine + Sessionmaker ---
async_engine: AsyncEngine | None = None
async_sessionmaker: sessionmaker | None = None

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Application startup initiated.")
    await trace_logger.log_event({"event_type": "app_startup", "message": "Application is starting up."})

    global async_engine, async_sessionmaker

    try:
        # 1. Connect to database
        logger.info("Connecting to database...")
        async_engine = create_async_engine(settings.DATABASE_URL, echo=False)
        async_sessionmaker = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

        await _retry_db_connection(async_engine)
        logger.info("✅ Database connection successful.")

        # 2. Create tables
        logger.info("Ensuring database tables are created...")
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables ensured.")

        # 3. Initialize sandbox
        logger.info("Initializing sandbox service...")
        await initialize_sandbox_service()
        logger.info("✅ Sandbox service initialized.")

        # 4. Load tools
        logger.info("Loading and verifying tools from database...")
        async with async_sessionmaker() as session:
            all_loaded_tools = await tool_registry_service.get_all_tool_definitions_for_llm(session)

            app.state.tools: Dict[str, Any] = {tool["function"]["name"]: tool for tool in all_loaded_tools}

            required_tools: List[str] = settings.CRITICAL_TOOLS
            missing = [t for t in required_tools if t not in app.state.tools]
            if missing:
                error_msg = f"🛑 Missing critical tools: {', '.join(missing)}. Startup halted."
                logger.critical(error_msg)
                await trace_logger.log_event({"event_type": "app_startup_failure", "reason": error_msg})
                raise RuntimeError(error_msg)

            logger.info(f"✅ Loaded {len(app.state.tools)} tools and verified critical tools.")
            await trace_logger.log_event({
                "event_type": "app_startup_success",
                "message": "Application startup complete.",
                "tools_loaded_count": len(app.state.tools)
            })

            yield

    except Exception as e:
        error_details = f"🛑 Application startup failed: {e.__class__.__name__}: {e}"
        logger.critical(error_details, exc_info=True)
        await trace_logger.log_event({"event_type": "app_startup_failure", "reason": error_details})
        raise

    finally:
        logger.info("👋 Application shutdown initiated.")
        await trace_logger.log_event({"event_type": "app_shutdown", "message": "Shutting down."})

        # Shutdown sandbox
        logger.info("Shutting down sandbox service...")
        await shutdown_sandbox_service()
        logger.info("✅ Sandbox shut down.")

        # Dispose DB engine
        if async_engine:
            logger.info("Disposing database engine...")
            await async_engine.dispose()
            logger.info("✅ Database engine disposed.")

        logger.info("👋 Shutdown complete.")

# --- Retry Logic for DB ---
@retry(stop=stop_after_attempt(settings.DB_RETRY_ATTEMPTS), wait=wait_fixed(settings.DB_RETRY_DELAY), reraise=True)
async def _retry_db_connection(engine: AsyncEngine):
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))

# --- FastAPI App ---
app = FastAPI(
    title="AI Employee Generator",
    description="A production-ready platform for dynamic AI agent operations, including tool generation and sandboxed execution.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url=None,
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(agent.router, prefix=settings.API_V1_STR + "/agent", tags=["Agent Operations"])

# --- Endpoints ---
@app.get("/", summary="Health Check", response_model=Dict[str, Any])
async def root():
    return {
        "message": "AI Employee Generator Backend is live",
        "status": "operational",
        "tools_loaded": len(getattr(app.state, "tools", {}))
    }

@app.get("/tools", summary="List Loaded Tools (Dev Only)", response_model=Dict[str, Any])
async def list_tools():
    if settings.ENVIRONMENT != "development":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not available in production.")
    return {"tools": list(getattr(app.state, "tools", {}).keys())}
