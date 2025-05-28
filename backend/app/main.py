
from contextlib import asynccontextmanager
from typing import Dict, Any, List, AsyncGenerator
import logging

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from tenacity import retry, stop_after_attempt, wait_fixed

from app.api import agent # Assuming this is where agent-related routers are
from app.schemas.prompt import PromptRequest, PromptResponse # Ensure this file exists and contains these schemas
from backend.config.config import settings
from app.models.base import Base # Assuming your SQLAlchemy Base is here
from app.utils.logger import trace_logger_service
from backend.app.services.sandbox_executor import initialize_sandbox_service, shutdown_sandbox_service
from app.services.tool_registry import tool_registry_service
from app.services.agent_orchestrator import agent_orchestrator_service # Import the orchestrator service




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

# --- Dependency to get DB session ---
# This function is crucial for FastAPI's dependency injection
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_sessionmaker() as session:
        yield session

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
            all_loaded_tools = await tool_registry_service.get_llm_tools_for_agent(session)

            # Store tool definitions in app.state for quick access by other parts if needed
            # The orchestrator directly uses tool_registry_service, but this can be helpful.
            app.state.tools: Dict[str, Any] = {tool["function"]["name"]: tool for tool in all_loaded_tools}

            required_tools: List[str] = settings.CRITICAL_TOOLS # Assuming settings.CRITICAL_TOOLS is defined
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
# Include other routers if you have them, e.g., for agent management
app.include_router(agent.router, prefix=settings.API_V1_STR + "/agent", tags=["Agent Operations"])

# --- NEW: Endpoint for processing user prompts ---
@app.post("/api/v1/prompt", response_model=PromptResponse, status_code=status.HTTP_200_OK)
async def process_user_prompt_endpoint(
    request: PromptRequest,
    db_session: AsyncSession = Depends(get_db) # Inject database session
):
    """
    Processes a user prompt through the agent orchestrator, handling sequential chaining and tool generation.
    """
    logger.info(f"Received prompt: {request.user_prompt}")
    await trace_logger.log_event({"event_type": "user_prompt_received", "prompt": request.user_prompt})

    try:
        response_data = await agent_orchestrator_service.process_user_prompt(
            user_prompt=request.user_prompt,
            db_session=db_session
        )
        
        # Log the final response from the orchestrator
        logger.info(f"Orchestrator final response: {response_data.get('message')}")
        await trace_logger.log_event({
            "event_type": "orchestrator_response_sent",
            "status": response_data.get("status"),
            "message": response_data.get("message"),
            "tool_name": response_data.get("tool_name"),
            "tool_output": response_data.get("tool_output"),
            "thought_history_length": len(response_data.get("thought_history", []))
        })

        return PromptResponse(
            status=response_data.get("status", "success"), # Default to success if not explicitly set
            message=response_data.get("message", "Prompt processed successfully."),
            tool_output=response_data.get("tool_output"),
            tool_name=response_data.get("tool_name"),
            thought_process=response_data.get("thought_history") # Include thought_history for debugging
        )
    except HTTPException as e:
        logger.error(f"HTTPException processing prompt: {e.detail}", exc_info=True)
        await trace_logger.log_event({"event_type": "prompt_processing_error", "error": str(e), "status_code": e.status_code})
        raise e
    except Exception as e:
        # Catch unexpected errors and return a 500
        logger.critical(f"An unexpected error occurred during prompt processing: {e}", exc_info=True)
        await trace_logger.log_event({"event_type": "prompt_processing_critical_error", "error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing your request: {str(e)}"
        )

# --- Existing Endpoints ---
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