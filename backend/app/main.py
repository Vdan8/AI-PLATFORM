# backend/app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# CORRECTED IMPORT: Now importing 'tools' (plural) as confirmed
from .api import auth, agent, routes, job, tools # Import 'tools' (plural)

from app.services.tool_loader import tool_loader_service
from .core.config import settings
import logging
from typing import Dict, Any
from app.services.sandbox_service import initialize_sandbox, shutdown_sandbox
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import json # Added to handle potential JSON parsing of tool output in call_tool

# NEW: Import the tool_registry functions to potentially use them in lifespan if needed
from app.services.tool_registry import get_tool_definitions, get_single_tool_definition_for_llm


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO if settings.ENVIRONMENT == "prod" else logging.DEBUG
)
logger = logging.getLogger(__name__)

# Create a global AsyncEngine instance
async_engine: AsyncEngine | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI 2.0+ lifespan manager"""
    # ==================== Startup Logic ====================
    logger.info("🚀 Starting application...")

    try:
        # 1. Warm connections (DB, APIs)
        await warmup_connections()

        # 2. Initialize the sandbox service (Docker client etc.)
        await initialize_sandbox()

        # 3. Optional: Perform an initial fetch or validation of tools from DB
        #    This replaces the old tool_loader.load_tools()
        #    You might want to fetch all tools and store them in app.state for quick access
        #    or just verify that critical tools exist in the DB.

        # Example: Fetch all tool definitions for validation/initial priming
        # You'll need to use the tool_loader_service (via tool_registry's get_tool_definitions)
        # Note: get_tool_definitions now directly pulls from the DB.
        all_loaded_tools = await get_tool_definitions()

        # Store a simplified dictionary of tool names for app.state.tools for compatibility
        # (This just gets the names, not the full tool objects/metadata from the old system)
        # If your frontend/other parts of your app depend on app.state.tools having the full dict
        # then you'd need to adapt this more deeply. For now, it's a basic list of names.
        app.state.tools = {tool['function']['name']: tool for tool in all_loaded_tools}


        # 4. Verify critical tools (now checks against DB-loaded tools)
        required_tools = getattr(settings, "CRITICAL_TOOLS", [])
        missing = [t_name for t_name in required_tools if t_name not in app.state.tools]
        if missing:
            raise RuntimeError(f"Missing critical tools: {missing}")

        logger.info(f"✅ Loaded {len(app.state.tools)} tools from database and verified critical tools.")
        logger.info("✅ Application startup complete.")

        yield  # ============ Application Runs Here ============

    except Exception as e:
        logger.critical(f"🛑 Startup failed: {str(e)}")
        raise # Re-raise to make sure Uvicorn reports failure

    finally:
        # ================== Shutdown Logic ==================
        logger.info("🛑 Shutting down...")

        # 5. Shutdown the sandbox
        await shutdown_sandbox()

        # Clean up resources
        global async_engine
        if async_engine:
            await async_engine.dispose()

        logger.info("👋 Application shutdown")

async def warmup_connections():
    """Initialize connections for stateful tools"""
    global async_engine
    try:
        # Warm up database connection
        async_engine = create_async_engine(settings.DATABASE_URL, echo=False)
        async with async_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.debug("🏓 Database ping successful")
    except Exception as e:
        logger.warning(f"Database warmup failed: {str(e)}")

app = FastAPI(
    title="AI Employee Generator",
    description="Production-ready AI agent platform",
    version="1.0.0",
    lifespan=lifespan,  # FastAPI 2.0+ lifespan manager
    docs_url="/docs" if settings.ENVIRONMENT == "dev" else None,
    redoc_url=None
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(routes.router)
app.include_router(agent.router)
app.include_router(job.router)
# CORRECTED ROUTER INCLUDE: Using 'tools.router' as confirmed by your filename
app.include_router(tools.router, prefix=settings.API_V1_STR + "/tools", tags=["tools"])

# ==================== Endpoints ====================
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Employee Generator Backend is live",
        "status": "operational",
        "tools_loaded": len(app.state.tools) # This will now be the count of tools from DB
    }

@app.get("/tools")
async def list_tools():
    """Debug endpoint (dev only)"""
    if settings.ENVIRONMENT != "dev":
        raise HTTPException(status_code=403, detail="Not available in production")
    return {"tools": list(app.state.tools.keys())}