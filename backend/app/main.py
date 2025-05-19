from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .api import auth, agent, routes, job, tools  # Import all routers
from .services import tool_loader
from .core.config import settings
import logging
from typing import Dict, Any
from app.services.sandbox_service import initialize_sandbox, shutdown_sandbox
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

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
        # 1. Load tools
        tools: Dict[str, Any] = tool_loader.load_tools()
        logger.info(f"✅ Loaded {len(tools)} tools")

        # 2. Verify critical tools
        required_tools = getattr(settings, "CRITICAL_TOOLS", [])
        missing = [t for t in required_tools if t not in tools]
        if missing:
            raise RuntimeError(f"Missing critical tools: {missing}")

        # 3. Warm connections (DB, APIs)
        await warmup_connections()

        # 4. Initialize the sandbox
        await initialize_sandbox()

        # Store in app state
        app.state.tools = tools

        yield  # ============ Application Runs Here ============

    except Exception as e:
        logger.critical(f"🛑 Startup failed: {str(e)}")
        raise

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
app.include_router(tools.router)

# ==================== Endpoints ====================
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Employee Generator Backend is live",
        "status": "operational",
        "tools_loaded": len(app.state.tools)
    }

@app.get("/tools")
async def list_tools():
    """Debug endpoint (dev only)"""
    if settings.ENVIRONMENT != "dev":
        raise HTTPException(status_code=403, detail="Not available in production")
    return {"tools": list(app.state.tools.keys())}