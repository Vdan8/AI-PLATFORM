from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.auth import router as auth_router
from app.services.tool_loader import tool_loader
from app.core.config import settings
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO if settings.ENVIRONMENT == "prod" else logging.DEBUG
)
logger = logging.getLogger(__name__)

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
        await warmup_connections(tools)
        
        # Store in app state
        app.state.tools = tools
        
        yield  # ============ Application Runs Here ============
        
    except Exception as e:
        logger.critical(f"🛑 Startup failed: {str(e)}")
        raise
    finally:
        # ================== Shutdown Logic ==================
        logger.info("🛑 Shutting down...")
        # Add cleanup logic here if needed

async def warmup_connections(tools: Dict[str, Any]):
    """Initialize connections for stateful tools"""
    if "database" in tools:
        try:
            await tools["database"]["function"]({"action": "ping"})
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

# Middleware (unchanged from your original)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)

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