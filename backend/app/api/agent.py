import logging # Import standard logging module
from typing import Optional # For Optional type hint

from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user import User
from app.models.agent import AgentConfiguration
from app.schemas.agent import AgentCreate, AgentRead
from app.exceptions import AgentNameAlreadyExistsError
from app.core.auth import get_current_user, has_role # Assuming these are functional
from app.models.base import get_db # Correct dependency for database session

# NEW IMPORTS for Sandbox Testing - Keep these IF you need the endpoint for restricted testing
from app.schemas.tool import MCPToolCall, MCPToolResponse
from app.services.sandbox_service import sandbox_service
from app.core.config import settings # Needed to check environment for dev-only endpoints


# --- Logger Setup ---
# Use standard Python logging for API-level logs (e.g., errors in this module)
logger = logging.getLogger(__name__)
# If this file ever directly logs agent *trace* events, you'd import get_trace_logger_instance
# from app.logger import get_trace_logger_instance
# trace_logger = get_trace_logger_instance()


router = APIRouter(prefix="/agents", tags=["Agents"]) # Consistent tag name

# =====================================================================
# Agent Management Endpoints
# =====================================================================

@router.post(
    "/",
    dependencies=[Depends(has_role("admin"))],
    response_model=AgentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create New Agent Configuration",
    description="Allows an admin user to create a new AI agent configuration in the database.",
)
async def create_agent(
    agent_data: AgentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Endpoint to create a new agent configuration (admin only).
    Ensures agent names are unique per user.
    """
    try:
        # 1. Check if an agent with this name already exists for this user.
        # Use .limit(1) as we only need to know if one exists, more efficient.
        result = await db.execute(
            select(AgentConfiguration)
            .where(
                AgentConfiguration.owner_id == current_user.id,
                AgentConfiguration.name == agent_data.name,
            )
            .limit(1)
        )
        existing_agent = result.scalar_one_or_none()
        if existing_agent:
            # Rollback prior to raising custom exception (best practice if it's not a standard HTTP error)
            await db.rollback()
            raise AgentNameAlreadyExistsError(
                agent_name=agent_data.name,
                user_id=current_user.id # Add user_id for better error context
            )

        # 2. Create a new AgentConfiguration object.
        agent_config = AgentConfiguration(
            owner_id=current_user.id,
            name=agent_data.name,
            system_prompt=agent_data.system_prompt,
            tools_config=agent_data.tools_config,
            llm_model_name=agent_data.llm_model_name,
            max_steps=agent_data.max_steps,
        )
        db.add(agent_config)

        # 3. Commit the changes to the database.
        await db.commit()

        # 4. Refresh the agent_config object to get the generated ID and relationships.
        await db.refresh(agent_config)

        logger.info(f"Agent '{agent_data.name}' created by user {current_user.email} (ID: {agent_config.id})")

        # 5. Return the created agent configuration.
        return AgentRead.model_validate(agent_config) # Use model_validate for Pydantic v2+

    except AgentNameAlreadyExistsError as e:
        # Custom exception automatically translated to HTTP 409 by global exception handler
        # if you have one, or raise specific HTTPException here if not.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.message # Use the message from your custom exception
        )
    except Exception as e:
        logger.error(
            f"Failed to create agent for user {current_user.email} with name '{agent_data.name}': {e.__class__.__name__}: {e}",
            exc_info=True, # Log full traceback
        )
        await db.rollback() # Ensure rollback on general errors too
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while creating the agent.",
        )

# =====================================================================
# Development/Testing Endpoints - CRITICAL SECURITY CONSIDERATION
# =====================================================================

@router.post(
    "/test-tool-sandbox",
    response_model=MCPToolResponse,
    status_code=status.HTTP_200_OK,
    summary="Test Tool Execution in Sandbox (DEV ONLY)",
    description="""
    **WARNING: This endpoint is for development and testing purposes ONLY.**
    It allows direct execution of a tool call in the sandbox.
    It should be REMOVED or PROTECTED by strong authentication/environment checks
    in any production deployment.
    """,
    # Protect this endpoint aggressively in production
    dependencies=[Depends(has_role("admin"))] if settings.ENVIRONMENT == "prod" else []
)
async def test_tool_sandbox_execution(tool_call: MCPToolCall):
    """
    Endpoint to test tool execution in the sandbox.
    Sends an MCPToolCall to the sandbox service for execution.
    """
    if settings.ENVIRONMENT == "prod":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is disabled in production environments for security reasons."
        )

    logger.info(f"Received sandbox test request: Tool='{tool_call.tool_name}', Call ID='{tool_call.call_id or 'N/A'}'")
    try:
        response = await sandbox_service.run_tool_in_sandbox(tool_call)
        logger.info(f"Sandbox test response for '{tool_call.tool_name}': Status={response.status}")
        return response
    except Exception as e:
        logger.error(f"Sandbox test execution failed for tool '{tool_call.tool_name}': {e.__class__.__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sandbox test failed: {e}"
        )

# Add other agent management endpoints here (GET, PUT, DELETE)
# Example:
# @router.get("/{agent_id}", response_model=AgentRead)
# async def get_agent(agent_id: int, db: AsyncSession = Depends(get_db)):
#     # ... logic to fetch agent by ID ...
#     pass