# backend/app/api/agent.py
from fastapi import APIRouter, Depends, status, HTTPException
from app.models.user import User
from app.core.auth import get_current_user, has_role
from app.utils.logger import logger
from app.models.base import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.agent import AgentConfiguration
from app.schemas.agent import AgentCreate, AgentRead
from app.exceptions import AgentNameAlreadyExistsError

# NEW IMPORTS for Sandbox Testing
from app.schemas.tool import MCPToolCall, MCPToolResponse # Import tool schemas
from app.services.sandbox_service import sandbox_service   # Import the sandbox service instance

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post(
    "/",
    dependencies=[Depends(has_role("admin"))],
    response_model=AgentRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_agent(
    agent_data: AgentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Endpoint to create a new agent (admin only)."""
    try:
        # 1. Check if an agent with this name already exists for this user.
        result = await db.execute(
            select(AgentConfiguration).where(
                AgentConfiguration.owner_id == current_user.id,
                AgentConfiguration.name == agent_data.name,
            )
        )
        existing_agent = result.scalar_one_or_none()
        if existing_agent:
            raise AgentNameAlreadyExistsError(
                agent_name=agent_data.name
            )  # Use custom exception

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

        # 4. Refresh the agent_config object to get the generated ID.
        await db.refresh(agent_config)

        # 5. Return the created agent configuration.
        return AgentRead.from_orm(agent_config)
    except AgentNameAlreadyExistsError as e:  # Catch custom exception
        await db.rollback()
        raise e
    except Exception as e:
        logger.error(
            f"Failed to create agent for user {current_user.email}: {str(e)}",
            exc_info=True,
        )
        await db.rollback()
        raise HTTPException(  # Raise HTTPException here
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create agent",
        )

# =====================================================================
# NEW: Temporary endpoint to test sandbox execution
# This is for development and testing purposes. Remove or restrict access
# in a production environment.
@router.post("/test-tool-sandbox", response_model=MCPToolResponse, status_code=status.HTTP_200_OK)
async def test_tool_sandbox_execution(tool_call: MCPToolCall):
    """
    Endpoint to test tool execution in the sandbox.
    Sends an MCPToolCall to the sandbox service for execution.
    """
    logger.info(f"Received request to test tool in sandbox: {tool_call.tool_name} (Call ID: {tool_call.call_id or 'N/A'})")
    response = await sandbox_service.run_tool_in_sandbox(tool_call)
    logger.info(f"Sandbox response for tool '{tool_call.tool_name}': Status={response.status}")
    return response
# =====================================================================