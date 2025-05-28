import logging # Import standard logging module
from typing import List, Literal, Any, Dict, Optional # Added Any, Dict for more precise type hints
import json

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field # Import Field for descriptions if needed

from app.services.llm_agent import generate_agent_profile
from app.services.tool_registry import ToolRegistryService # Use the service directly
from app.services.autonomy_loop import run_autonomous_agent # Assuming this is the orchestrator
from app.utils.logger import trace_logger_service # Corrected logger import
from backend.config.config import settings
from backend.app.core.llm_clients import openai_client
from app.models.base import get_db # To get a DB session
from sqlalchemy.ext.asyncio import AsyncSession # For type hinting DB session

import aiofiles # For asynchronous file I/O
import os

# --- Logger Setup ---
trace_logger = trace_logger_service() # Get the global trace logger instance
logger = logging.getLogger(__name__) # Get a standard logger for API-level logs

router = APIRouter(prefix="/agent_operations", tags=["Agent Operations"]) # Renamed tag for clarity

# ===========================
# Data Models
# ===========================

class PromptRequest(BaseModel):
    """Request model for generating an agent profile."""
    prompt: str = Field(..., min_length=1, description="A prompt describing the desired agent profile.")

class Message(BaseModel):
    """Represents a single message in a chat history."""
    role: Literal["user", "assistant", "tool"]
    content: str
    name: Optional[str] = None # Optional for 'tool' messages
    tool_call_id: Optional[str] = None # Optional for 'assistant' tool_call messages

class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    system_prompt: str = Field(..., description="The system prompt for the chat session.")
    history: List[Message] = Field(default_factory=list, description="A list of previous chat messages.")

class RunAgentRequest(BaseModel):
    """Request model for running an autonomous agent."""
    system_prompt: str = Field(..., description="The system prompt for the autonomous agent.")
    goal: str = Field(..., description="The primary goal for the autonomous agent to achieve.")
    max_steps: int = Field(5, gt=0, description="Maximum number of steps the agent can take.")

class ToolCallRequest(BaseModel):
    """Request model for directly executing a single tool."""
    args: Dict[str, Any] = Field(..., description="Dictionary of arguments for the tool function.")

# ===========================
# Routes
# ===========================

@router.post(
    "/generate-agent",
    summary="Generate Agent Profile",
    description="Generates an AI agent profile based on a given prompt.",
    response_model=Dict[str, Any], # Assuming generate_agent_profile returns a dict
    status_code=status.HTTP_200_OK
)
async def generate_agent(req: PromptRequest): # Make function async
    """Endpoint to generate a new agent profile."""
    try:
        # Assuming generate_agent_profile is an async function or handles its own async I/O
        result = await generate_agent_profile(req.prompt)
        return {"agent_profile": result}
    except Exception as e:
        logger.error(f"Agent profile generation failed: {e.__class__.__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate agent profile."
        )

@router.post(
    "/chat",
    summary="Chat with Agent (Tool-Enabled)",
    description="Engages in a chat conversation with the LLM, enabling tool usage.",
    response_model=Dict[str, Any], # Can return content or tool_calls
    status_code=status.HTTP_200_OK
)
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db) # Inject database session
):
    """Endpoint to handle chat interactions with tool calling capabilities."""
    try:
        tool_registry_service = ToolRegistryService(db=db)
        # Fetch tool definitions as Pydantic models for OpenAI's `tools` parameter
        # OpenAI expects a list of tool definitions, e.g., from ToolRegistryService
        tools_definitions_for_llm = await tool_registry_service.get_all_tool_definitions_for_llm()


        messages = [{"role": "system", "content": req.system_prompt}] + [
            m.model_dump(exclude_unset=True) # Use model_dump to convert Pydantic to dict, excluding unset fields
            for m in req.history
        ]

        response = await openai_client.chat.completions.create( # Await the OpenAI call
            model=settings.LLM_MODEL,
            messages=messages,
            tools=tools_definitions_for_llm, # Pass the correctly formatted tool definitions
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            tool_outputs = []
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    # It's safer to parse arguments as JSON. A bad LLM response might send invalid JSON.
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    logger.warning(f"LLM returned invalid JSON for tool arguments: {tool_call.function.arguments}")
                    # You might want to return an error or a specific message to the LLM
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "result": f"Error: Invalid JSON arguments for tool {name}."
                    })
                    continue # Skip to next tool call

                # Call tool via the service, passing the database session
                tool_result = await tool_registry_service.call_tool(
                    tool_name=name,
                    tool_arguments=arguments,
                    tool_call_id=tool_call.id # Pass call_id for potential tracing
                )
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "result": tool_result
                })
            return {"tool_calls": tool_outputs}

        return {"response": message.content}
    except Exception as e:
        logger.error(f"Chat API failed: {e.__class__.__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request."
        )

@router.post(
    "/run-agent",
    summary="Run Autonomous Agent",
    description="Initiates an autonomous AI agent to achieve a specified goal.",
    response_model=Dict[str, Any], # Assuming run_autonomous_agent returns a dict/json-compatible result
    status_code=status.HTTP_200_OK
)
async def run_agent(
    req: RunAgentRequest,
    db: AsyncSession = Depends(get_db) # Inject database session
):
    """Endpoint to run an autonomous agent."""
    try:
        # Assuming run_autonomous_agent needs the DB session
        result = await run_autonomous_agent(
            db=db, # Pass db session
            system_prompt=req.system_prompt,
            goal=req.goal,
            max_steps=req.max_steps
        )
        return {"result": result}
    except Exception as e:
        logger.error(f"Autonomous agent run failed: {e.__class__.__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run autonomous agent."
        )

@router.post(
    "/tools/{tool_name}",
    summary="Execute Single Tool",
    description="Directly executes a specific tool by its name with provided arguments. **ADMIN/DEV ONLY IN PROD**",
    response_model=Dict[str, Any], # Assuming tool result is dict/json-compatible
    status_code=status.HTTP_200_OK
)
async def run_single_tool(
    tool_name: str,
    req: ToolCallRequest,
    db: AsyncSession = Depends(get_db) # Inject database session
    # Add a security dependency here for production, e.g., Depends(has_role("admin"))
    # dependencies=[Depends(has_role("admin"))] # Uncomment and import has_role if needed
):
    """
    Endpoint to execute a single tool directly.
    Consider restricting this endpoint in production environments.
    """
    if settings.ENVIRONMENT == "prod":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Direct tool execution endpoint is disabled in production environments for security reasons."
        )
    try:
        tool_registry_service = ToolRegistryService(db=db)
        result = await tool_registry_service.call_tool(
            tool_name=tool_name,
            tool_arguments=req.args,
            tool_call_id=f"direct-call-{tool_name}-{os.urandom(4).hex()}" # Generate a simple ID
        )
    except Exception as e:
        logger.error(f"Direct tool execution failed for '{tool_name}': {e.__class__.__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Direct tool execution failed: {e}"
        )

    # Log the tool execution event
    await trace_logger.log_event(
        event_type="direct_tool_execution",
        payload={
            "tool_name": tool_name,
            "tool_args": req.args,
            "tool_result": result,
        }
    )

    return {"result": result}


@router.get(
    "/trace",
    summary="Retrieve Trace Logs",
    description="Retrieves the application's trace logs (for debugging/monitoring). **ADMIN/DEV ONLY IN PROD**",
    response_model=Dict[str, Any], # Returns a dict with a 'logs' key
    status_code=status.HTTP_200_OK
    # Add a security dependency here for production, e.g., Depends(has_role("admin"))
    # dependencies=[Depends(has_role("admin"))] # Uncomment and import has_role if needed
)
async def get_trace_logs(): # Make function async
    """Endpoint to retrieve trace logs."""
    if settings.ENVIRONMENT == "prod":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Trace log retrieval endpoint is disabled in production environments for security reasons."
        )
    try:
        logs = []
        log_path = settings.LOG_FILE_PATH

        if os.path.exists(log_path):
            # Use aiofiles for asynchronous file reading
            async with aiofiles.open(log_path, mode="r") as f:
                async for line in f:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON line in log file: {line.strip()}")
                        continue # Skip malformed lines

        return {"logs": logs[::-1]}  # reverse for most recent first
    except Exception as e:
        logger.error(f"Failed to retrieve trace logs: {e.__class__.__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trace logs."
        )