# backend/app/services/tool_registry.py
import json
import logging  # Import logging module
from typing import List, Dict, Any, Optional

from app.services.tool_loader import tool_loader_service
from app.services.sandbox_service import sandbox_service
from app.models.base import get_db  # Import the corrected get_db
from app.schemas.tool import MCPToolCall, MCPToolResponse, MCPToolDefinition


# Initialize logger for this module
logger = logging.getLogger(__name__)


async def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Retrieves all available tool definitions from the database,
    formatted for LLM consumption (e.g., OpenAI function calling).
    """
    try:
        async with get_db() as db:
            db_tool_definitions: List[MCPToolDefinition] = (
                await tool_loader_service.get_all_tool_definitions_for_llm(db)
            )

        if not isinstance(db_tool_definitions, list):
            logger.warning(
                f"Expected get_all_tool_definitions_for_llm to return a list, but received {type(db_tool_definitions)}. Returning empty list."
            )
            return []

        openai_tool_definitions = []
        for mcp_tool in db_tool_definitions:
            parameters_properties = {}
            parameters_required = []

            if hasattr(mcp_tool, "parameters") and mcp_tool.parameters:
                for param in mcp_tool.parameters:
                    parameters_properties[param.name] = {
                        "type": param.type,
                        "description": param.description,
                    }
                    if param.required:
                        parameters_required.append(param.name)

            openai_tool_definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": mcp_tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": parameters_properties,
                            "required": parameters_required,
                        },
                    },
                }
            )
        logger.info(f"Loaded {len(openai_tool_definitions)} tools for LLM.")
        return openai_tool_definitions
    except Exception as e:
        logger.error(f"Error fetching tool definitions: {e}")
        raise



async def call_tool(name: str, arguments: dict) -> Any:
    """
    Executes the specified tool by delegating to the sandbox service.
    """
    tool_call = MCPToolCall(
        tool_name=name,
        tool_arguments=arguments,
        call_id="auto-generated",  # Consider a UUID for better ID generation
    )
    response: MCPToolResponse = await sandbox_service.run_tool_in_sandbox(tool_call)

    if response.status == "success":
        try:
            return json.loads(response.output)
        except (json.JSONDecodeError, TypeError):
            logger.debug(
                f"Tool '{name}' output is not JSON, returning raw string: {response.output[:100]}..."
            )
            return response.output
    else:
        error_detail = response.output.get("error", "Unknown error")
        logger.error(f"Tool execution failed for '{name}': {error_detail}")
        raise ValueError(f"Tool execution failed for '{name}': {error_detail}")



async def get_single_tool_definition_for_llm(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Fetches a single tool definition from the database by name, formatted for LLM use.
    """
    try:
        async with get_db() as db:
            mcp_tool: Optional[MCPToolDefinition] = (
                await tool_loader_service.get_tool_definition_for_llm_by_name(db, tool_name)
            )

        if not mcp_tool:
            logger.warning(f"Tool '{tool_name}' not found.")
            return None

        parameters_properties = {}
        parameters_required = []
        if hasattr(mcp_tool, "parameters") and mcp_tool.parameters:
            for param in mcp_tool.parameters:
                parameters_properties[param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    parameters_required.append(param.name)

        logger.info(f"Loaded single tool '{tool_name}' for LLM.")
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": {
                    "type": "object",
                    "properties": parameters_properties,
                    "required": parameters_required,
                },
            },
        }
    except Exception as e:
        logger.error(f"Error fetching single tool definition '{tool_name}': {e}")
        raise