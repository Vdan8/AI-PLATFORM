# backend/app/services/tool_registry.py
import json
import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4 # For better call_id generation

# Assuming these services/schemas are correctly defined and exist at these paths
from app.services.tool_loader import tool_loader_service
from app.services.sandbox_service import sandbox_service
# Corrected import: MCPToolParameter -> ToolParameter
from app.schemas.tool import MCPToolCall, MCPToolResponse, MCPToolDefinition, ToolParameter
from sqlalchemy.ext.asyncio import AsyncSession # For type hinting the db session

logger = logging.getLogger(__name__)

class ToolRegistryService:
    """
    Manages the retrieval, formatting, and execution of tools.
    Encapsulates logic for interacting with tool definitions from the database
    and delegating execution to the sandbox.
    """
    def __init__(self):
        # Services that ToolRegistryService depends on (injected as singletons)
        self.tool_loader = tool_loader_service
        self.sandbox = sandbox_service
        self._cached_tool_definitions: List[MCPToolDefinition] = [] # Cache for raw definitions

    async def get_all_tool_definitions_for_llm(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """
        Retrieves all available tool definitions from the database,
        formatted for LLM consumption (e.g., OpenAI function calling).
        """
        try:
            # Use the tool_loader_service to get MCPToolDefinition objects
            # It's good practice to refresh the cache here if this is the primary way tools are fetched
            mcp_tool_definitions: List[MCPToolDefinition] = await self.tool_loader.get_all_tool_definitions_for_llm(db)
            self._cached_tool_definitions = mcp_tool_definitions # Update cache

            if not isinstance(mcp_tool_definitions, list):
                logger.warning(
                    f"Expected tool_loader_service.get_all_tool_definitions to return a list, but received {type(mcp_tool_definitions)}. Returning empty list."
                )
                return []

            openai_tool_definitions = []
            for mcp_tool in mcp_tool_definitions:
                parameters_properties = {}
                parameters_required = []

                if mcp_tool.parameters: # Check if parameters list is not empty
                    for param in mcp_tool.parameters:
                        # Ensure 'type' is a valid JSON schema type (e.g., 'string', 'integer', 'boolean')
                        json_type = param.type.lower()
                        if json_type == 'int':
                            json_type = 'integer'
                        elif json_type == 'bool':
                            json_type = 'boolean'
                        elif json_type == 'float':
                            json_type = 'number'
                        elif json_type == 'list':
                            # For list, define as an array with default string items.
                            # Adjust 'items' type if you have specific list item types.
                            parameters_properties[param.name] = {
                                "type": "array",
                                "description": param.description or f"List of {param.name}",
                                "items": {"type": "string"} # Default item type, can be more specific
                            }
                            # Skip adding to parameters_properties with a simple type if it's a list/dict
                            continue # Move to next parameter after handling complex type
                        elif json_type == 'dict':
                            # For dict, define as an object.
                            # You might need to define 'properties' if the structure is known.
                            parameters_properties[param.name] = {
                                "type": "object",
                                "description": param.description or f"Dictionary for {param.name}",
                                "properties": {} # Define sub-properties here if structure is known
                            }
                            continue # Move to next parameter after handling complex type

                        # Default handling for primitive types (string, integer, boolean, number)
                        parameters_properties[param.name] = {
                            "type": json_type,
                            "description": param.description or f"Parameter for {param.name}",
                        }

                        # Corrected logic: Use param.required directly
                        if param.required:
                            parameters_required.append(param.name)

                # Ensure 'parameters' object is only included if there are properties or required fields
                parameters_schema = {
                    "type": "object",
                    "properties": parameters_properties,
                }
                if parameters_required:
                    parameters_schema["required"] = parameters_required

                openai_tool_definitions.append(
                    {
                        "type": "function",
                        "function": {
                            "name": mcp_tool.name,
                            "description": mcp_tool.description,
                            "parameters": parameters_schema,
                        },
                    }
                )
            logger.info(f"Loaded {len(openai_tool_definitions)} tools for LLM consumption.")
            return openai_tool_definitions
        except Exception as e:
            logger.error(f"Error fetching and formatting tool definitions for LLM: {e}", exc_info=True)
            raise # Re-raise for centralized error handling

    async def call_tool(self, tool_name: str, tool_arguments: dict, tool_call_id: Optional[str] = None) -> Any:
        """
        Executes the specified tool by delegating to the sandbox service.
        """
        if tool_call_id is None:
            tool_call_id = str(uuid4()) # Generate a UUID if not provided

        tool_call = MCPToolCall(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            call_id=tool_call_id,
        )
        logger.info(f"Delegating tool '{tool_name}' (ID: {tool_call_id}) to sandbox with args: {tool_arguments}")

        response: MCPToolResponse = await self.sandbox.run_tool_in_sandbox(tool_call)

        # Corrected: Check response.status == "success" instead of non-existent response.success
        if response.status == "success":
            try:
                # Attempt to parse output as JSON, but handle non-JSON strings
                parsed_output = json.loads(response.output)
                logger.info(f"Tool '{tool_name}' (ID: {tool_call_id}) executed successfully. Output (JSON): {parsed_output}")
                return parsed_output
            except (json.JSONDecodeError, TypeError):
                logger.debug(
                    f"Tool '{tool_name}' (ID: {tool_call_id}) output is not JSON, returning raw string: {response.output[:200]}..."
                )
                return response.output
        else:
            # Enhanced error handling for tool execution failure
            error_message = response.error_message or "Unknown error from sandbox."
            logger.error(f"Tool execution failed for '{tool_name}' (ID: {tool_call_id}): {error_message}")
            # Raise a specific exception that can be caught by the orchestrator
            raise RuntimeError(f"Tool execution failed for '{tool_name}': {error_message}")


    async def get_single_tool_definition_for_llm(self, db: AsyncSession, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetches a single tool definition from the database by name, formatted for LLM use.
        """
        try:
            # Use the tool_loader_service to get a single MCPToolDefinition object
            mcp_tool: Optional[MCPToolDefinition] = await self.tool_loader.get_tool_definition_by_name(db, tool_name)

            if not mcp_tool:
                logger.warning(f"Tool '{tool_name}' not found in database.")
                return None

            parameters_properties = {}
            parameters_required = []
            if mcp_tool.parameters:
                for param in mcp_tool.parameters:
                    json_type = param.type.lower()
                    if json_type == 'int':
                        json_type = 'integer'
                    elif json_type == 'bool':
                        json_type = 'boolean'
                    elif json_type == 'float':
                        json_type = 'number'
                    elif json_type == 'list':
                        parameters_properties[param.name] = {
                            "type": "array",
                            "description": param.description or f"List of {param.name}",
                            "items": {"type": "string"} # Default item type, can be more specific
                        }
                        continue
                    elif json_type == 'dict':
                        parameters_properties[param.name] = {
                            "type": "object",
                            "description": param.description or f"Dictionary for {param.name}",
                            "properties": {} # Define sub-properties here if structure is known
                        }
                        continue

                    parameters_properties[param.name] = {
                        "type": json_type,
                        "description": param.description or f"Parameter for {param.name}",
                    }
                    # Corrected logic: Use param.required directly
                    if param.required:
                        parameters_required.append(param.name)

            parameters_schema = {
                "type": "object",
                "properties": parameters_properties,
            }
            if parameters_required:
                parameters_schema["required"] = parameters_required

            logger.info(f"Loaded single tool '{tool_name}' for LLM consumption.")
            return {
                "type": "function",
                "function": {
                    "name": mcp_tool.name,
                    "description": mcp_tool.description,
                    "parameters": parameters_schema,
                },
            }
        except Exception as e:
            logger.error(f"Error fetching and formatting single tool definition '{tool_name}' for LLM: {e}", exc_info=True)
            raise # Re-raise

    async def get_tool_by_name(self, db: AsyncSession, tool_name: str) -> Optional[MCPToolDefinition]:
        """
        Retrieves a raw MCPToolDefinition object from the database by name.
        This is useful for internal logic that needs the full tool definition,
        not just the LLM-formatted version.
        """
        try:
            return await self.tool_loader.get_tool_definition_by_name(db, tool_name)
        except Exception as e:
            logger.error(f"Error retrieving MCPToolDefinition for '{tool_name}': {e}", exc_info=True)
            raise

    # NEW METHOD for AgentOrchestrator to get raw tool definitions for decision making
    def get_all_tool_definitions_list(self) -> List[MCPToolDefinition]:
        """
        Returns a list of all currently cached MCPToolDefinition objects.
        This is primarily for the AgentOrchestrator's internal decision-making
        where the raw definition (not the LLM-formatted one) is needed.
        The cache should be populated by calls to get_all_tool_definitions_for_llm
        or a dedicated refresh method.
        """
        if not self._cached_tool_definitions:
            logger.warning("Tool registry cache is empty. Call get_all_tool_definitions_for_llm first to populate.")
            # In a real system, you might trigger a refresh here if cache is empty,
            # but for this context, assume the LLM formatting call populates it.
        return self._cached_tool_definitions


# Instantiate the service as a singleton for easy import elsewhere
tool_registry_service = ToolRegistryService()