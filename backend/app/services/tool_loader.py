# backend/app/services/tool_loader.py
from sqlalchemy.ext.asyncio import AsyncSession # CORRECTED: Use AsyncSession
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from app.models.tool import Tool
from app.schemas.tool import MCPToolDefinition, ToolParameter
from app.crud.tool import tool_crud
from app.models.base import get_db as get_async_db_session

class ToolLoaderService:
    """
    Manages loading and retrieving tool definitions from the database.
    This service acts as the central registry for available tools.
    It replaces the file-based tool loading system.
    """

    def __init__(self):
        pass

    @asynccontextmanager
    async def get_db_session(self):
        """
        Provides an async context manager for database sessions using the project's get_db.
        """
        async for db_session in get_async_db_session():
            yield db_session

    async def get_tool_by_name(self, db: AsyncSession, name: str) -> Optional[Tool]: # CORRECTED: async def
        """
        Retrieves a single tool by its name from the database.
        Returns the SQLAlchemy Tool model instance, which includes its code.
        """
        return await tool_crud.get_by_name(db, name) # CORRECTED: await

    async def get_tool_definition_for_llm(self, db: AsyncSession, name: str) -> Optional[MCPToolDefinition]: # CORRECTED: async def
        """
        Retrieves a tool by name and converts it into the MCPToolDefinition schema
        suitable for LLM consumption (e.g., function calling).
        """
        db_tool = await self.get_tool_by_name(db, name) # CORRECTED: await
        if db_tool:
            llm_parameters = []
            if isinstance(db_tool.parameters, dict) and "properties" in db_tool.parameters:
                for param_name, param_schema in db_tool.parameters.get("properties", {}).items():
                    llm_parameters.append(ToolParameter(
                        name=param_name,
                        type=param_schema.get("type", "string"),
                        description=param_schema.get("description", ""),
                        required=param_name in db_tool.parameters.get("required", [])
                    ))
            elif isinstance(db_tool.parameters, list):
                for p in db_tool.parameters:
                    try:
                        llm_parameters.append(ToolParameter(**p))
                    except Exception as e:
                        print(f"Warning: Could not parse parameter {p} for tool {name}: {e}")
            
            return MCPToolDefinition(
                name=db_tool.name,
                description=db_tool.description,
                parameters=llm_parameters
            )
        return None

    async def get_all_tool_definitions_for_llm(self, db: AsyncSession, skip: int = 0, limit: int = 100) -> List[MCPToolDefinition]: # CORRECTED: async def
        """
        Retrieves all tool definitions from the database, formatted for LLM consumption.
        """
        db_tools = await tool_crud.get_all(db, skip=skip, limit=limit) # CORRECTED: await
        llm_definitions = []
        for db_tool in db_tools:
            llm_parameters = []
            if isinstance(db_tool.parameters, dict) and "properties" in db_tool.parameters:
                for param_name, param_schema in db_tool.parameters.get("properties", {}).items():
                    llm_parameters.append(ToolParameter(
                        name=param_name,
                        type=param_schema.get("type", "string"),
                        description=param_schema.get("description", ""),
                        required=param_name in db_tool.parameters.get("required", [])
                    ))
            elif isinstance(db_tool.parameters, list):
                for p in db_tool.parameters:
                    try:
                        llm_parameters.append(ToolParameter(**p))
                    except Exception as e:
                        print(f"Warning: Could not parse parameter {p} for tool {db_tool.name}: {e}")

            llm_definitions.append(MCPToolDefinition(
                name=db_tool.name,
                description=db_tool.description,
                parameters=llm_parameters
            ))
        return llm_definitions

# Instantiate the service for easy import
tool_loader_service = ToolLoaderService()