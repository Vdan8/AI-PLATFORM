# backend/app/api/tools.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

# Import your schemas and CRUD operations
from app.schemas.tool import ToolCreate, ToolRead, ToolUpdate
from app.crud.tool import tool_crud
from app.models.base import get_db # This function provides the AsyncSession dependency

# Define the FastAPI APIRouter instance
# This 'router' object is what main.py imports and includes.
router = APIRouter()

@router.post("/", response_model=ToolRead, status_code=status.HTTP_201_CREATED)
async def create_tool(
    tool_in: ToolCreate,
    db: AsyncSession = Depends(get_db), # Dependency injection for async DB session
):
    """
    **Create a new tool.**

    A tool represents a function or capability that the AI agent can use.
    The `name` of the tool must be unique.
    """
    # Check if a tool with the given name already exists
    db_tool = await tool_crud.get_by_name(db, name=tool_in.name)
    if db_tool:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tool with name '{tool_in.name}' already exists."
        )
    # Create the tool in the database
    return await tool_crud.create(db=db, tool_in=tool_in)

@router.get("/{tool_id}", response_model=ToolRead)
async def read_tool(
    tool_id: int,
    db: AsyncSession = Depends(get_db), # Dependency injection for async DB session
):
    """
    **Retrieve a tool by its ID.**
    """
    # Retrieve the tool by ID
    tool = await tool_crud.get(db=db, tool_id=tool_id)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool not found."
        )
    return tool

@router.get("/", response_model=List[ToolRead])
async def read_tools(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db), # Dependency injection for async DB session
):
    """
    **Retrieve a list of all tools.**

    Supports pagination with `skip` and `limit` parameters.
    """
    # Retrieve all tools from the database with pagination
    tools = await tool_crud.get_all(db=db, skip=skip, limit=limit)
    return tools

@router.put("/{tool_id}", response_model=ToolRead)
async def update_tool(
    tool_id: int,
    tool_in: ToolUpdate,
    db: AsyncSession = Depends(get_db), # Dependency injection for async DB session
):
    """
    **Update an existing tool by ID.**

    Allows updating various fields of a tool.
    If the name is updated, it must remain unique.
    """
    # Get the existing tool from the database
    db_tool = await tool_crud.get(db=db, tool_id=tool_id)
    if not db_tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool not found."
        )

    # If the name is being updated, ensure the new name is unique (excluding the current tool)
    if tool_in.name is not None and tool_in.name != db_tool.name:
        existing_tool_with_name = await tool_crud.get_by_name(db, name=tool_in.name)
        if existing_tool_with_name and existing_tool_with_name.id != tool_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tool with name '{tool_in.name}' already exists for another tool."
            )

    # Update the tool in the database
    return await tool_crud.update(db=db, db_tool=db_tool, tool_update=tool_in)

@router.delete("/{tool_id}", response_model=ToolRead)
async def delete_tool(
    tool_id: int,
    db: AsyncSession = Depends(get_db), # Dependency injection for async DB session
):
    """
    **Delete a tool by its ID.**
    """
    # Attempt to delete the tool
    tool = await tool_crud.delete(db=db, tool_id=tool_id)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool not found."
        )
    return tool