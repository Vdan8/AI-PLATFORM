# backend/app/api/tool.py
from fastapi import APIRouter, Depends, HTTPException, status
from app.models.user import User
from app.core.auth import get_current_user
from app.utils.logger import logger
from app.services.tool_registry import get_tool_definitions, call_tool
from pydantic import BaseModel

router = APIRouter(prefix="/tools", tags=["tools"])

class RunToolRequest(BaseModel):
    args: dict

@router.get("/")
async def list_available_tools(user: User = Depends(get_current_user)):
    """Endpoint to list all available tools"""
    try:
        tools = get_tool_definitions()
        return {"tools": [tool["function"]["name"] for tool in tools]}
    except Exception as e:
        logger.error(f"Failed to list available tools for user {user.id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list available tools"
        )

@router.post("/{tool_name}")
async def run_specific_tool(tool_name: str, request: RunToolRequest, user: User = Depends(get_current_user)):
    """Endpoint to run a specific tool"""
    try:
        result = call_tool(tool_name, request.args)
        return {"result": result}
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to run tool '{tool_name}' for user {user.id} with args {request.args}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run tool '{tool_name}'"
        )

# You can add more tool-related endpoints here (e.g., get tool details)