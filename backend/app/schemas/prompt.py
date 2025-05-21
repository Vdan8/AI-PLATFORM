# backend/app/schemas/prompt.py
from pydantic import BaseModel
from typing import Optional, Dict, Any

class PromptRequest(BaseModel):
    user_prompt: str

class PromptResponse(BaseModel):
    status: str
    message: str
    tool_output: Dict[str, Any] | None = None # Use Dict[str, Any] for flexibility
    tool_name: str | None = None
    thought_process: list[Dict[str, Any]] | None = None # List of dictionaries for thought history