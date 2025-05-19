# backend/app/schemas/agent.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime # Moved to the top

class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Name of the agent")
    system_prompt: str = Field(..., min_length=1, description="System prompt for the agent")
    tools_config: Optional[Dict[str, Any]] = Field(None, description="Configuration for tools used by the agent")
    llm_model_name: Optional[str] = Field(None, description="Name of the LLM model to use")
    max_steps: int = Field(10, ge=1, description="Maximum number of steps for the agent")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Web Research Agent",
                "system_prompt": "You are a helpful web research assistant.",
                "tools_config": {"search": {"max_results": 3}},
                "llm_model_name": "gpt-3.5-turbo",
                "max_steps": 20,
            }
        }

class AgentRead(BaseModel):
    id: int
    owner_id: int
    name: str
    system_prompt: str
    tools_config: Optional[Dict[str, Any]] = None
    llm_model_name: Optional[str] = None
    max_steps: int
    created_at: datetime # Used here
    updated_at: Optional[datetime] = None # And here

    class Config:
        from_attributes = True