# backend/app/api/routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import json

from app.services.llm_agent import generate_agent_profile
from app.services.tool_registry import get_tool_definitions, call_tool
from app.services.autonomy_loop import run_autonomous_agent
from app.utils.logger import log_trace

import os
import json

router = APIRouter()

# ===========================
# Data Models
# ===========================

class PromptRequest(BaseModel):
    prompt: str

class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    name: str = None
    tool_call_id: str = None

class ChatRequest(BaseModel):
    system_prompt: str
    history: List[Message]

class RunAgentRequest(BaseModel):
    system_prompt: str
    goal: str
    max_steps: int = 5

class ToolCallRequest(BaseModel):
    args: dict

# ===========================
# Routes
# ===========================

@router.post("/generate-agent")
def generate_agent(req: PromptRequest):
    result = generate_agent_profile(req.prompt)
    return {"agent_profile": result}

@router.post("/chat")
def chat(req: ChatRequest):
    tools = get_tool_definitions()

    messages = [{"role": "system", "content": req.system_prompt}] + [
        {
            "role": m.role,
            "content": m.content,
            **({"name": m.name} if m.name else {}),
            **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {})
        } for m in req.history
    ]

    response = openai_client.chat.completions.create( # USING THE SHARED CLIENT
        model=settings.DEFAULT_GPT_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_outputs = []
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            result = call_tool(name, arguments)
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "name": name,
                "result": result
            })
        return {"tool_calls": tool_outputs}

    return {"response": message.content}

@router.post("/run-agent")
def run_agent(req: RunAgentRequest):
    result = run_autonomous_agent(req.system_prompt, req.goal, req.max_steps)
    return {"result": result}

@router.post("/tools/{tool_name}")
def run_single_tool(tool_name: str, req: ToolCallRequest):
    try:
        result = call_tool(tool_name, req.args)
    except Exception as e:
        log_trace({
            "tool_name": tool_name,
            "tool_args": req.args,
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {e}")

    log_trace({
        "tool_name": tool_name,
        "tool_args": req.args,
        "tool_result": result,
    })

    return {"result": result}


@router.get("/trace")
def get_trace_logs():
    logs = []
    log_path = "settings.LOG_FILE_PATH"

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except:
                    continue
    return {"logs": logs[::-1]}  # reverse for most recent first
