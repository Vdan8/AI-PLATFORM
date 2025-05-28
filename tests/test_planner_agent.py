import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.app.services.planner_agent import PlannerAgentService
from backend.app.schemas.tool import MCPToolDefinition, ToolParameter

@pytest.mark.asyncio
async def test_generate_plan_from_prompt_use_tool():
    # Setup
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create = AsyncMock(return_value=MockToolCallResponse())
    
    mock_registry = MagicMock()
    mock_registry.get_all_tool_definitions_list.return_value = [
        MCPToolDefinition(
            name="test_tool",
            description="Just a test tool",
            parameters=[ToolParameter(name="text", type="str", description="Some input", required=True)]
        )
    ]
    mock_registry.get_llm_tools_for_agent = AsyncMock(return_value=[{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "Just a test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Some input"}
                },
                "required": ["text"]
            }
        }
    }])

    mock_trace_logger = MagicMock()

    planner = PlannerAgentService(
        openai_client=mock_openai_client,
        registry=mock_registry,
        trace_logger=mock_trace_logger
    )

    context = {
        "user_prompt": "summarize this",
        "last_tool_output": None,
        "last_tool_status": None
    }

    plan = await planner.generate_plan_from_prompt(context, [], db_session=None)

    assert plan["action"] == "use_tool"
    assert plan["tool_name"] == "test_tool"
    assert "parameters" in plan

class MockToolCallResponse:
    def __init__(self):
        self.id = "mock-id"
        self.tool_calls = [self]
        self.function = self
        self.name = "test_tool"
        self.arguments = '{"text": "test input"}'
        self.content = None
