print("--- agent_orchestrator.py is being executed! ---")
import logging
from typing import Any, Dict, List
from pydantic import ValidationError, BaseModel, Field, create_model
from openai import AsyncOpenAI

from app.services.tool_analyzer import ToolAnalyzer, ToolAnalysisResult, tool_analyzer_service
from app.services.tool_generator import ToolGeneratorService , tool_generator_service
from app.services.tool_registry import ToolRegistryService , tool_registry_service
from backend.app.services.sandbox_executor import SandboxService , sandbox_service
from app.crud.tool import CRUDBase , tool_crud
from app.schemas.tool import MCPToolDefinition, ToolParameter, MCPToolCall, MCPToolResponse, ToolCreate # Import ToolCreate
from app.utils.logger import TraceLogger,trace_logger_service
from backend.config.config import settings
from app.services.executor import ExecutorService
from app.services.planner_agent import PlannerAgentService
from app.services.tool_resolver import ToolResolverService



logger = logging.getLogger(__name__)

# Define a Pydantic schema for the LLM's parameter extraction output (can be reused)
class ExtractedParameters(BaseModel):
    parameters: Dict[str, Any] = Field(
        ...,
        description="A dictionary where keys are parameter names and values are the extracted values.",
    )

class AgentOrchestratorService:
    def __init__(
        self,
        executor: ExecutorService # Now only depends on the executor
    ):
        self.executor = executor
        logger.info("AgentOrchestratorService initialized with ExecutorService.")

    async def process_user_prompt(self, user_prompt: str, db_session: Any) -> Dict[str, Any]:
        """
        Delegates the user prompt processing to the ExecutorService.
        This method becomes the public interface for the orchestrator.
        """
        logger.info(f"AgentOrchestrator: Delegating prompt '{user_prompt}' to ExecutorService.")
        return await self.executor.process_user_prompt(user_prompt, db_session)

   
    def _add_thought(self, thought_history: List[Dict[str, Any]], role: str, content: Dict[str, Any]):
        """Adds a structured thought/action to the history."""
        thought_history.append({"role": role, "content": content})


# --- Initialization of the OpenAI Client and AgentOrchestratorService ---

# Crucial: Initialize OpenAI client with max_retries=0 so tenacity handles retries
# This prevents OpenAI's built-in retry logic from interfering with tenacity.
openai_client_instance = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    max_retries=0 # Disable OpenAI's internal retries for tenacity to take over
)

# Consider removing these debug prints in production, or moving them to proper logging
logger.debug(f"🪲DEBUG: Type of trace_logger_service before orchestrator init: {type(trace_logger_service)}")
logger.debug(f"🪲DEBUG: Value of trace_logger_service before orchestrator init: {trace_logger_service}")

# Instantiate the PlannerAgentService
planner_agent_instance = PlannerAgentService(
    openai_client=openai_client_instance, # Pass the shared client to Planner
    registry=tool_registry_service, # Planner needs registry
    trace_logger=trace_logger_service # Planner needs trace logger
)

# Instantiate the ToolResolverService
tool_resolver_instance = ToolResolverService(
    openai_client=openai_client_instance, # Resolver needs OpenAI client for parameter extraction and tool generation
    trace_logger=trace_logger_service,
    generator=tool_generator_service, # Resolver needs tool generator
    tool_crud=tool_crud, # Resolver needs tool crud for saving generated tools
    registry=tool_registry_service # Resolver needs registry to add new tools
)

# Instantiate the ExecutorService
# This service now encapsulates the Planner, Resolver, Registry, and Sandbox
executor_service_instance = ExecutorService(
    planner=planner_agent_instance,
    resolver=tool_resolver_instance,
    registry=tool_registry_service, # Executor needs registry to get tool functions
    sandbox=sandbox_service, # Executor needs sandbox for tool execution
    trace_logger=trace_logger_service
)

agent_orchestrator_service = AgentOrchestratorService(
    executor=executor_service_instance
)