import logging
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from openai import APIStatusError, APIConnectionError, InternalServerError, RateLimitError, APITimeoutError

from backend.config.config import settings
from app.schemas.tool import MCPToolDefinition, ToolParameter, MCPToolCall, MCPToolResponse
from app.services.tool_registry import ToolRegistryService
from app.services.tool_resolver import ToolResolverService
from app.services.planner_agent import PlannerAgentService
from app.services.tool_loader import tool_loader_service
from backend.app.services.sandbox_executor import sandbox_service, SandboxService
from app.utils.logger import TraceLogger

logger = logging.getLogger(__name__)

class ExecutorService:
    def __init__(
        self,
        planner: PlannerAgentService,
        resolver: ToolResolverService,
        registry: ToolRegistryService,
        sandbox: SandboxService,
        trace_logger: TraceLogger
    ):
        self.planner = planner
        self.resolver = resolver
        self.registry = registry
        self.sandbox = sandbox
        self.trace_logger = trace_logger
        self.max_iterations = settings.ORCHESTRATOR_MAX_ITERATIONS

    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any], call_id: str, db_session: Any) -> Tuple[str, Any]:
        """Executes a tool and returns its status and output/error."""
        await self.trace_logger.log_event("Executor: Attempting Tool Execution", {"tool_name": tool_name, "args": tool_args})
        logger.info(f"Executor: Attempting to execute tool: {tool_name} with args: {tool_args}")

        try:
            # You need a database session to use get_tool_by_name
            db_tool = await tool_loader_service.get_tool_by_name(db_session, tool_name)

            if db_tool is None or not db_tool.code or db_tool.code.strip() == "":
                error_msg = f"Tool '{tool_name}' not found or contains no code in the database."
                logger.error(error_msg)
                return "error", {"message": error_msg}

            tool_script_content = db_tool.code # Extract the code from the Tool object

            # Execute via sandbox
            # Create an MCPToolCall object as expected by run_tool_in_sandbox
            tool_call_obj = MCPToolCall(
                tool_name=tool_name,
                tool_arguments=tool_args, # Corrected: Use 'tool_arguments' as per MCPToolCall schema
                call_id=call_id
            )

            execution_result: MCPToolResponse = await sandbox_service.run_tool_in_sandbox(
                tool_call=tool_call_obj,
                tool_script_content=tool_script_content
            )

            # Process the execution_result
            if execution_result.status == "success":
                logger.info(f"Tool '{tool_name}' executed successfully. Output: {execution_result.output}")
                await self.trace_logger.log_event("Tool Execution Success", {
                    "tool_name": tool_name,
                    "output": execution_result.output,
                    "call_id": call_id
                })
                return "success", execution_result.output
            else:
                # Prioritize error_message if available, otherwise use output content or a default message
                error_msg = execution_result.error_message
                if not error_msg and isinstance(execution_result.output, dict):
                    error_msg = execution_result.output.get("message", f"Unknown error during tool '{tool_name}' execution.")
                elif not error_msg and isinstance(execution_result.output, str):
                    error_msg = execution_result.output
                elif not error_msg:
                    error_msg = f"Unknown error during tool '{tool_name}' execution."

                logger.error(f"Tool '{tool_name}' execution failed: {error_msg}")
                await self.trace_logger.log_event("Tool Execution Failed", {
                    "tool_name": tool_name,
                    "error": error_msg,
                    "call_id": call_id
                })
                return "error", {"message": error_msg}

        except Exception as e:
            error_msg = f"An unexpected error occurred during execution of tool '{tool_name}': {e}"
            logger.error(error_msg, exc_info=True)
            await self.trace_logger.log_event("Tool Execution Exception", {
                "tool_name": tool_name,
                "error": str(e),
                "call_id": call_id
            })
            return "error", {"message": error_msg}

    async def _handle_respond_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Handle respond action type."""
        final_response = {
            "status": "success",
            "message": decision.get("response_message", "Task completed.")
        }
        await self.trace_logger.log_event("Orchestrator: Final Response Decided", final_response)
        return final_response

    async def _handle_generate_tool_action(
        self, 
        decision: Dict[str, Any], 
        db_session: Any,
        current_context: Dict[str, Any],
        thought_history: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Handle generate_tool action type. Returns (should_break, final_response)."""
        tool_name = decision.get("tool_name")
        tool_description = decision.get("description")
        llm_decision_parameters = decision.get("parameters", {})

        await self.trace_logger.log_event("Orchestrator: Attempting Tool Generation", {
            "tool_name": tool_name,
            "description": tool_description,
            "parameters": llm_decision_parameters
        })

        # Step 2: Resolver handles tool generation
        generation_success, error_message = await self.resolver.resolve_tool_generation(
            tool_name,
            tool_description,
            llm_decision_parameters,
            db_session
        )

        if generation_success:
            current_context["last_tool_output"] = f"Tool '{tool_name}' successfully generated and registered."
            current_context["last_tool_status"] = "success"
            thought_history.append({"action": "generate_tool", "tool_name": tool_name, "outcome": "success"})
            logger.info(f"Successfully generated tool: {tool_name}. Will continue to next iteration.")
            return False, None
        else:
            current_context["last_tool_output"] = f"Failed to generate tool '{tool_name}': {error_message}"
            current_context["last_tool_status"] = "error"
            thought_history.append({"action": "generate_tool", "tool_name": tool_name, "outcome": "failed", "error": error_message})
            logger.error(f"Failed to generate tool: {tool_name}. Error: {error_message}")
            final_response = {"status": "error", "message": f"Failed to generate tool '{tool_name}': {error_message}"}
            return True, final_response

    async def _handle_use_tool_action(
        self,
        decision: Dict[str, Any],
        user_prompt: str,
        db_session: Any,
        current_context: Dict[str, Any],
        thought_history: List[Dict[str, Any]]
    ) -> bool:
        """Handle use_tool action type. Returns should_continue."""
        tool_name = decision.get("tool_name")
        llm_decision_parameters = decision.get("parameters", {})
        call_id = decision.get("call_id", "unknown")

        await self.trace_logger.log_event("Orchestrator: Attempting Tool Parameter Resolution", {
            "tool_name": tool_name,
            "llm_params": llm_decision_parameters
        })

        # Step 2: Resolver handles tool parameter validation
        extracted_tool_args, param_error_message = await self.resolver.resolve_tool_parameters_for_execution(
            tool_name,
            llm_decision_parameters,
            user_prompt,
            db_session
        )

        if param_error_message:
            current_context["last_tool_output"] = f"Parameter error for tool '{tool_name}': {param_error_message}"
            current_context["last_tool_status"] = "error"
            thought_history.append({"action": "use_tool", "tool_name": tool_name, "outcome": "param_error", "error": param_error_message})
            logger.warning(f"Parameter resolution failed for tool '{tool_name}': {param_error_message}. Returning to planner.")
            return True  # Continue to next iteration

        # Step 3: Execute the tool
        tool_status, tool_output = await self._execute_tool(tool_name, extracted_tool_args, call_id, db_session)

        current_context["last_tool_output"] = tool_output
        current_context["last_tool_status"] = tool_status
        thought_history.append({"action": "use_tool", "tool_name": tool_name, "outcome": tool_status, "output": tool_output})

        if tool_status == "success":
            logger.info(f"Tool '{tool_name}' execution successful. Continuing to next iteration.")
        else:
            logger.warning(f"Tool '{tool_name}' execution failed. Continuing to next iteration for re-evaluation.")
        
        return False  # Don't continue, proceed to next iteration

    async def _handle_use_piece_action(
        self,
        decision: Dict[str, Any],
        current_context: Dict[str, Any],
        thought_history: List[Dict[str, Any]]
    ) -> bool:
        """Handle use_piece action type. Returns should_continue."""
        from app.services.activepieces_engine import run_activepieces_flow

        flow_path = decision.get("flow_path")
        inputs = decision.get("inputs", {})
        call_id = decision.get("call_id", "unknown")

        await self.trace_logger.log_event("Orchestrator: Running Activepieces Piece", {
            "flow_path": flow_path,
            "inputs": inputs,
            "call_id": call_id
        })

        result = await run_activepieces_flow(flow_path, inputs)

        current_context["last_tool_output"] = result.get("output", result.get("message"))
        current_context["last_tool_status"] = result["status"]
        thought_history.append({
            "action": "use_piece",
            "flow_path": flow_path,
            "outcome": result["status"],
            "output": result.get("output", result.get("message"))
        })

        if result["status"] == "success":
            logger.info(f"Activepieces flow '{flow_path}' executed successfully.")
            return False  # Don't continue
        else:
            logger.warning(f"Activepieces flow '{flow_path}' failed: {result.get('message')}")
            return True  # Continue to next iteration

    async def _handle_error_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Handle error action type from planner (e.g., JSON parsing errors)."""
        error_message = decision.get("message", "Unknown error from planner")
        logger.error(f"Planner returned error action: {error_message}")
        return {
            "status": "error", 
            "message": f"Planning error: {error_message}"
        }

    async def process_user_prompt(self, user_prompt: str, db_session: Any) -> Dict[str, Any]:
        """
        Orchestrates the AI agent's response to a user prompt, potentially involving multiple steps.
        This is the main chaining loop.
        """
        await self.trace_logger.log_event("Orchestrator: User Prompt Received", {"prompt": user_prompt})
        logger.info(f"Starting orchestration for user prompt: '{user_prompt}'")

        thought_history: List[Dict[str, Any]] = []
        current_context: Dict[str, Any] = {"user_prompt": user_prompt}
        final_response: Dict[str, Any] = {"status": "error", "message": "No response generated."}

        for iteration in range(self.max_iterations):
            await self.trace_logger.log_event(f"Orchestrator: Iteration {iteration + 1}", {"current_context": current_context})
            logger.info(f"--- Orchestrator Iteration {iteration + 1}/{self.max_iterations} ---")
            logger.info(f"Current Context: {current_context}")
            logger.info(f"Thought History: {thought_history}")

            try:
                # Step 1: Planner decides the next action
                decision = await self.planner.generate_plan_from_prompt(current_context, thought_history, db_session)
                action_type = decision.get("action")

                await self.trace_logger.log_event(f"Orchestrator: Planner Decision (Iteration {iteration + 1})", decision)
                logger.info(f"Planner decided: {action_type}")

                # Handle different action types
                if action_type == "respond":
                    final_response = await self._handle_respond_action(decision)
                    break

                elif action_type == "generate_tool":
                    should_break, error_response = await self._handle_generate_tool_action(
                        decision, db_session, current_context, thought_history
                    )
                    if should_break:
                        final_response = error_response
                        break

                elif action_type == "use_tool":
                    should_continue = await self._handle_use_tool_action(
                        decision, user_prompt, db_session, current_context, thought_history
                    )
                    if should_continue:
                        continue

                elif action_type == "use_piece":
                    should_continue = await self._handle_use_piece_action(
                        decision, current_context, thought_history
                    )
                    if should_continue:
                        continue

                elif action_type == "error":
                    final_response = await self._handle_error_action(decision)
                    break

                else:
                    logger.error(f"Unknown action type received from planner: {action_type}. Terminating.")
                    final_response = {
                        "status": "error", 
                        "message": f"The AI decided on an unknown action type: {action_type}. Please contact support."
                    }
                    break

            except (APIStatusError, APIConnectionError, InternalServerError, RateLimitError, APITimeoutError) as e:
                logger.error(f"LLM API communication error during chaining for prompt '{user_prompt}': {e}", exc_info=True)
                await self.trace_logger.log_event("Chaining LLM API Error", {"error": str(e), "prompt": user_prompt, "iteration": iteration})
                final_response = {
                    "status": "error", 
                    "message": "I'm having trouble communicating with the AI services during this multi-step process. Please try again shortly."
                }
                break
            except Exception as e:
                logger.exception(f"An unexpected error occurred during orchestration for prompt '{user_prompt}' at iteration {iteration + 1}.")
                await self.trace_logger.log_event("Orchestrator Unexpected Error", {"error": str(e), "prompt": user_prompt, "iteration": iteration})
                final_response = {
                    "status": "error", 
                    "message": f"An unexpected internal error occurred: {e}. Please try again later."
                }
                break

        # Handle max iterations reached
        if final_response.get("status") != "success" and iteration == self.max_iterations - 1:
            logger.warning(f"Orchestrator reached max iterations ({self.max_iterations}) without a successful final response for prompt: '{user_prompt}'.")
            await self.trace_logger.log_event("Orchestrator Max Iterations Reached", {"prompt": user_prompt})
            if "No response generated." in final_response.get("message", ""):
                final_response = {
                    "status": "error", 
                    "message": "I couldn't complete the task within the allowed steps. Can you provide more details or try a different request?"
                }

        await self.trace_logger.log_event("Orchestrator: Process Completed", final_response)
        logger.info(f"Orchestration completed for prompt: '{user_prompt}'. Final response: {final_response}")
        return final_response