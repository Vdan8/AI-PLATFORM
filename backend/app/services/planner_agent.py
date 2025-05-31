import logging
import json
import re
from typing import Any, Dict, List, Optional, Union
from openai import AsyncOpenAI, APIStatusError, APIConnectionError, InternalServerError, RateLimitError, APITimeoutError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_log, after_log
from backend.config.config import settings
from app.schemas.tool import MCPToolDefinition, ToolParameter
from app.services.tool_registry import ToolRegistryService # Will be passed in __init__
from app.utils.logger import TraceLogger # Will be passed in __init__

logger = logging.getLogger(__name__)

class PlannerAgentService:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        registry: ToolRegistryService,
        trace_logger: TraceLogger
    ):
        self.openai_client = openai_client
        self.registry = registry
        self.trace_logger = trace_logger
        self.response_model = settings.ORCHESTRATOR_LLM_RESPONSE_MODEL

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(settings.OPENAI_MAX_RETRIES),
        retry=retry_if_exception_type((
            APIStatusError,
            APIConnectionError,
            InternalServerError,
            RateLimitError,
            APITimeoutError
        )),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
        reraise=True
    )
    async def _call_llm_with_retries(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    ) -> Any:
        logger.debug(f"Attempting LLM call with model: {self.response_model}")
        if self.openai_client.api_key:
            logger.info(f"OpenAI API Key in use (first 5 chars): {self.openai_client.api_key[:5]}*****")
        else:
            logger.warning("OpenAI API Key is not set on the client!")

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.response_model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice
            )
            logger.debug(f"LLM call successful. Response ID: {response.id}")
            return response.choices[0].message
        except Exception as e:
            logger.error(f"LLM API call failed after retries: {e}", exc_info=True)
            raise

    def _build_decision_prompt(self, current_context: Dict[str, Any], thought_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Builds the system prompt for the LLM to decide the next action in the chaining process.
        The LLM needs to decide whether to:
        1. GENERATE a new tool.
        2. USE an existing tool.
        3. RESPOND directly to the user.
        """
        # Get available tools from registry's cache
        available_tools_raw = self.registry.get_all_tool_definitions_list()
        # Format for LLM's internal consideration
        available_tools_for_llm_consideration = []
        for tool in available_tools_raw:
            # Ensure parameters are correctly formatted for LLM (List[ToolParameter] -> List[dict])
            formatted_params = []
            # IMPORTANT: PlannerAgentService needs to know about ToolParameter schema for this loop
            # For now, let's assume it's just a dict, but later we'll import it.
            # For this step, we'll keep it simple, assuming tool.parameters is already structured for direct use.
            # However, the original code shows conversion.
            # So, you will need to add:
            # from app.schemas.tool import MCPToolDefinition, ToolParameter
            # to planner_agent.py imports.

            # This logic is adapted from your original _build_decision_prompt
            if isinstance(tool.parameters, list): # If it's already List[ToolParameter]
                formatted_params = [p.model_dump() for p in tool.parameters]
            elif isinstance(tool.parameters, dict) and "properties" in tool.parameters: # If it's JSON schema dict
                for p_name, p_info in tool.parameters["properties"].items():
                    formatted_params.append({
                        "name": p_name,
                        "type": p_info.get('type', 'str'),
                        "description": p_info.get('description', ''),
                        "required": p_name in tool.parameters.get('required', [])
                    })

            available_tools_for_llm_consideration.append(
                {"name": tool.name, "description": tool.description, "parameters": formatted_params}
            )

        context_str = json.dumps(current_context, indent=2)
        history_str = json.dumps(thought_history, indent=2)

        system_message = f"""
        You are an advanced AI orchestrator. Your goal is to fulfill the user's request by intelligently deciding the next best action.
        You can either:
        1. GENERATE a new tool (by outputting a JSON object).
        2. USE an existing tool (by calling a function).
        3. RESPOND directly to the user (by outputting a JSON object).

        If you decide to GENERATE or RESPOND, your decision must be returned as a JSON object, strictly following one of these formats:

        --- Option 1: GENERATE a new tool (JSON output) ---
        {{
            "action": "generate_tool",
            "tool_name": "suggested_unique_tool_name_in_snake_case",
            "description": "A clear, concise description of what this new tool will do.",
            "parameters": {{
                "param_name_1": {{"type": "str", "description": "Description of param 1", "optional": false}},
                "param_name_2": {{"type": "int", "description": "Description of param 2", "optional": true}}
            }}
        }}
        (Use this if no existing tool can fulfill the request, or a new, specific capability is required based on user prompt or previous tool outputs.)

        --- Option 3: RESPOND directly to the user (JSON output) ---
        {{
            "action": "respond",
            "response_message": "A clear and concise message to the user, indicating the task is complete, or asking for clarification, or stating that the task cannot be fulfilled."
        }}
        (Use this if the task is complete, if you need more information from the user, or if you cannot proceed with tools.)

        If you decide to USE an existing tool, you MUST use the tool calling mechanism. DO NOT output JSON.

        --- Current State and Context ---
        User's Original Prompt: "{current_context.get('user_prompt')}"
        Previous Tool Output (if any): {current_context.get('last_tool_output', 'None')}
        Previous Tool Status (if any): {current_context.get('last_tool_status', 'None')}
        Thought History (past steps, tools used, outcomes): {history_str}

        --- Available Tools ---
        {json.dumps(available_tools_for_llm_consideration, indent=2)}

        --- Your Task ---
        Analyze the original prompt, the current context, and the tool history.
        Based on the current state, decide the SINGLE BEST next action (GENERATE, USE, or RESPOND).
        If GENERATE or RESPOND, strictly output only the JSON object for your chosen action.
        If USE, call the appropriate tool with its parameters using the function calling mechanism.
        """
        return [{"role": "system", "content": system_message}]

    async def generate_plan_from_prompt(self, current_context: Dict[str, Any], thought_history: List[Dict[str, Any]], db_session: Any) -> Dict[str, Any]:
        """
        Calls the LLM to get the next action (plan).
        This function will wrap the LLM call and parse its response.
        """
        await self.trace_logger.log_event("Planner: Generating Plan", {"context": current_context, "history_length": len(thought_history)})
        logger.info("Planner: Fetching available tools for LLM decision.")

        # Ensure the tool registry is populated with current tools
        # This call will also populate the internal cache used by get_all_tool_definitions_list
        await self.registry.get_llm_tools_for_agent(db=db_session) # This populates the internal cache

        # Fetch available tools for the LLM to choose from
        llm_tools = await self.registry.get_llm_tools_for_agent(db=db_session) # This formats tools for OpenAI's `tools` parameter

        decision_messages = self._build_decision_prompt(current_context, thought_history)

        raw_llm_response_message = await self._call_llm_with_retries(
            messages=decision_messages,
            temperature=0.4,
            tools=llm_tools,
            tool_choice="auto"
        )

        if raw_llm_response_message.tool_calls:
            tool_call = raw_llm_response_message.tool_calls[0] # Assuming one tool call per turn
            action_type = "use_tool"
            tool_name = tool_call.function.name
            llm_decision_parameters_raw = json.loads(tool_call.function.arguments)
            logger.info(f"Planner: LLM decided to call tool via `tool_calls`: {tool_name}")
            await self.trace_logger.log_event("Planner: LLM Tool Call Decision", {
                "tool_name": tool_name,
                "function_arguments": llm_decision_parameters_raw,
                "call_id": tool_call.id
            })
            return {
                "action": action_type,
                "tool_name": tool_name,
                "parameters": llm_decision_parameters_raw,
                "call_id": tool_call.id # Pass call_id for tracing
            }

        elif raw_llm_response_message.content:
            raw_content = raw_llm_response_message.content
            logger.debug(f"Planner: Raw LLM content for decision (non-tool call): {raw_content}")

            # --- START FIX FOR INVALID JSON ---
            json_string = None
            # Attempt to extract JSON from a markdown code block (e.g., ```json { ... } ```)
            json_match = re.search(r'```json\s*(\{.*\})\s*```', raw_content, re.DOTALL)
            if json_match:
                json_string = json_match.group(1)
                logger.debug(f"Planner: Extracted JSON string from markdown: {json_string}")
            else:
                # If no markdown block, assume the entire content is JSON
                json_string = raw_content
                logger.debug("Planner: No JSON markdown block found. Attempting to parse raw content as JSON.")
            # --- END FIX FOR INVALID JSON ---

            try:
                if json_string: # Only attempt to load if we have a string to load
                    decision = json.loads(json_string)
                else: # Fallback if no string was extracted (shouldn't happen with default="{}")
                    raise json.JSONDecodeError("No JSON string extracted from LLM response.", raw_content, 0)

                action_type = decision.get("action")
                logger.info(f"Planner: LLM decided (via JSON content): {action_type}")
                await self.trace_logger.log_event("Planner: LLM Decision", decision)

                if action_type == "respond":
                    return {
                        "action": "respond",
                        "response_message": decision.get("response_message", "Action completed.")
                    }
                elif action_type == "generate_tool":
                    tool_name = decision.get("tool_name")
                    tool_description = decision.get("description")
                    llm_decision_parameters_raw = decision.get("parameters", {})
                    return {
                        "action": "generate_tool",
                        "tool_name": tool_name,
                        "description": tool_description,
                        "parameters": llm_decision_parameters_raw
                    }
                else:
                    logger.warning(f"Planner: LLM returned unknown action type in JSON: {action_type}. Assuming conversational fallback.")
                    return {"action": "respond", "response_message": "I'm not sure how to proceed with that. Could you clarify?"}

            except json.JSONDecodeError as e:
                logger.error(f"Planner: LLM returned invalid JSON for decision: {raw_content}. Error: {e}")
                return {"action": "error", "message": "I'm having trouble understanding my next steps. Please rephrase your request."}
        else:
            logger.error("Planner: LLM returned no content and no tool calls. Cannot proceed.")
            return {"action": "error", "message": "The AI did not provide a clear response. Please try again."}


