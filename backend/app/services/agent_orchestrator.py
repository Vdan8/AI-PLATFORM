print("--- agent_orchestrator.py is being executed! ---")
import ast
import logging
import json
import asyncio
from typing import Any, Dict, Optional, List, Type, Tuple
from pydantic import ValidationError, BaseModel, Field, create_model
from openai import OpenAI, AsyncOpenAI, APIStatusError, APIConnectionError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from sqlalchemy.exc import IntegrityError
from asyncpg.exceptions import UniqueViolationError
from uuid import uuid4 # For creating unique model names

from app.services.tool_analyzer import ToolAnalyzer, ToolAnalysisResult, tool_analyzer_service
from app.services.tool_generator import ToolGenerator , tool_generator_service
from app.services.tool_registry import ToolRegistryService , tool_registry_service
from app.services.sandbox_service import SandboxService , sandbox_service
from app.crud.tool import CRUDBase , tool_crud
from app.schemas.tool import MCPToolDefinition, ToolParameter, MCPToolCall, MCPToolResponse, ToolCreate # Import ToolCreate
from app.utils.logger import TraceLogger,trace_logger_service
from app.core.config import settings

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
        analyzer: ToolAnalyzer,
        generator: ToolGenerator,
        registry: ToolRegistryService,
        sandbox: SandboxService,
        tool_crud: CRUDBase,
        trace_logger:  TraceLogger,
        openai_client: AsyncOpenAI
    ):
        self.analyzer = analyzer
        self.generator = generator
        self.registry = registry
        self.sandbox = sandbox
        self.tool_crud = tool_crud
        self.trace_logger = trace_logger
        self.openai_client = openai_client

    # --- Retries for LLM API Calls ---
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10),
           stop=stop_after_attempt(3),
           retry=retry_if_exception_type((APIStatusError, APIConnectionError)),
           reraise=True)
    async def _call_llm_with_retries(self, messages: List[Dict[str, str]], response_format: Optional[Dict[str, str]] = None, temperature: float = 0.7) -> str:
        """
        Wrapper for OpenAI API calls with retry logic.
        """
        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API call failed after retries: {e}", exc_info=True)
            raise

    async def process_user_prompt(self, user_prompt: str, db_session: Any, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Processes a user prompt, potentially involving sequential tool chaining,
        tool generation, and conversational responses.
        """
        await self.trace_logger.log_event("Processing User Prompt (Start Chaining)", {"user_prompt": user_prompt})
        logger.info(f"Orchestrator processing prompt for chaining: '{user_prompt}'")

        current_context = {"user_prompt": user_prompt}
        thought_history = [] # To track previous LLM thoughts and tool actions
        final_response = None

        # Ensure the tool registry is populated with current tools
        # This call will also populate the internal cache used by get_all_tool_definitions_list
        await self.registry.get_all_tool_definitions_for_llm(db=db_session)

        for iteration in range(max_iterations):
            await self.trace_logger.log_event("Chaining Iteration Start", {"iteration": iteration, "context": current_context})
            logger.info(f"--- Chaining Iteration {iteration + 1} ---")

            try:
                # --- Step 1: LLM-based Decision Making (What to do next?) ---
                decision_messages = self._build_decision_prompt(current_context, thought_history)

                raw_decision_json = await self._call_llm_with_retries(
                    messages=decision_messages,
                    response_format={"type": "json_object"},
                    temperature=0.4 # More deterministic for decision making
                )

                try:
                    decision = json.loads(raw_decision_json)
                    action_type = decision.get("action")
                    await self.trace_logger.log_event("LLM Decision", decision)
                    logger.info(f"LLM decided: {action_type}")
                except json.JSONDecodeError as e:
                    logger.error(f"LLM returned invalid JSON for decision: {raw_decision_json}. Error: {e}")
                    thought_history.append({"thought": "LLM failed to produce valid decision JSON. Trying a conversational fallback."})
                    final_response = {"status": "error", "message": "I'm having trouble understanding my next steps. Please rephrase your request."}
                    break # Exit loop

                if action_type == "respond":
                    final_response = {"status": "success", "message": decision.get("response_message", "Action completed.")}
                    logger.info(f"LLM decided to respond: {final_response['message']}")
                    break # Exit loop, task is done

                elif action_type in ["generate_tool", "use_tool"]:
                    tool_name = decision.get("tool_name")
                    tool_description = decision.get("description") # Only for generate_tool

                    # Capture the LLM's suggested parameters directly from the decision
                    # This will be either the schema (for generate_tool) or the values (for use_tool)
                    llm_decision_parameters_raw = decision.get("parameters", {})
                    llm_decision_parameters: Dict[str, Any]

                    # Defensive check for the raw parameters from LLM decision
                    if not isinstance(llm_decision_parameters_raw, dict):
                        logger.warning(f"LLM suggested 'parameters' as a non-dict type ({type(llm_decision_parameters_raw)}): {llm_decision_parameters_raw}. Defaulting to empty dict.")
                        # Attempt a specific conversion if it's a list containing a single dict, common LLM mistake
                        if isinstance(llm_decision_parameters_raw, list) and len(llm_decision_parameters_raw) == 1 and isinstance(llm_decision_parameters_raw[0], dict):
                            llm_decision_parameters = llm_decision_parameters_raw[0]
                        else:
                            llm_decision_parameters = {} # Default to empty dict
                    else:
                        llm_decision_parameters = llm_decision_parameters_raw

                    tool_definition = None # Initialize tool_definition here

                    if action_type == "generate_tool":
                        logger.info(f"Decision: Generate new tool '{tool_name}'")
                        mcp_parameters: List[ToolParameter] = []
                        if llm_decision_parameters: # Use llm_decision_parameters here as it contains the schema
                            for param_name, param_info in llm_decision_parameters.items():
                                # Infer type from LLM's structure or default to str
                                mcp_parameters.append(ToolParameter(
                                    name=param_name,
                                    type=param_info.get('type', 'str'),
                                    description=param_info.get('description', f"Parameter for {param_name}"),
                                    required=not param_info.get('optional', False)
                                ))

                        new_tool_definition_proto = MCPToolDefinition(
                            name=tool_name,
                            description=tool_description,
                            parameters=mcp_parameters,
                            code=""
                        )
                        generated_code = await self._call_llm_with_retries(
                            messages=[{"role": "system", "content": self._get_tool_generation_system_prompt(new_tool_definition_proto)}],
                            temperature=0.7
                        )

                        if not generated_code or not self._validate_generated_code(generated_code, tool_name):
                            logger.error(f"Generated code for '{tool_name}' failed validation.")
                            thought_history.append({"thought": f"Failed to generate valid code for tool '{tool_name}'."})
                            current_context["last_tool_status"] = "failed_generation"
                            continue

                        new_tool_definition_proto.code = generated_code

                        # Map MCPToolDefinition parameters (List[ToolParameter]) to the Dict[str, Any] JSON schema format for ToolCreate
                        tool_create_params_schema = {
                            "type": "object",
                            "properties": {p.name: {"type": p.type, "description": p.description} for p in new_tool_definition_proto.parameters},
                            "required": [p.name for p in new_tool_definition_proto.parameters if p.required]
                        }

                        tool_to_save = ToolCreate(
                            name=new_tool_definition_proto.name,
                            description=new_tool_definition_proto.description,
                            parameters=tool_create_params_schema,
                            code=generated_code
                        )

                        db_tool = None
                        try:
                            # Attempt to create the tool
                            db_tool = await self.tool_crud.create(db_session, tool_to_save)
                            logger.info(f"Successfully generated and saved new tool: {tool_to_save.name}")
                            await self.trace_logger.log_event("New Tool Generated & Saved", {"tool_name": tool_to_save.name})
                        except (IntegrityError, UniqueViolationError) as e:
                            # IMMEDIATELY ROLLBACK THE SESSION to clear its state
                            await db_session.rollback()
                            logger.warning(f"Tool '{tool_to_save.name}' already exists. Attempting to retrieve. Original error: {e}")
                            db_tool = await self.tool_crud.get_by_name(db_session, tool_to_save.name)
                            if db_tool:
                                logger.info(f"Retrieved existing tool '{tool_to_save.name}' after duplicate creation attempt.")
                                await self.trace_logger.log_event("Existing Tool Retrieved (Duplicate Attempt)", {"tool_name": tool_to_save.name})
                            else:
                                logger.error(f"Critical: Failed to retrieve tool '{tool_to_save.name}' after unique violation and rollback. Cannot proceed. Error: {e}")
                                final_response = {"status": "error", "message": f"A critical database error occurred for tool '{tool_to_save.name}'. Please contact support."}
                                break # Exit loop on critical error

                        # If we have a db_tool (either newly created or retrieved existing)
                        if db_tool:
                            # After generating/retrieving a tool, it's good to refresh the registry cache
                            await self.registry.get_all_tool_definitions_for_llm(db_session)
                            current_context["last_tool_status"] = "tool_generated" # Inform LLM next turn
                            thought_history.append({"thought": f"Successfully generated and registered tool '{tool_name}'."})
                            continue # Go to the next iteration to let LLM decide to use the new tool
                        else:
                            # If db_tool is still None here, it means we hit the critical error path above
                            # and final_response would have been set, so we just continue to the break.
                            continue # continue to break out of the main loop

                    elif action_type == "use_tool":
                        logger.info(f"Decision: Use existing tool '{tool_name}'")
                        db_tool = await self.registry.get_tool_by_name(db_session, tool_name)

                        if not db_tool:
                            logger.warning(f"Tool '{tool_name}' not found in registry.")
                            thought_history.append({"thought": f"Tool '{tool_name}' was suggested but not found."})
                            current_context["last_tool_status"] = "tool_not_found"
                            continue

                        # Handle the 'parameters' based on its actual type from db_tool
                        mcp_parameters_list: List[ToolParameter] = []
                        if isinstance(db_tool.parameters, dict) and "properties" in db_tool.parameters:
                            # It's a JSON schema dict, convert to List[ToolParameter]
                            for p_name, p_info in db_tool.parameters['properties'].items():
                                mcp_parameters_list.append(
                                    ToolParameter(
                                        name=p_name,
                                        type=p_info.get('type', 'str'),
                                        description=p_info.get('description', ''),
                                        required=p_name in db_tool.parameters.get('required', [])
                                    )
                                )
                        elif isinstance(db_tool.parameters, list):
                            # It's already a List[ToolParameter], assume correct structure
                            # We should ideally validate this list against ToolParameter schema here too
                            mcp_parameters_list = db_tool.parameters
                        else:
                            logger.error(f"Unexpected type for db_tool.parameters: {type(db_tool.parameters)} for tool {tool_name}")
                            final_response = {"status": "error", "message": f"Internal error: Malformed tool definition for {tool_name}. Please contact support."}
                            break # Exit loop
                        
                        tool_definition = MCPToolDefinition(
                            name=db_tool.name,
                            description=db_tool.description,
                            parameters=mcp_parameters_list, # Use the correctly parsed list
                            code=db_tool.code
                        )

                        # This should not be hit if db_tool is found and converted to tool_definition
                        if not tool_definition:
                            final_response = {"status": "error", "message": f"Could not find or generate tool: {tool_name}. Please refine your request."}
                            break

                        # --- Step 2: Extract / Confirm Parameters for current tool ---
                        extracted_tool_args, param_error_message = await self._extract_parameters_for_next_step(
                            tool_definition, # Pass the ToolDefinition directly
                            llm_decision_parameters, # LLM's provided parameters, potentially with placeholders
                            user_prompt # The original user prompt
                        )

                        if param_error_message:
                            # If parameter extraction/validation failed, add a thought and let the LLM try again.
                            self._add_thought(thought_history, "assistant", {"action": "param_extraction_failed", "tool_name": tool_name, "error": param_error_message})
                            await self.trace_logger.log_event('Parameter Validation Error Next Step', {'error': param_error_message, 'extracted_params': llm_decision_parameters})
                            logger.warning(f"Failed to extract or validate parameters for tool '{tool_name}': {param_error_message}")
                            current_context["last_tool_status"] = "failed_param_extraction"
                            continue # Continue to next iteration, letting LLM re-evaluate

                        # If parameters are valid, call the tool
                        await self.trace_logger.log_event('Parameters Validated for Next Step', extracted_tool_args)
                        await self.trace_logger.log_event('Executing Tool Call (Chaining)', {'tool_name': tool_name, 'tool_arguments': extracted_tool_args})

                        mcp_tool_call = MCPToolCall(tool_name=tool_name, tool_arguments=extracted_tool_args, call_id=str(uuid4()))

                        # --- Step 3: Execute Tool ---
                        tool_raw_output: MCPToolResponse = await self.registry.call_tool(mcp_tool_call.tool_name, mcp_tool_call.tool_arguments, mcp_tool_call.call_id)

                        # --- BEGIN: Changes related to MCPToolResponse.error_message ---
                        if tool_raw_output.status == "error":
                            # Propagate the error message from the sandbox
                            await self.trace_logger.log_event('Tool Execution Failed (Chaining)', {'tool_name': tool_name, 'error': tool_raw_output.error_message})
                            self._add_thought(thought_history, "tool_output", {
                                "tool_name": tool_name,
                                "status": "error",
                                "output": tool_raw_output.output, # Can still provide raw output for debug
                                "error_message": tool_raw_output.error_message # Use the new field
                            })
                            current_context["last_tool_status"] = "tool_execution_failed"
                            current_context["last_tool_output"] = tool_raw_output.error_message # Set error message as output for context
                            # Continue to next iteration to let LLM re-evaluate based on the error
                            continue
                        else:
                            self._add_thought(thought_history, "tool_output", {
                                "tool_name": tool_name,
                                "status": "success",
                                "output": tool_raw_output.output
                            })
                            current_context["last_tool_status"] = "tool_executed_successfully"
                            current_context["last_tool_output"] = tool_raw_output.output # Set successful output for context
                            # Continue to next iteration to process tool output
                            continue
                        # --- END: Changes related to MCPToolResponse.error_message ---

                else:
                    logger.warning(f"LLM returned unknown action type: {action_type}. Falling back to conversational.")
                    final_response = {"status": "info", "message": "I'm not sure how to proceed with that. Could you clarify?"}
                    break

            except RuntimeError as e:
                logger.error(f"Tool execution failed during chaining for prompt '{user_prompt}' (iteration {iteration}): {e}", exc_info=True)
                error_tool_name = tool_name if 'tool_name' in locals() else "unknown_tool"
                current_context["last_tool_name"] = error_tool_name
                current_context["last_tool_error"] = str(e)
                current_context["last_tool_status"] = "failed"
                self._add_thought(thought_history, "tool", {"tool_name": error_tool_name, "error": str(e), "status": "failed"})
                await self.trace_logger.log_event("Chaining Runtime Error", {"error": str(e), "prompt": user_prompt, "iteration": iteration})
                continue
            except (APIStatusError, APIConnectionError) as e:
                logger.error(f"LLM API communication error during chaining for prompt '{user_prompt}': {e}", exc_info=True)
                await self.trace_logger.log_event("Chaining LLM API Error", {"error": str(e), "prompt": user_prompt, "iteration": iteration})
                final_response = {"status": "error", "message": "I'm having trouble communicating with the AI services during this multi-step process. Please try again shortly."}
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred during chaining for prompt '{user_prompt}' (iteration {iteration}): {e}", exc_info=True)
                await self.trace_logger.log_event("Chaining Unexpected Error", {"error": str(e), "prompt": user_prompt, "iteration": iteration})
                final_response = {"status": "error", "message": f"An unexpected error occurred while processing your request: {e}. Please contact support."}
                break

        if final_response:
            return final_response
        else:
            logger.warning(f"Max iterations ({max_iterations}) reached without a final response.")
            return {"status": "info", "message": "I've reached the maximum number of steps for this request. Could you provide more specific instructions or rephrase?"}

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
        1. GENERATE a new tool.
        2. USE an existing tool.
        3. RESPOND directly to the user.

        Your decision must be returned as a JSON object, strictly following one of these formats:

        --- Option 1: GENERATE a new tool ---
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

        --- Option 2: USE an existing tool ---
        {{
            "action": "use_tool",
            "tool_name": "name_of_existing_tool",
            "parameters": {{
                "param_name_1": "extracted_value_1",
                "param_name_2": extracted_value_2
            }}
        }}
        (Use this if an available tool can help. Extract ALL necessary parameters from the user's original prompt or previous tool outputs (from current_context).
        IMPORTANT: DO NOT use placeholder values like "extracted_value_1". Provide the actual value.
        If a parameter's value cannot be determined, OMIT that parameter from the 'parameters' object.)

        --- Option 3: RESPOND directly to the user ---
        {{
            "action": "respond",
            "response_message": "A clear and concise message to the user, indicating the task is complete, or asking for clarification, or stating that the task cannot be fulfilled."
        }}
        (Use this if the task is complete, if you need more information from the user, or if you cannot proceed with tools.)

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
        Strictly output only the JSON object for your chosen action.
        """
        return [{"role": "system", "content": system_message}]

    def _get_python_type_from_str(self, type_str: str) -> Type:
        """Converts string type names to Python types."""
        type_map = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "boolean": bool,
            "bool": bool,
            "list": List[Any],
            "array": List[Any],
            "dict": Dict[str, Any],
            "object": Dict[str, Any],
        }
        py_type = type_map.get(type_str.lower())
        if py_type is None:
            logger.warning(f"Unknown parameter type '{type_str}'. Defaulting to str.")
            return str # Fallback to string if type is unknown
        return py_type

    async def _extract_parameters_for_next_step(
        self,
        tool_definition: MCPToolDefinition, # Changed to accept ToolDefinition
        llm_extracted_params: Dict[str, Any],
        user_prompt: str # The original user prompt
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Validates and extracts parameters for the next tool call.
        It first attempts to validate what the LLM provided.
        If validation fails due to placeholders or missing required fields,
        it suggests feedback for the LLM to try again.
        Returns validated parameters and an error message if extraction/validation fails.
        """
        if not tool_definition.parameters: # Use tool_definition.parameters
            return {}, None

        # Dynamic Pydantic model for strict validation
        dynamic_model_fields = {}
        for param in tool_definition.parameters: # Iterate through ToolParameter objects
            py_type = self._get_python_type_from_str(param.type)

            if param.required:
                dynamic_model_fields[param.name] = (py_type, Field(..., description=param.description))
            else:
                dynamic_model_fields[param.name] = (Optional[py_type], Field(None, description=param.description))

        # Use a unique name for the dynamic model to avoid conflicts if called repeatedly
        # Include optional/required in hash for uniqueness too
        model_name_hash = abs(hash(frozenset((p.name, p.type, p.description, p.required) for p in tool_definition.parameters)))
        DynamicParamSchema = create_model(
            f'DynamicParamSchema_{model_name_hash}',
            __base__=BaseModel,
            **dynamic_model_fields
        )

        try:
            # Validate parameters directly from LLM's extraction
            validated_params = DynamicParamSchema.model_validate(llm_extracted_params)
            await self.trace_logger.log_event("Parameters Validated for Next Step", validated_params.model_dump())
            return validated_params.model_dump(exclude_unset=True), None # Success, no error
        except ValidationError as e:
            await self.trace_logger.log_event("Parameter Validation Error Next Step", {"errors": e.errors(), "llm_extracted": llm_extracted_params})

            error_details = e.errors()
            for error in error_details:
                param_name = error['loc'][0] # e.g., 'number'
                error_type = error['type']
                error_input = error.get('input') # The value that caused the error

                # Specific check for LLM placeholder issue (e.g., 'extracted_value_1')
                if isinstance(error_input, str) and ('extracted_value' in error_input.lower() or 'placeholder' in error_input.lower()):
                    if error_type == 'int_parsing':
                        return {}, f"Parameter '{param_name}' requires an integer, but you provided a placeholder '{error_input}'. Please extract the concrete integer value from the user's request (e.g., '5') or indicate if you cannot determine it."
                    elif error_type == 'string_type':
                        return {}, f"Parameter '{param_name}' requires a string, but you provided a placeholder '{error_input}'. Please extract the concrete string value from the user's request or indicate if you cannot determine it."
                    # Add more types if needed
                    return {}, f"Parameter '{param_name}' has an invalid value: '{error_input}'. It seems to be a placeholder. Please provide the actual value."
                elif error_type == 'missing':
                    # This happens if a required parameter is not provided at all by the LLM
                    return {}, f"Missing required parameter '{param_name}'. Please ensure you extract this value from the user's prompt or ask the user for it if it's missing."
                # Fallback for any other validation error not specifically handled
            return {}, f"Parameter validation failed: {e.errors()}"
        except Exception as e:
            logger.exception(f"An unexpected error occurred during parameter extraction for tool {tool_definition.name}.")
            return {}, f"An unexpected error occurred during parameter extraction: {e.__class__.__name__}: {e}"

    async def _extract_initial_prompt_parameters(self, user_prompt: str, tool_parameters: List[ToolParameter]) -> Optional[Dict[str, Any]]:
        """
        Extracts parameter values directly from the user_prompt using an LLM.
        This is for initial, single-step parameter extraction.
        """
        if not tool_parameters:
            return {}

        params_description = []
        param_json_schema = {}
        dynamic_model_fields = {}

        for p in tool_parameters:
            optional_str = " (optional)" if not p.required else ""
            params_description.append(f"- '{p.name}' ({p.type}){optional_str}: {p.description or 'No specific description provided.'}")

            json_type = p.type.lower()
            if json_type == 'str': json_type = 'string'
            elif json_type == 'int': json_type = 'integer'
            elif json_type == 'float': json_type = 'number'
            elif json_type == 'bool': json_type = 'boolean'
            elif json_type == 'list': json_type = 'array'
            elif json_type == 'dict': json_type = 'object'

            param_json_schema[p.name] = {"type": json_type, "description": p.description or f"Value for {p.name}"}

            param_python_type = self._get_python_type_from_str(p.type)

            if not p.required:
                dynamic_model_fields[p.name] = (Optional[param_python_type], None)
            else:
                dynamic_model_fields[p.name] = (param_python_type, ...)

        model_name_hash = abs(hash(frozenset((p.name, p.type, p.description, p.required) for p in tool_parameters)))
        DynamicParamSchema = create_model(
            f'DynamicParamSchemaInitial_{model_name_hash}',
            __base__=BaseModel,
            **dynamic_model_fields
        )

        required_params_for_schema = [p.name for p in tool_parameters if p.required]
        json_schema_definition = {
            "type": "object",
            "properties": param_json_schema
        }
        if required_params_for_schema:
            json_schema_definition["required"] = required_params_for_schema

        system_prompt = f"""
        You are an expert at extracting information from natural language and adhering to strict JSON schemas.
        Given a user's prompt and the JSON schema for the required parameters of a tool, your task is to extract the values for those parameters.
        Return a JSON object where keys are parameter names and values are the extracted values.
        You MUST adhere strictly to the provided JSON schema, including data types and required fields.
        If a parameter is not explicitly mentioned or cannot be clearly inferred from the prompt, and it is a REQUIRED field according to the schema, you MUST NOT include it in the output. If it is OPTIONAL, you can omit it.
        Boolean values MUST be `true` or `false` (lowercase). Numbers MUST be valid JSON numbers. Strings MUST be valid JSON strings.

        User Prompt: "{user_prompt}"

        Tool Parameters JSON Schema:
        ```json
        {json.dumps(json_schema_definition, indent=2)}
        ```

        Your response MUST be a valid JSON object matching the schema for the parameters.
        Example for a tool with 'query' (string, required) and 'limit' (integer, optional) parameters:
        User Prompt: "Find latest news with max 10 results"
        Output: {{ "query": "latest news", "limit": 10 }}

        User Prompt: "Just find latest news"
        Output: {{ "query": "latest news" }}

        User Prompt: "Search for weather in London"
        Output: {{ "location": "London" }}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        await self.trace_logger.log_event("Parameter Extraction Prompt (Initial)", {"tool_parameters": [p.model_dump() for p in tool_parameters], "user_prompt": user_prompt})
        logger.info(f"Extracting parameters for tool with params: {[p.name for p in tool_parameters]} from prompt: '{user_prompt}' using LLM.")

        try:
            raw_llm_response_content = await self._call_llm_with_retries(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            await self.trace_logger.log_event("Raw LLM Parameter Extraction Response (Initial)", {"response": raw_llm_response_content})

            extracted_params_validated = DynamicParamSchema.model_validate_json(raw_llm_response_content)
            extracted_params = extracted_params_validated.model_dump()

            await self.trace_logger.log_event("Extracted Parameters (Validated, Initial)", extracted_params)
            return extracted_params

        except ValidationError as e:
            logger.error(f"Pydantic validation error during initial parameter extraction: {e.errors()}. Raw LLM response: {raw_llm_response_content}", exc_info=True)
            await self.trace_logger.log_event("Parameter Extraction Validation Error (Initial)", {"error": str(e), "raw_response": raw_llm_response_content})
            return None
        except Exception as e:
            logger.error(f"Unexpected error during initial parameter extraction LLM call: {e}", exc_info=True)
            await self.trace_logger.log_event("Parameter Extraction Unexpected Error (Initial)", {"error": str(e), "prompt": user_prompt})
            return None

    def _validate_generated_code(self, code: str, expected_function_name: str) -> bool:
        """
        Performs robust validation of the generated Python code for security and correctness.
        """
        original_code = code # Keep the original code for detailed logging

        # --- Strip Markdown code block fences ---
        stripped_code = code.strip()
        if stripped_code.startswith("```python"):
            stripped_code = stripped_code[len("```python"):].strip()
            if stripped_code.endswith("```"):
                stripped_code = stripped_code[:-len("```")].strip()
        elif stripped_code.startswith("```"):
            stripped_code = stripped_code[len("```"):].strip()
            if stripped_code.endswith("```"):
                stripped_code = stripped_code[:-len("```")].strip()
        code_to_parse = stripped_code

        logger.debug(f"Attempting to validate code:\n{code_to_parse}")

        try:
            tree = ast.parse(code_to_parse)

            # --- Check 1: Basic Syntax Validity (handled by ast.parse successfully) ---

            # --- Check 2: Expected Async Function Signature ---
            function_found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef) and node.name == expected_function_name:
                    function_found = True
                    break
            if not function_found:
                logger.warning(f"Validation failed: Code does not contain the expected async function '{expected_function_name}'.")
                return False

            # --- Check 3: Forbidden Imports ---
            forbidden_modules = {
                'os', 'sys', 'subprocess', 'shutil', 'threading', 'multiprocessing',
                'socket', 'http', 'ftplib', 'smtplib', 'imaplib', 'poplib',
                'paramiko', 'scp', 'pexpect', 'webbrowser', 'pickle', 'cPickle',
                'yaml', # YAML can be dangerous if not loaded safely
                'json', # Can be dangerous if used with untrusted data for `json.loads` without proper validation
                'importlib', 'inspect', # Added for deeper security
                'distutils', 'setuptools', 'pip', # Installation/system altering
                'ctypes', 'sysconfig', 'gc', 'resource', # Low-level system/memory access
            }

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imported_names = []
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            imported_names.append(n.name.split('.')[0])
                    else: # ast.ImportFrom
                        if node.module:
                            imported_names.append(node.module.split('.')[0])
                        for n in node.names:
                            imported_names.append(n.name) # Check specific names like 'system' from 'os'

                    for name in imported_names:
                        if name in forbidden_modules:
                            logger.error(f"Validation failed: Forbidden module import detected: '{name}'.")
                            return False

            # --- Check 4: Forbidden Function Calls / Expressions ---
            forbidden_builtins = {'eval', 'exec', 'open', 'input', 'breakpoint', 'compile', 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'dir'} # Added globals, locals, dir
            dangerous_attributes = {
                '__import__', '__subclasses__', '__globals__', '__closure__', '__bases__', '__mro__',
                'system', 'popen', 'read', 'write', 'delete', 'exit', 'quit',
                '__dict__', '__class__', '__builtins__', # More introspection prevention
                '__reduce__', '__reduce_ex__', # Deserialization attacks
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Direct calls to forbidden built-ins (e.g., eval())
                    if isinstance(node.func, ast.Name) and node.func.id in forbidden_builtins:
                        logger.error(f"Validation failed: Forbidden built-in function call detected: '{node.func.id}'.")
                        return False
                    # Calls to dangerous attributes (e.g., os.system())
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in dangerous_attributes:
                            logger.error(f"Validation failed: Call to dangerous attribute as function detected: '{node.func.attr}'.")
                            return False
                        # Advanced check: if calling a method from a forbidden module (e.g., sys.exit)
                        if isinstance(node.func.value, ast.Name) and node.func.value.id in forbidden_modules:
                            logger.error(f"Validation failed: Call to method of forbidden module detected: '{node.func.value.id}.{node.func.attr}'.")
                            return False

                # Accessing dangerous attributes (e.g., obj.__subclasses__)
                if isinstance(node, ast.Attribute) and node.attr in dangerous_attributes:
                    logger.error(f"Validation failed: Access to dangerous attribute detected: '{node.attr}'.")
                    return False

            logger.info(f"Generated code for '{expected_function_name}' passed all static validations.")
            return True

        except SyntaxError as e:
            logger.error(f"Code validation failed due to SyntaxError: {e}. Code snippet: '{code_to_parse[:200]}...'")
            logger.debug(f"Full original code (SyntaxError):\n{original_code}")
            return False
        except Exception as e:
            logger.error(f"Code validation failed due to unexpected error: {e}. Code snippet: '{code_to_parse[:200]}...'")
            logger.debug(f"Full original code (Unexpected error):\n{original_code}")
            return False

    def _get_tool_generation_system_prompt(self, tool_definition: MCPToolDefinition) -> str:
        """
        Constructs the system prompt for the LLM to generate tool code.
        """
        parameters_schema = {}
        for param in tool_definition.parameters:
            parameters_schema[param.name] = {
                "type": param.type,
                "description": param.description,
                "optional": not param.required
            }

        # Dynamically create the list of parameters for the example structure
        param_args = ", ".join([f"{param.name}: {param.type}" for param in tool_definition.parameters])
        if not param_args and tool_definition.parameters: # Handle cases where parameters are present but maybe malformed
            param_args = ", ".join([f"{p_name}: Any" for p_name in parameters_schema.keys()])
        elif not tool_definition.parameters: # No parameters
            param_args = ""

        forbidden_list_str = ", ".join(AgentOrchestratorService._get_forbidden_list())

        return f"""
        You are an expert Python developer tasked with writing a self-contained, asynchronous Python function.
        This function will be used as a tool in an AI agent system.

        Here is the definition of the tool you need to implement:
        Name: {tool_definition.name}
        Description: {tool_definition.description}
        Parameters: {json.dumps(parameters_schema, indent=2)}

        Constraints:
        - The function MUST be an `async def` function with the name `{tool_definition.name}`.
        - The function MUST accept all its parameters as direct arguments, matching the names and types specified.
        - The function MUST NOT import any modules or use any built-in functions or attributes that could pose a security risk or allow arbitrary code execution. Specifically, do NOT import or NOT use: {forbidden_list_str}.
        - The function should perform the task described and return its result. The return type should be a JSON-serializable object (dict, list, string, number, boolean).
        - If the tool requires network access, use `aiohttp` or `requests` (if synchronous operations are explicitly allowed and managed). `aiohttp` is preferred for async contexts.
        - Provide ONLY the Python function code. Do NOT include example usage, comments outside the function body, or any other surrounding text (e.g., no markdown fences like ```python```).
        - Your response MUST contain ONLY the Python function code, enclosed within a single markdown code block (```python ... ```). Do NOT include any other text, explanation, or example usage outside this code block.

        Example Structure:
        ```python
        async def {tool_definition.name}({param_args}) -> dict:
            # Your implementation here
            # For demonstration, a simple placeholder or an example using aiohttp if applicable:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                response = await session.get("[https://api.example.com](https://api.example.com)")
                data = await response.json()
            return {{"result": data}}
        ```
        """

    # Helper to provide forbidden list to the LLM (optional but good for transparency)
    @staticmethod
    def _get_forbidden_list():
        # This should ideally match the lists in _validate_generated_code
        forbidden_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'threading', 'multiprocessing',
            'socket', 'http', 'ftplib', 'smtplib', 'imaplib', 'poplib',
            'paramiko', 'scp', 'pexpect', 'webbrowser', 'pickle', 'cPickle',
            'yaml', 'json', 'importlib', 'inspect',
            'distutils', 'setuptools', 'pip',
            'ctypes', 'sysconfig', 'gc', 'resource',
        }
        forbidden_builtins = {'eval', 'exec', 'open', 'input', 'breakpoint', 'compile', 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'dir'}
        dangerous_attributes = {
            '__import__', '__subclasses__', '__globals__', '__closure__', '__bases__', '__mro__',
            'system', 'popen', 'read', 'write', 'delete', 'exit', 'quit',
            '__dict__', '__class__', '__builtins__',
            '__reduce__', '__reduce__', # Deserialization attacks
        }

        # Combine all and sort for consistent prompt generation
        return sorted(list(forbidden_modules.union(forbidden_builtins).union(dangerous_attributes)))

    def _add_thought(self, thought_history: List[Dict[str, Any]], role: str, content: Dict[str, Any]):
        """Adds a structured thought/action to the history."""
        thought_history.append({"role": role, "content": content})


openai_client_instance = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Consider removing these debug prints in production, or moving them to proper logging
logger.debug(f"🪲DEBUG: Type of trace_logger_service before orchestrator init: {type(trace_logger_service)}")
logger.debug(f"🪲DEBUG: Value of trace_logger_service before orchestrator init: {trace_logger_service}")


# Instantiate the orchestrator for easy import
agent_orchestrator_service = AgentOrchestratorService(
    analyzer=tool_analyzer_service,
    generator=tool_generator_service,
    registry=tool_registry_service,
    sandbox=sandbox_service,
    tool_crud=tool_crud ,
    trace_logger=trace_logger_service,
    openai_client=openai_client_instance
)