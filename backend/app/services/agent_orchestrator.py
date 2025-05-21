# backend/app/services/agent_orchestrator.py
import ast
import logging
import json
import asyncio
from typing import Any, Dict, Optional, List, Type
from pydantic import ValidationError, BaseModel, Field, create_model
from openai import OpenAI, AsyncOpenAI, APIStatusError, APIConnectionError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from app.services.tool_analyzer import tool_analyzer_service, ToolAnalysisResult
from app.services.tool_generator import tool_generator_service
from app.services.tool_registry import tool_registry_service
from app.services.sandbox_service import sandbox_service
from app.crud.tool import tool_crud
from app.schemas.tool import MCPToolDefinition, MCPToolParameter, MCPToolCall, MCPToolResponse
from app.utils.logger import trace_logger_service

from app.core.config import settings

logger = logging.getLogger(__name__)

# Define a Pydantic schema for the LLM's parameter extraction output (can be reused)
class ExtractedParameters(BaseModel):
    parameters: Dict[str, Any] = Field(
        ...,
        description="A dictionary where keys are parameter names and values are the extracted values.",
    )

class AgentOrchestrator:
    def __init__(self):
        self.analyzer = tool_analyzer_service
        self.generator = tool_generator_service
        self.registry = tool_registry_service
        self.sandbox = sandbox_service
        self.tool_crud = tool_crud
        self.trace_logger = trace_logger_service
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

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
        self.trace_logger.log_event("Processing User Prompt (Start Chaining)", {"user_prompt": user_prompt})
        logger.info(f"Orchestrator processing prompt for chaining: '{user_prompt}'")

        current_context = {"user_prompt": user_prompt}
        thought_history = [] # To track previous LLM thoughts and tool actions
        final_response = None

        # Ensure the tool registry is populated with current tools
        # This call will also populate the internal cache used by get_all_tool_definitions_list
        await self.registry.get_all_tool_definitions_for_llm(db_session)

        for iteration in range(max_iterations):
            self.trace_logger.log_event("Chaining Iteration Start", {"iteration": iteration, "context": current_context})
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
                    self.trace_logger.log_event("LLM Decision", decision)
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
                    llm_suggested_params = decision.get("parameters", {}) # Parameters extracted by LLM decision

                    # --- Determine Tool Definition (new or existing) ---
                    tool_definition = None
                    if action_type == "generate_tool":
                        logger.info(f"Decision: Generate new tool '{tool_name}'")
                        mcp_parameters: List[MCPToolParameter] = []
                        if llm_suggested_params:
                            for param_name, param_info in llm_suggested_params.items():
                                # Infer type from LLM's structure or default to str
                                mcp_parameters.append(MCPToolParameter(
                                    name=param_name,
                                    type=param_info.get('type', 'str'),
                                    description=param_info.get('description', f"Parameter for {param_name}"),
                                    optional=param_info.get('optional', False)
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
                            continue # Try next iteration or change strategy

                        new_tool_definition_proto.code = generated_code
                        db_tool = await self.tool_crud.create(db_session, obj_in=new_tool_definition_proto)
                        tool_definition = db_tool
                        self.trace_logger.log_event("New Tool Generated & Saved", {"tool_name": tool_definition.name})
                        await self.registry.get_all_tool_definitions_for_llm(db_session) # Refresh registry cache

                    elif action_type == "use_tool":
                        logger.info(f"Decision: Use existing tool '{tool_name}'")
                        # Use the new method to get raw MCPToolDefinition
                        tool_definition = await self.registry.get_tool_by_name(db_session, tool_name)
                        if not tool_definition:
                            logger.warning(f"Tool '{tool_name}' not found in registry.")
                            thought_history.append({"thought": f"Tool '{tool_name}' was suggested but not found."})
                            current_context["last_tool_status"] = "tool_not_found"
                            continue # Try next iteration or change strategy

                    if not tool_definition:
                        final_response = {"status": "error", "message": f"Could not find or generate tool: {tool_name}. Please refine your request."}
                        break

                    # --- Step 2: Extract / Confirm Parameters for current tool ---
                    extracted_tool_args = await self._extract_parameters_for_next_step(
                        user_prompt, # Original prompt for broader context
                        tool_definition.parameters,
                        llm_suggested_params, # Parameters from LLM's decision
                        current_context # Context from previous tool output/states
                    )
                    
                    if extracted_tool_args is None:
                        logger.warning(f"Failed to extract or validate parameters for tool '{tool_definition.name}'.")
                        thought_history.append({"thought": f"Failed to extract valid parameters for tool '{tool_definition.name}'."})
                        current_context["last_tool_status"] = "failed_param_extraction"
                        continue # Try next iteration or change strategy

                    # Check for missing required parameters (same logic as before)
                    missing_required_params = [p.name for p in tool_definition.parameters if not p.optional and p.name not in extracted_tool_args]
                    if missing_required_params:
                        logger.warning(f"Missing required parameters for tool '{tool_definition.name}': {missing_required_params}")
                        thought_history.append({"thought": f"Missing required parameters for '{tool_definition.name}': {', '.join(missing_required_params)}. Need to ask user."})
                        final_response = {"status": "info", "message": f"I need more information to use the '{tool_definition.name}' tool. Please provide values for: {', '.join(missing_required_params)}."}
                        break # Exit loop for user input

                    mcp_tool_call = MCPToolCall(tool_name=tool_definition.name, tool_arguments=extracted_tool_args) # Changed tool_args to tool_arguments as per MCPToolCall schema
                    self.trace_logger.log_event("Executing Tool Call (Chaining)", mcp_tool_call.model_dump())

                    # --- Step 3: Execute Tool ---
                    # Use the tool_registry_service.call_tool for execution
                    tool_raw_output = await self.registry.call_tool(
                        tool_name=tool_definition.name,
                        tool_arguments=extracted_tool_args,
                        tool_call_id=mcp_tool_call.call_id # Pass the generated call_id
                    )
                    
                    # Update current_context with the output
                    current_context["last_tool_name"] = tool_definition.name
                    current_context["last_tool_output"] = tool_raw_output
                    current_context["last_tool_status"] = "success" # Assuming call_tool raises on failure
                    thought_history.append({"tool": tool_definition.name, "output": tool_raw_output, "status": "success"})
                    logger.info(f"Tool '{tool_definition.name}' executed successfully. Output: {tool_raw_output}")
                    # Loop back to decision making for next step

                else:
                    logger.warning(f"LLM returned unknown action type: {action_type}. Falling back to conversational.")
                    final_response = {"status": "info", "message": "I'm not sure how to proceed with that. Could you clarify?"}
                    break # Exit loop

            except RuntimeError as e: # Catch specific tool execution failures
                logger.error(f"Tool execution failed during chaining for prompt '{user_prompt}' (iteration {iteration}): {e}", exc_info=True)
                current_context["last_tool_name"] = tool_name
                current_context["last_tool_error"] = str(e)
                current_context["last_tool_status"] = "failed"
                thought_history.append({"tool": tool_name, "error": str(e), "status": "failed"})
                # Decide if a new attempt should be made or if this is unrecoverable
                # For now, continue to let LLM decide next step based on failure
                continue 
            except (APIStatusError, APIConnectionError) as e:
                logger.error(f"LLM API communication error during chaining for prompt '{user_prompt}': {e}", exc_info=True)
                self.trace_logger.log_event("Chaining LLM API Error", {"error": str(e), "prompt": user_prompt, "iteration": iteration})
                final_response = {"status": "error", "message": "I'm having trouble communicating with the AI services during this multi-step process. Please try again shortly."}
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred during chaining for prompt '{user_prompt}' (iteration {iteration}): {e}", exc_info=True)
                self.trace_logger.log_event("Chaining Unexpected Error", {"error": str(e), "prompt": user_prompt, "iteration": iteration})
                final_response = {"status": "error", "message": f"An unexpected error occurred while processing your request: {e}. Please contact support."}
                break

        if final_response:
            return final_response
        else:
            # If loop finishes without explicit "respond" or error, might be max_iterations reached
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
        available_tools_for_llm_consideration = [
            {"name": tool.name, "description": tool.description, "parameters": [p.model_dump() for p in tool.parameters]}
            for tool in available_tools_raw
        ]
        
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
        (Use this if an available tool can help. Extract ALL necessary parameters from the user's original prompt or previous tool outputs (from current_context). If a required parameter is missing and cannot be inferred, do NOT choose this action; instead consider asking the user or generating a tool if that's more appropriate.)

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

    async def _extract_parameters_for_next_step(
        self,
        user_prompt: str,
        tool_parameters: List[MCPToolParameter],
        llm_suggested_params: Dict[str, Any],
        current_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts or confirms parameter values for a tool during a chaining sequence.
        Prioritizes LLM-suggested parameters, then attempts to infer from context.
        """
        if not tool_parameters:
            return {}

        extracted_params = llm_suggested_params.copy()

        # Try to fill in missing required parameters from `current_context`
        # This is a basic example; more sophisticated logic might be needed
        for p in tool_parameters:
            if p.name not in extracted_params and not p.optional:
                # Attempt to find value in previous tool output if it's a dict
                if isinstance(current_context.get("last_tool_output"), dict) and p.name in current_context["last_tool_output"]:
                    extracted_params[p.name] = current_context["last_tool_output"][p.name]
                    logger.info(f"Filled missing required parameter '{p.name}' from last tool output.")
                # You could add more sophisticated logic here, e.g., using another LLM call
                # with a more targeted prompt for parameter extraction given context.
                # For now, if a required param is missing, it will fail validation below.

        # Dynamic Pydantic model for strict validation
        dynamic_model_fields = {}
        for p in tool_parameters:
            python_type_map = {
                'str': str, 'int': int, 'float': float, 'bool': bool,
                'list': List[Any], 'dict': Dict[str, Any]
            }
            param_python_type = python_type_map.get(p.type.lower(), str)
            if p.optional:
                dynamic_model_fields[p.name] = (Optional[param_python_type], None)
            else:
                dynamic_model_fields[p.name] = (param_python_type, ...)

        DynamicParamSchema = create_model(
            f'DynamicParamSchema_{len(tool_parameters)}_{abs(hash(frozenset(p.name for p in tool_parameters)))}',
            __base__=BaseModel,
            **dynamic_model_fields
        )

        try:
            # Validate what we've gathered so far using the dynamic Pydantic model
            validated_params = DynamicParamSchema.model_validate(extracted_params)
            self.trace_logger.log_event("Parameters Validated for Next Step", validated_params.model_dump())
            return validated_params.model_dump()
        except ValidationError as e:
            logger.error(f"Validation error for parameters in next step: {e.errors()}. Extracted: {extracted_params}", exc_info=True)
            self.trace_logger.log_event("Parameter Validation Error Next Step", {"error": str(e), "extracted_params": extracted_params})
            return None

    # Renamed the original function for clarity
    async def _extract_initial_prompt_parameters(self, user_prompt: str, tool_parameters: List[MCPToolParameter]) -> Optional[Dict[str, Any]]:
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
            optional_str = " (optional)" if p.optional else ""
            params_description.append(f"- '{p.name}' ({p.type}){optional_str}: {p.description or 'No specific description provided.'}")

            json_type = p.type.lower()
            if json_type == 'str': json_type = 'string'
            elif json_type == 'int': json_type = 'integer'
            elif json_type == 'float': json_type = 'number'
            elif json_type == 'bool': json_type = 'boolean'

            param_json_schema[p.name] = {"type": json_type, "description": p.description or f"Value for {p.name}"}

            python_type_map = {
                'str': str, 'int': int, 'float': float, 'bool': bool,
                'list': List[Any], 'dict': Dict[str, Any]
            }
            param_python_type = python_type_map.get(p.type.lower(), str)

            if p.optional:
                dynamic_model_fields[p.name] = (Optional[param_python_type], None)
            else:
                dynamic_model_fields[p.name] = (param_python_type, ...)

        DynamicParamSchema = create_model(
            f'DynamicParamSchema_{len(tool_parameters)}_{abs(hash(frozenset(p.name for p in tool_parameters)))}',
            __base__=BaseModel,
            **dynamic_model_fields
        )

        required_params_for_schema = [p.name for p in tool_parameters if not p.optional]
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

        self.trace_logger.log_event("Parameter Extraction Prompt (Initial)", {"tool_parameters": [p.model_dump() for p in tool_parameters], "user_prompt": user_prompt})
        logger.info(f"Extracting parameters for tool with params: {[p.name for p in tool_parameters]} from prompt: '{user_prompt}' using LLM.")

        try:
            raw_llm_response_content = await self._call_llm_with_retries(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            self.trace_logger.log_event("Raw LLM Parameter Extraction Response (Initial)", {"response": raw_llm_response_content})

            extracted_params_validated = DynamicParamSchema.model_validate_json(raw_llm_response_content)
            extracted_params = extracted_params_validated.model_dump()
            
            self.trace_logger.log_event("Extracted Parameters (Validated, Initial)", extracted_params)
            return extracted_params

        except ValidationError as e:
            logger.error(f"Pydantic validation error during initial parameter extraction: {e.errors()}. Raw LLM response: {raw_llm_response_content}", exc_info=True)
            self.trace_logger.log_event("Parameter Extraction Validation Error (Initial)", {"error": str(e), "raw_response": raw_llm_response_content})
            return None
        except Exception as e:
            logger.error(f"Unexpected error during initial parameter extraction LLM call: {e}", exc_info=True)
            self.trace_logger.log_event("Parameter Extraction Unexpected Error (Initial)", {"error": str(e), "prompt": user_prompt})
            return None


    def _validate_generated_code(self, code: str, expected_function_name: str) -> bool:
        """
        Performs robust validation of the generated Python code for security and correctness.
        Checks for:
        1. Syntax validity.
        2. Presence of the expected async function signature.
        3. Forbidden imports (e.g., 'os', 'subprocess').
        4. Forbidden function calls (e.g., 'eval', 'exec', 'open').
        """
        try:
            tree = ast.parse(code) # Parse the code into an Abstract Syntax Tree

            # --- Check 1: Basic Syntax Validity (handled by ast.parse) ---

            # --- Check 2: Expected Function Signature ---
            function_found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef) and node.name == expected_function_name:
                    function_found = True
                    break
            if not function_found:
                logger.warning(f"Code does not contain the expected async function '{expected_function_name}'.")
                return False

            # --- Check 3: Forbidden Imports ---
            forbidden_modules = {
                'os', 'sys', 'subprocess', 'shutil', 'threading', 'multiprocessing',
                'socket', 'http', 'ftplib', 'smtplib', 'imaplib', 'poplib',
                'paramiko', 'scp', 'pexpect', 'webbrowser',
            }

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imported_names = []
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            imported_names.append(n.name.split('.')[0])
                    else:
                        if node.module:
                            imported_names.append(node.module.split('.')[0])

                    for name in imported_names:
                        if name in forbidden_modules:
                            logger.error(f"Generated code attempts to import forbidden module: '{name}'.")
                            return False

            # --- Check 4: Forbidden Function Calls / Expressions ---
            forbidden_builtins = {'eval', 'exec', 'open', 'input', 'breakpoint', 'compile'}
            dangerous_attributes = {'__import__', '__subclasses__', '__globals__', '__closure__', '__bases__', '__mro__', 'system', 'popen', 'read', 'write', 'delete'}

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in forbidden_builtins:
                        logger.error(f"Generated code attempts to call forbidden built-in function: '{node.func.id}'.")
                        return False
                    if isinstance(node.func, ast.Attribute) and node.func.attr in dangerous_attributes:
                         logger.error(f"Generated code attempts to call dangerous attribute as function: '{node.func.attr}'.")
                         return False

                if isinstance(node, ast.Attribute) and node.attr in dangerous_attributes:
                    logger.error(f"Generated code attempts to access dangerous attribute: '{node.attr}'.")
                    return False

            logger.info(f"Generated code for '{expected_function_name}' passed all static validations.")
            return True

        except SyntaxError as e:
            logger.error(f"Generated code has a syntax error: {e}", exc_info=True)
            self.trace_logger.log_event("Code Validation Syntax Error", {"error": str(e), "code_snippet": code[:500]})
            return False
        except Exception as e:
            logger.error(f"Unexpected error during code validation: {e}", exc_info=True)
            self.trace_logger.log_event("Code Validation Unexpected Error", {"error": str(e), "code_snippet": code[:500]})
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
                "optional": param.optional
            }
        
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
        - The function MUST NOT import any modules other than those explicitly allowed by the sandbox (e.g., `requests`, `aiohttp`, `beautifulsoup4`, `numpy`, `pandas`, `scipy`, `matplotlib`, `google.cloud.storage`). Do NOT import `os`, `sys`, `subprocess`, `eval`, `exec`, `open`, etc.
        - The function should perform the task described and return its result. The return type should be a JSON-serializable object (dict, list, string, number, boolean).
        - If the tool requires network access, use `aiohttp` or `requests` (if sync operations are allowed, though `aiohttp` is preferred for async contexts).
        - Provide only the function code. Do NOT include example usage, comments outside the function body, or any other surrounding text.

        Example Structure:
        ```python
        async def {tool_definition.name}(param1: str, param2: int) -> dict:
            # Your implementation here
            import aiohttp
            async with aiohttp.ClientSession() as session:
                response = await session.get("[https://api.example.com](https://api.example.com)")
                data = await response.json()
            return {{"result": data}}
        ```
        """


# Instantiate the orchestrator for easy import
agent_orchestrator_service = AgentOrchestrator()