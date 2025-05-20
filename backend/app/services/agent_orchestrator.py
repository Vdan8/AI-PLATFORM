# backend/app/services/agent_orchestrator.py
import ast
import logging
import json
import asyncio
from typing import Any, Dict, Optional, List, Type
from pydantic import ValidationError, BaseModel, Field, create_model # Added create_model for dynamic Pydantic schema
from openai import OpenAI, AsyncOpenAI, APIStatusError, APIConnectionError # Explicitly import exceptions
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type # For robust retries

from app.services.tool_analyzer import tool_analyzer_service, ToolAnalysisResult
from app.services.tool_generator import tool_generator_service
from app.services.tool_registry import tool_registry_service
from app.services.sandbox_service import sandbox_service
from app.crud.tool import tool_crud
from app.schemas.tool import MCPToolDefinition, MCPToolParameter, MCPToolCall, MCPToolResponse
from app.utils.logger import trace_logger_service # Using the consolidated logger

from app.core.config import settings

logger = logging.getLogger(__name__)

# Define a Pydantic schema for the LLM's parameter extraction output
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
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) # Instantiate client once

    # --- Retries for LLM API Calls ---
    # Apply a retry decorator for transient API errors
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
            raise # Re-raise if all retries fail

    async def process_user_prompt(self, user_prompt: str, db_session: Any) -> Dict[str, Any]:
        self.trace_logger.log_event("Processing User Prompt", {"user_prompt": user_prompt})
        logger.info(f"Orchestrator processing prompt: '{user_prompt}'")

        try:
            # --- Step 1: Analyze the user's prompt ---
            analysis_result: ToolAnalysisResult = await self.analyzer.analyze_prompt(user_prompt)
            self.trace_logger.log_event("Prompt Analysis Result", analysis_result.model_dump())

            if analysis_result.needs_new_tool_generation:
                logger.info(f"Analysis indicates new tool generation is needed: {analysis_result.suggested_tool_name}")

                # Create MCPToolDefinition from analysis result
                mcp_parameters: List[MCPToolParameter] = []
                if analysis_result.required_parameters:
                    for param_name, param_type in analysis_result.required_parameters.items():
                        mcp_parameters.append(MCPToolParameter(
                            name=param_name,
                            type=param_type,
                            description=f"Parameter for {param_name}"
                        ))

                new_tool_definition = MCPToolDefinition(
                    name=analysis_result.suggested_tool_name,
                    description=analysis_result.tool_description,
                    parameters=mcp_parameters,
                    code=""
                )
                self.trace_logger.log_event("Generated MCPToolDefinition from Analysis", new_tool_definition.model_dump())

                # --- Step 2: Generate the Tool Code ---
                # Use retry mechanism for code generation
                generated_code = await self._call_llm_with_retries(
                    messages=[
                        {"role": "system", "content": self._get_tool_generation_system_prompt(new_tool_definition)},
                    ],
                    temperature=0.7 # Higher temperature for creativity in code
                )
                self.trace_logger.log_event("Generated Tool Code", {"code_length": len(generated_code) if generated_code else 0, "code_snippet": generated_code[:200] if generated_code else "None"})

                if not generated_code:
                    logger.error(f"Failed to generate code for tool: {new_tool_definition.name}")
                    return {"status": "error", "message": f"Could not generate code for the requested tool: {new_tool_definition.name}. Please try again or refine your request."}

                # --- Step 3: Robust Code Validation ---
                if not self._validate_generated_code(generated_code, new_tool_definition.name):
                    logger.error(f"Generated code failed robust validation for tool: {new_tool_definition.name}")
                    return {"status": "error", "message": "Generated tool code failed security/syntax validation. Please try again or contact support."}
                self.trace_logger.log_event("Code Validation Result", {"status": "passed"})

                new_tool_definition.code = generated_code

                # --- Step 4: Save the new tool to the database ---
                try:
                    db_tool = await self.tool_crud.create(db_session, obj_in=new_tool_definition)
                    logger.info(f"New tool '{db_tool.name}' saved to database with ID: {db_tool.id}")
                    self.trace_logger.log_event("Tool Saved to DB", {"tool_id": db_tool.id, "tool_name": db_tool.name})
                except Exception as e:
                    logger.error(f"Failed to save new tool '{new_tool_definition.name}' to database: {e}", exc_info=True)
                    return {"status": "error", "message": "Failed to save the newly generated tool. Please try again."}

                # Refresh the tool registry to include the newly added tool
                await self.registry.get_tool_definitions(db_session)

                # --- Step 5: Prepare and Execute the new tool ---
                tool_args = await self._extract_parameters_from_prompt(user_prompt, new_tool_definition.parameters)
                if tool_args is None: # Parameter extraction failed
                     return {"status": "error", "message": "Could not accurately extract parameters for the tool from your request. Please rephrase or provide more details."}

                # Check if any REQUIRED parameters are missing from the extracted args
                missing_required_params = [
                    p.name for p in new_tool_definition.parameters
                    if not p.optional and p.name not in tool_args
                ]
                if missing_required_params:
                    logger.warning(f"Missing required parameters for tool '{new_tool_definition.name}': {missing_required_params}")
                    return {"status": "error", "message": f"I need more information to use the '{new_tool_definition.name}' tool. Please provide values for: {', '.join(missing_required_params)}."}

                mcp_tool_call = MCPToolCall(tool_name=new_tool_definition.name, tool_args=tool_args)
                self.trace_logger.log_event("Executing Tool Call", mcp_tool_call.model_dump())

                try:
                    tool_response: MCPToolResponse = await self.sandbox.run_tool_in_sandbox(mcp_tool_call, db_session)
                    self.trace_logger.log_event("Tool Execution Result", tool_response.model_dump())
                    if tool_response.success:
                        return {"status": "success", "tool_output": tool_response.output, "tool_name": new_tool_definition.name}
                    else:
                        logger.error(f"Tool '{new_tool_definition.name}' execution failed in sandbox: {tool_response.error_message}")
                        return {"status": "error", "message": f"The tool '{new_tool_definition.name}' failed to execute: {tool_response.error_message}. Please check logs for details."}
                except Exception as e:
                    logger.error(f"Error running tool '{new_tool_definition.name}' in sandbox: {e}", exc_info=True)
                    return {"status": "error", "message": f"An error occurred while trying to run the tool '{new_tool_definition.name}': {e}"}

            else:
                logger.info("Analysis indicates no new tool generation is needed.")
                # --- Scenario: No new tool needed / Existing tool identified ---
                suggested_tool_name = analysis_result.suggested_tool_name
                if suggested_tool_name:
                    logger.info(f"Attempting to use existing tool: {suggested_tool_name}")
                    existing_tool_definition = self.registry.get_tool_by_name(suggested_tool_name)

                    if existing_tool_definition:
                        self.trace_logger.log_event("Existing Tool Identified", existing_tool_definition.model_dump())

                        tool_args = await self._extract_parameters_from_prompt(user_prompt, existing_tool_definition.parameters)
                        if tool_args is None: # Parameter extraction failed
                            return {"status": "error", "message": "Could not accurately extract parameters for the existing tool from your request. Please rephrase or provide more details."}

                        # Check if any REQUIRED parameters are missing from the extracted args
                        missing_required_params = [
                            p.name for p in existing_tool_definition.parameters
                            if not p.optional and p.name not in tool_args
                        ]
                        if missing_required_params:
                            logger.warning(f"Missing required parameters for existing tool '{existing_tool_definition.name}': {missing_required_params}")
                            return {"status": "error", "message": f"I need more information to use the '{existing_tool_definition.name}' tool. Please provide values for: {', '.join(missing_required_params)}."}

                        mcp_tool_call = MCPToolCall(tool_name=existing_tool_definition.name, tool_args=tool_args)
                        self.trace_logger.log_event("Executing Existing Tool Call", mcp_tool_call.model_dump())

                        try:
                            tool_response: MCPToolResponse = await self.sandbox.run_tool_in_sandbox(mcp_tool_call, db_session)
                            self.trace_logger.log_event("Existing Tool Execution Result", tool_response.model_dump())
                            if tool_response.success:
                                return {"status": "success", "tool_output": tool_response.output, "tool_name": existing_tool_definition.name}
                            else:
                                logger.error(f"Existing tool '{existing_tool_definition.name}' execution failed in sandbox: {tool_response.error_message}")
                                return {"status": "error", "message": f"The existing tool '{existing_tool_definition.name}' failed to execute: {tool_response.error_message}. Please check logs for details."}
                        except Exception as e:
                            logger.error(f"Error running existing tool '{existing_tool_definition.name}' in sandbox: {e}", exc_info=True)
                            return {"status": "error", "message": f"An error occurred while trying to run the existing tool '{existing_tool_definition.name}': {e}"}

                    else:
                        logger.warning(f"Suggested existing tool '{suggested_tool_name}' not found in registry. {analysis_result.tool_description}")
                        # Fallback to conversational response if tool not found but suggested
                        # This should ideally be handled by the agent_response directly in a full loop
                        return {"status": "info", "message": f"I analyzed your request, but the suggested tool '{suggested_tool_name}' was not found. {analysis_result.tool_description}. I can try to help conversationally."}
                else:
                    logger.info("No specific tool suggested by analyzer. Returning conversational response.")
                    # This is where you would integrate with llm_agent.get_agent_response for general chat
                    return {"status": "info", "message": analysis_result.tool_description or "I can help with that, but it doesn't require a specific tool at the moment."}

        except (APIStatusError, APIConnectionError) as e:
            logger.error(f"LLM API communication error during orchestration for prompt '{user_prompt}': {e}", exc_info=True)
            self.trace_logger.log_event("Orchestration LLM API Error", {"error": str(e), "prompt": user_prompt})
            return {"status": "error", "message": "I'm having trouble communicating with the AI services right now. Please try again shortly."}
        except Exception as e:
            logger.error(f"An unexpected error occurred during orchestration for prompt '{user_prompt}': {e}", exc_info=True)
            self.trace_logger.log_event("Orchestration Unexpected Error", {"error": str(e), "prompt": user_prompt})
            return {"status": "error", "message": f"An unexpected error occurred while processing your request: {e}. Please contact support."}

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
                    # Optional: Further checks on parameter count/names if rigid matching is needed
                    # Example: param_names = [arg.arg for arg in node.args.args]
                    break
            if not function_found:
                logger.warning(f"Code does not contain the expected async function '{expected_function_name}'.")
                return False

            # --- Check 3: Forbidden Imports ---
            # IMPORTANT: Adjust this list based on your sandbox's explicit permissions and tool requirements.
            # E.g., if a tool legitimately needs 'requests', remove it from this list.
            forbidden_modules = {
                'os', 'sys', 'subprocess', 'shutil', 'threading', 'multiprocessing',
                'socket', 'http', 'ftplib', 'smtplib', 'imaplib', 'poplib',
                'paramiko', 'scp', 'pexpect', 'webbrowser', # Potentially dangerous network/system modules
                # 'requests', 'urllib' # Only forbid if NO network access is allowed for any tool
            }

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imported_names = []
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            imported_names.append(n.name.split('.')[0]) # Get top-level module name
                    else: # ast.ImportFrom
                        if node.module:
                            imported_names.append(node.module.split('.')[0])

                    for name in imported_names:
                        if name in forbidden_modules:
                            logger.error(f"Generated code attempts to import forbidden module: '{name}'.")
                            return False

            # --- Check 4: Forbidden Function Calls / Expressions ---
            # Define a list of function names or attributes that are NOT allowed
            forbidden_builtins = {'eval', 'exec', 'open', 'input', 'breakpoint', 'compile'}
            # 'print' can be controlled by sandbox output redirection, often permitted for debugging.
            # 'exit', 'quit' can also be handled by sandbox resource limits/process management.
            dangerous_attributes = {'__import__', '__subclasses__', '__globals__', '__closure__', '__bases__', '__mro__', 'system', 'popen', 'read', 'write', 'delete'} # Common dangerous methods

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check for direct calls (e.g., eval())
                    if isinstance(node.func, ast.Name) and node.func.id in forbidden_builtins:
                        logger.error(f"Generated code attempts to call forbidden built-in function: '{node.func.id}'.")
                        return False
                    # Check for attribute calls (e.g., os.system(), file.read())
                    if isinstance(node.func, ast.Attribute) and node.func.attr in dangerous_attributes:
                         logger.error(f"Generated code attempts to call dangerous attribute as function: '{node.func.attr}'.")
                         return False

                # Check for direct attribute access of dangerous dunder methods or properties
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


    async def _extract_parameters_from_prompt(self, user_prompt: str, tool_parameters: List[MCPToolParameter]) -> Optional[Dict[str, Any]]:
        """
        Extracts parameter values from the user_prompt using an LLM,
        respecting whether parameters are optional or required.
        Returns None if extraction or validation fails critically.
        """
        if not tool_parameters:
            return {} # No parameters to extract

        # Dynamically build the description of expected parameters for the LLM
        params_description = []
        param_json_schema = {}
        dynamic_model_fields = {} # For Pydantic's create_model

        for p in tool_parameters:
            optional_str = " (optional)" if p.optional else ""
            params_description.append(f"- '{p.name}' ({p.type}){optional_str}: {p.description or 'No specific description provided.'}")

            # Map Python types to JSON schema types for the LLM prompt
            json_type = p.type.lower()
            if json_type == 'str': json_type = 'string'
            elif json_type == 'int': json_type = 'integer'
            elif json_type == 'float': json_type = 'number'
            elif json_type == 'bool': json_type = 'boolean'

            param_json_schema[p.name] = {"type": json_type, "description": p.description or f"Value for {p.name}"}

            # For the dynamic Pydantic model:
            # If optional, use Optional[Type] and default to None. Otherwise, it's required (...).
            python_type_map = {
                'str': str, 'int': int, 'float': float, 'bool': bool,
                'list': List[Any], 'dict': Dict[str, Any] # Add more specific types as needed
            }
            param_python_type = python_type_map.get(p.type.lower(), str) # Default to str if unknown

            if p.optional:
                dynamic_model_fields[p.name] = (Optional[param_python_type], None) # Optional, defaults to None
            else:
                dynamic_model_fields[p.name] = (param_python_type, ...) # Required


        # Create a dynamic Pydantic model for strict validation of LLM's output
        DynamicParamSchema = create_model(
            f'DynamicParamSchema_{len(tool_parameters)}_{abs(hash(frozenset(p.name for p in tool_parameters)))}', # Unique name
            __base__=BaseModel,
            **dynamic_model_fields
        )

        # Construct the system prompt for the LLM, emphasizing JSON schema and type correctness
        # Crucially, add `required` array to JSON schema for LLM to distinguish
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

        # ... (rest of the _extract_parameters_from_prompt function - no changes needed below this point) ...

        self.trace_logger.log_event("Parameter Extraction Prompt", {"tool_parameters": [p.model_dump() for p in tool_parameters], "user_prompt": user_prompt})
        logger.info(f"Extracting parameters for tool with params: {[p.name for p in tool_parameters]} from prompt: '{user_prompt}' using LLM.")

        try:
            raw_llm_response_content = await self._call_llm_with_retries(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0 # Make it highly deterministic for extraction
            )
            self.trace_logger.log_event("Raw LLM Parameter Extraction Response", {"response": raw_llm_response_content})

            # Attempt to parse directly into the dynamically created Pydantic model
            # This handles type coercion and validation
            extracted_params_validated = DynamicParamSchema.model_validate_json(raw_llm_response_content)
            extracted_params = extracted_params_validated.model_dump() # Convert to dict
            
            self.trace_logger.log_event("Extracted Parameters (Validated)", extracted_params)
            return extracted_params

        except ValidationError as e:
            logger.error(f"Pydantic validation error during parameter extraction: {e.errors()}. Raw LLM response: {raw_llm_response_content}", exc_info=True)
            self.trace_logger.log_event("Parameter Extraction Validation Error", {"error": str(e), "raw_response": raw_llm_response_content})
            return None # Indicate critical failure in extraction
        except Exception as e:
            logger.error(f"Unexpected error during parameter extraction LLM call: {e}", exc_info=True)
            self.trace_logger.log_event("Parameter Extraction Unexpected Error", {"error": str(e), "prompt": user_prompt})
            return None # Indicate critical failure


# Instantiate the orchestrator for easy import
agent_orchestrator_service = AgentOrchestrator()