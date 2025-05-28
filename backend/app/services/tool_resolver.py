import logging
import json
import ast
import os
import sys # Needed for example in prompt, but also for sys.exit in generated code example
import asyncio # Needed for asyncio.run in generated code example
from typing import Any, Dict, Optional, List, Type, Tuple, Union
from pydantic import ValidationError, BaseModel, Field, create_model
from openai import AsyncOpenAI, APIStatusError, APIConnectionError, InternalServerError, RateLimitError, APITimeoutError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_log, after_log
from sqlalchemy.exc import IntegrityError
from asyncpg.exceptions import UniqueViolationError
from uuid import uuid4 # For creating unique model names

from backend.config.config import settings
from app.services.tool_registry import ToolRegistryService
from app.services.tool_generator import ToolGeneratorService # ToolGeneratorService will be called by resolver
from app.crud.tool import CRUDBase # For interacting with the database CRUD operations for tools
from app.schemas.tool import MCPToolDefinition, ToolParameter, ToolCreate # Import ToolCreate
from app.utils.logger import TraceLogger
from app.services.tool_writer import save_tool_file
from app.services.tool_loader import dynamic_import_tool


logger = logging.getLogger(__name__)

# Define a Pydantic schema for the LLM's parameter extraction output (can be reused)
class ExtractedParameters(BaseModel):
    parameters: Dict[str, Any] = Field(
        ...,
        description="A dictionary where keys are parameter names and values are the extracted values.",
    )

class ToolResolverService:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        trace_logger: TraceLogger,
        generator: ToolGeneratorService, # Will pass tool_generator_service instance
        tool_crud: CRUDBase, # Will pass tool_crud instance
        registry: ToolRegistryService # Will pass tool_registry_service instance
    ):
        self.openai_client = openai_client
        self.trace_logger = trace_logger
        self.generator = generator # Assign generator service
        self.tool_crud = tool_crud # Assign CRUD operations for tools
        self.registry = registry # Assign registry to add newly generated tools
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
            return str
        return py_type

    async def _extract_parameters_for_next_step(
        self,
        tool_definition: MCPToolDefinition,
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
        if not tool_definition.parameters:
            return {}, None

        # Dynamic Pydantic model for strict validation
        dynamic_model_fields = {}
        for param in tool_definition.parameters:
            py_type = self._get_python_type_from_str(param.type)

            if param.required:
                dynamic_model_fields[param.name] = (py_type, Field(..., description=param.description))
            else:
                dynamic_model_fields[param.name] = (Optional[py_type], Field(None, description=param.description))

        # Use a unique name for the dynamic model to avoid conflicts if called repeatedly
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
                param_name = error['loc'][0]
                error_type = error['type']
                error_input = error.get('input')

                # Specific check for LLM placeholder issue (e.g., 'extracted_value_1')
                if isinstance(error_input, str) and ('extracted_value' in error_input.lower() or 'placeholder' in error_input.lower()):
                    if error_type == 'int_parsing':
                        return {}, f"Parameter '{param_name}' requires an integer, but you provided a placeholder '{error_input}'. Please extract the concrete integer value from the user's request (e.g., '5') or indicate if you cannot determine it."
                    elif error_type == 'string_type':
                        return {}, f"Parameter '{param_name}' requires a string, but you provided a placeholder '{error_input}'. Please extract the concrete string value from the user's request or indicate if you cannot determine it."
                    return {}, f"Parameter '{param_name}' has an invalid value: '{error_input}'. It seems to be a placeholder. Please provide the actual value."
                elif error_type == 'missing':
                    # This happens if a required parameter is not provided at all by the LLM
                    return {}, f"Missing required parameter '{param_name}'. Please ensure you extract this value from the user's prompt or ask the user for it if it's missing."
            # Fallback for any other validation error not specifically handled
            return {}, f"Parameter validation failed: {e.errors()}"
        except Exception as e:
            logger.exception(f"An unexpected error occurred during parameter extraction for tool {tool_definition.name}.")
            return {}, f"An unexpected error occurred during parameter extraction: {e.__class__.__name__}: {e}"

    async def extract_initial_prompt_parameters(self, user_prompt: str, tool_parameters: List[ToolParameter]) -> Optional[Dict[str, Any]]:
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
            raw_llm_response_message = await self._call_llm_with_retries(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            raw_llm_response_content = raw_llm_response_message.content

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
        original_code = code

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
            forbidden_modules = self._get_forbidden_modules_list() # Using static helper
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
                            imported_names.append(n.name)

                    for name in imported_names:
                        if name in forbidden_modules:
                            logger.error(f"Validation failed: Forbidden module import detected: '{name}'.")
                            return False

            # --- Check 4: Forbidden Function Calls / Expressions ---
            forbidden_builtins = self._get_forbidden_builtins_list() # Using static helper
            dangerous_attributes = self._get_dangerous_attributes_list() # Using static helper

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

    def _get_tool_generation_system_prompt(self, tool_definition: MCPToolDefinition) -> List[Dict[str, str]]:
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
        if not param_args and tool_definition.parameters:
            param_args = ", ".join([f"{p_name}: Any" for p_name in parameters_schema.keys()])
        elif not tool_definition.parameters:
            param_args = ""

        # Ensure you have _get_forbidden_list defined as a static method in this class
        forbidden_list_str = ", ".join(self._get_forbidden_list_for_llm_prompt())

        system_message_content = f"""
        You are an expert Python developer tasked with writing a self-contained, asynchronous Python function.
        This function will be used as a tool in an AI agent system.

        Here is the definition of the tool you need to implement:
        Name: {tool_definition.name}
        Description: {tool_definition.description}
        Parameters: {json.dumps(parameters_schema, indent=2)}

        Constraints and Requirements:
        1.  **Function Signature**: The function MUST be an `async def` function with the name `{tool_definition.name}`.
        2.  **Parameter Handling**: The function MUST accept all its parameters as direct arguments, matching the names and types specified.
        3.  **JSON Output to stdout**: The function MUST print its final result to standard output (stdout) as a single, valid JSON object.
            * **For success**: The JSON object should have a "status" field set to "success" and an "output" field containing the result.
                Example: `print(json.dumps({{"status": "success", "output": <your_result_here>}}))`
            * **For errors**: If an error occurs during the tool's execution, print a JSON object to standard error (stderr) with "status" set to "error" and a "message" field describing the error. Then, exit with a non-zero status code (`sys.exit(1)`).
                Example: `print(json.dumps({{"status": "error", "message": "An error occurred."}}), file=sys.stderr); sys.exit(1)`
        4.  **Executable Script**: Include an `if __name__ == "__main__":` block to demonstrate how the function would be called and how its output should be printed. This block should parse arguments (e.g., from environment variables like `TOOL_ARGUMENTS`) and call your function, then print the result as JSON to stdout.
        5.  **Forbidden Imports/Usage**: The function MUST NOT import any modules or use any built-in functions or attributes that could pose a security risk or allow arbitrary code execution. Specifically, do NOT import or use: {forbidden_list_str}.
        6.  **Network Access**: If the tool requires network access, use `aiohttp` for asynchronous HTTP requests.
        7.  **No Extra Text**: Your response MUST contain ONLY the Python script, enclosed within a single markdown code block (```python ... ```). Do NOT include any other text, explanation, or example usage outside this code block.

        Example Structure for `if __name__ == "__main__":` block:
        ```python
        import json
        import os
        import sys
        import asyncio # For async functions

        # Import aiohttp if network access is common for your tools
        # import aiohttp

        # Define your tool function here
        async def {tool_definition.name}({param_args}):
            # Your implementation here
            # Example:
            # async with aiohttp.ClientSession() as session:
            #     response = await session.get("[https://api.example.com](https://api.example.com)")
            #     data = await response.json()
            # return {{"api_data": data}}
            return {{"message": "Tool executed successfully"}} # Placeholder

        if __name__ == "__main__":
            # Arguments parsing and execution
            args_json = os.getenv("TOOL_ARGUMENTS")
            tool_args = {{}}
            if args_json:
                try:
                    tool_args = json.loads(args_json)
                except json.JSONDecodeError:
                    print(json.dumps({{"status": "error", "message": "Invalid TOOL_ARGUMENTS JSON."}}), file=sys.stderr)
                    sys.exit(1)

            try:
                # Call the async tool function
                result = asyncio.run({tool_definition.name}(**tool_args))
                print(json.dumps({{"status": "success", "output": result}}))
            except Exception as e:
                print(json.dumps({{"status": "error", "message": f"Error executing tool: {{e}}"}}), file=sys.stderr)
                sys.exit(1)
        ```
        """
        return [{"role": "system", "content": system_message_content}]

    # Helper to provide forbidden list to the LLM (optional but good for transparency)
    @staticmethod
    def _get_forbidden_list_for_llm_prompt():
        """Returns a combined list of forbidden modules, built-ins, and attributes for LLM prompt."""
        forbidden_modules = ToolResolverService._get_forbidden_modules_list()
        forbidden_builtins = ToolResolverService._get_forbidden_builtins_list()
        dangerous_attributes = ToolResolverService._get_dangerous_attributes_list()
        return sorted(list(forbidden_modules.union(forbidden_builtins).union(dangerous_attributes)))

    # Static helper methods for forbidden lists (used by _validate_generated_code and _get_forbidden_list_for_llm_prompt)
    @staticmethod
    def _get_forbidden_modules_list():
        return {
            'os', 'sys', 'subprocess', 'shutil', 'threading', 'multiprocessing',
            'socket', 'http', 'ftplib', 'smtplib', 'imaplib', 'poplib',
            'paramiko', 'scp', 'pexpect', 'webbrowser', 'pickle', 'cPickle',
            'yaml', 'json', 'importlib', 'inspect',
            'distutils', 'setuptools', 'pip',
            'ctypes', 'sysconfig', 'gc', 'resource',
        }

    @staticmethod
    def _get_forbidden_builtins_list():
        return {'eval', 'exec', 'open', 'input', 'breakpoint', 'compile', 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'dir'}

    @staticmethod
    def _get_dangerous_attributes_list():
        return {
            '__import__', '__subclasses__', '__globals__', '__closure__', '__bases__', '__mro__',
            'system', 'popen', 'read', 'write', 'delete', 'exit', 'quit',
            '__dict__', '__class__', '__builtins__',
            '__reduce__', '__reduce_ex__',
        }

    async def resolve_tool_generation(self, tool_name: str, tool_description: str, llm_decision_parameters: Dict[str, Any], db_session: Any) -> Tuple[bool, Optional[str]]:
        """
        Handles the complete process of generating, validating, saving, and registering a new tool.
        Returns (success_status, error_message)
        """
        logger.info(f"Resolver: Attempting to generate new tool '{tool_name}'")
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
            code="" # Code will be filled by LLM
        )

        generated_code_message = await self._call_llm_with_retries(
            messages=self._get_tool_generation_system_prompt(new_tool_definition_proto),
            temperature=0.7
        )
        generated_code = generated_code_message.content

        if not generated_code or not self._validate_generated_code(generated_code, tool_name):
            error_msg = f"Generated code for '{tool_name}' failed validation."
            logger.error(error_msg)
            await self.trace_logger.log_event("Tool Generation Failed", {"tool_name": tool_name, "error": error_msg})
            return False, error_msg

        new_tool_definition_proto.code = generated_code

        try:
            tool_file_path = save_tool_file(
                location="app/tools/",
                filename=f"tool_{tool_name}.py",
                code=generated_code
            )
            logger.info(f"✅ Tool script saved to: {tool_file_path}")
            await self.trace_logger.log_event("Tool Script Saved to Disk", {"path": tool_file_path})
        except Exception as file_err:
            error_msg = f"❌ Failed to save tool to disk: {file_err}"
            logger.error(error_msg)
            await self.trace_logger.log_event("Tool Save to Disk Failed", {"error": str(file_err)})
            return False, error_msg

        try:
            tool_path = os.path.join("app/tools", f"tool_{tool_name}.py")
            tool_function = dynamic_import_tool(tool_path)
            self.registry.add_tool_to_memory_registry(tool_name, tool_function)
            logger.info(f"✅ Tool '{tool_name}' dynamically imported and added to live registry.")
            await self.trace_logger.log_event("Tool Live Registered", {"tool_name": tool_name})
        except Exception as import_err:
            error_msg = f"❌ Failed to dynamically import/register tool '{tool_name}': {import_err}"
            logger.error(error_msg)
            await self.trace_logger.log_event("Dynamic Import Failed", {"error": str(import_err)})
            return False, error_msg

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

        try:
            # Attempt to create the tool in DB
            db_tool = await self.tool_crud.create(db_session, tool_to_save)
            logger.info(f"Successfully generated and saved new tool to DB: {tool_to_save.name}")
            await self.trace_logger.log_event("New Tool Generated & Saved to DB", {"tool_name": tool_to_save.name})
            return True, None
        except (IntegrityError, UniqueViolationError) as e:
            await db_session.rollback() # IMMEDIATELY ROLLBACK
            logger.warning(f"Tool '{tool_to_save.name}' already exists in DB. Attempting to retrieve. Original error: {e}")
            db_tool = await self.tool_crud.get_by_name(db_session, tool_to_save.name)
            if db_tool:
                logger.info(f"Retrieved existing tool '{tool_to_save.name}' after duplicate creation attempt.")
                await self.trace_logger.log_event("Existing Tool Retrieved (Duplicate Attempt)", {"tool_name": tool_to_save.name})
                return True, None # Treat as success if existing tool is found
            else:
                error_msg = f"Critical: Failed to retrieve tool '{tool_to_save.name}' after unique violation and rollback. Cannot proceed. Error: {e}"
                logger.error(error_msg)
                return False, error_msg
        except Exception as e:
            error_msg = f"An unexpected database error occurred during tool save: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg


    async def resolve_tool_parameters_for_execution(self, tool_name: str, llm_decision_parameters: Dict[str, Any], user_prompt: str, db_session: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Resolves and validates parameters for an existing tool before execution.
        Returns (extracted_parameters, error_message).
        """
        logger.info(f"Resolver: Resolving parameters for existing tool '{tool_name}'")
        db_tool = await self.registry.get_tool_by_name(db_session, tool_name)

        if not db_tool:
            return None, f"Tool '{tool_name}' not found in registry."

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
            mcp_parameters_list = db_tool.parameters
        else:
            return None, f"Internal error: Malformed tool definition parameters for {tool_name}."

        tool_definition = MCPToolDefinition(
            name=db_tool.name,
            description=db_tool.description,
            parameters=mcp_parameters_list,
            code=db_tool.code
        )

        extracted_tool_args, param_error_message = await self._extract_parameters_for_next_step(
            tool_definition,
            llm_decision_parameters,
            user_prompt
        )

        if param_error_message:
            return None, param_error_message
        else:
            await self.trace_logger.log_event('Parameters Validated for Next Step', extracted_tool_args)
            return extracted_tool_args, None