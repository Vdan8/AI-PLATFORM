# backend/app/services/tool_generator.py
import logging
import re
from typing import Optional, Dict, Any

import openai
from app.core.config import settings
from app.schemas.tool import MCPToolDefinition, ToolParameter # Import the schemas

logger = logging.getLogger(__name__)

# --- ToolGenerator Class ---
class ToolGenerator:
    def __init__(self):
        """
        Initializes the ToolGenerator with the OpenAI client and LLM model.
        """
        self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.llm_model = settings.LLM_MODEL # Use the LLM model from settings

    async def generate_tool_code(self, tool_definition: MCPToolDefinition) -> Optional[str]:
        """
        Generates Python code for a tool based on its MCPToolDefinition using an LLM.
        """
        tool_name = tool_definition.name
        tool_description = tool_definition.description
        parameters = tool_definition.parameters if tool_definition.parameters else []

        # Dynamically build the function signature string
        # e.g., "async def searchWeb(query: str) -> Any:"
        params_str = ", ".join([f"{p.name}: {p.type}" for p in parameters])
        function_signature = f"async def {tool_name}({params_str}) -> Any:"

        # Construct the system prompt for the LLM
        # This prompt instructs the LLM on how to generate the Python code.
        system_prompt = f"""
        You are an expert Python programmer. Your task is to generate Python code for an asynchronous tool function.
        The tool's purpose is: "{tool_description}".
        It must strictly adhere to the following asynchronous function signature: `{function_signature}`.
        The function should return a Python native type (e.g., str, int, dict, list, bool).
        **Crucially, after performing its operation, the function MUST print its final result to standard output (stdout).**
        If an error occurs, print an error message to stdout as well.
        If a parameter is named 'query', it typically implies searching or retrieving information.

        **Constraints:**
        - Your response MUST contain ONLY the Python code for the function, enclosed within triple backticks (```python...```).
        - Do NOT include any explanations, comments (unless they are part of the function logic), or text outside the code block.
        - Do NOT include any import statements unless absolutely necessary within the function, as they will be handled by the execution environment. If you do include imports, ensure they are inside the function scope.
        - If the tool requires external libraries (e.g., 'requests' for web access, 'json' for parsing), assume they are available.
        - Ensure the function returns a value that directly represents the tool's output. **Remember to also print this final output to stdout.**

        Example:
        If the tool definition is for a tool named `searchWeb` with description "Searches the web for a given query" and parameters `query: str`:

        ```python
        import requests

        async def searchWeb(query: str) -> str:
            "Searches the web using a placeholder API and returns the search results."
            try:
                response = requests.get(f"[https://api.example.com/search?q=](https://api.example.com/search?q=){{query}}")
                response.raise_for_status() # Raise an exception for HTTP errors
                data = response.json()
                result = data.get("results", "No results found.")
                print(result) # <--- ADDED: Explicitly printing the result
                return result
            except requests.exceptions.RequestException as e:
                error_msg = f"Error searching web: {e}"
                print(error_msg) # <--- ADDED: Also print errors for debugging
                return error_msg
        ```

        Now, generate the code for the tool named `{tool_name}`:
        """

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        logger.info(f"Generating code for tool: '{tool_name}' with LLM model: {self.llm_model}")

        try:
            # Make the asynchronous call to the OpenAI API for code generation
            response = await self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.7, # A slightly higher temperature can lead to more diverse code solutions
            )
            raw_llm_response_content = response.choices[0].message.content
            logger.debug(f"Raw LLM code generation response: \n{raw_llm_response_content[:500]}...") # Log first 500 chars

            # Extract the Python code block from the LLM's response
            generated_code = self._extract_code_block(raw_llm_response_content)

            if generated_code:
                logger.info(f"Successfully generated code for tool: '{tool_name}'")
                return generated_code
            else:
                logger.warning(f"No Python code block found in LLM response for tool '{tool_name}'. Raw response: {raw_llm_response_content}")
                return None

        except openai.APIStatusError as e:
            logger.error(f"OpenAI API error during code generation: {e.status_code} - {e.response.text}")
            raise
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API connection error during code generation: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during code generation for tool '{tool_name}': {e}", exc_info=True)
            return None

    def _extract_code_block(self, text: str) -> Optional[str]:
        """
        Helper method: Extracts a Python code block from a Markdown-formatted string.
        Looks for text enclosed in triple backticks with 'python' language specifier.
        """
        # Regex to find a block starting with ```python and ending with ```
        # re.DOTALL makes '.' match newlines as well
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip() # Return the content within the code block
        return None

# Instantiate the generator for easy import as a singleton service
tool_generator_service = ToolGenerator()