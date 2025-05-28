import logging
from typing import Optional, Dict, Any
import openai # We'll use OpenAI's Python client for LLM interaction
from pydantic import BaseModel, Field # For defining structured data schemas
from backend.config.config import settings # To access your OpenAI API key and LLM model name

logger = logging.getLogger(__name__) # Initialize a logger for this module

# --- Pydantic Schemas for LLM Output ---
class ToolAnalysisResult(BaseModel):
    """
    Schema for the analysis result from the LLM.
    This defines the structured JSON output we expect from the LLM.
    """
    needs_new_tool_generation: bool = Field(
        ..., # ... indicates this field is required
        description="True if a new tool needs to be generated based on the prompt, False if an existing tool might suffice or no tool is needed.",
    )
    suggested_tool_name: Optional[str] = Field(
        None, # None indicates this field is optional
        description="Suggested name for the new tool (if generation is needed) or an existing tool (if identified). Use camelCase (e.g., 'readFile').",
    )
    tool_description: Optional[str] = Field(
        None,
        description="A brief description of what the tool should do (if generation is needed).",
    )
    required_parameters: Optional[Dict[str, str]] = Field(
        None,
        description="Dictionary of required parameters for the tool, mapping parameter names to their Python type hints (e.g., {'query': 'str', 'file_path': 'str'}). Set to null if no parameters are needed.",
    )
    raw_llm_response: Optional[str] = Field(
        None,
        description="The raw JSON response string from the LLM before parsing, useful for debugging.",
    )

    # --- ToolAnalyzer Class ---
class ToolAnalyzer:
    def __init__(self):
        """
        Initializes the ToolAnalyzer with the OpenAI client and LLM model.
        """
        # Initialize the OpenAI client using the API key from your settings
        self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        # Get the LLM model name from your settings (e.g., "gpt-4o", "gpt-3.5-turbo")
        self.llm_model = settings.LLM_MODEL

    async def analyze_prompt(self, user_prompt: str) -> ToolAnalysisResult:
        """
        Analyzes the user's natural language prompt using an LLM to determine
        if a new tool needs to be generated, and if so, suggests its properties.
        """
        # The System Prompt: This is the instruction set for the LLM.
        # It defines the LLM's role, the task, and the REQUIRED JSON output format.
        system_prompt = f"""
        You are an AI assistant specialized in analyzing user requests to determine if a new tool needs to be generated or if an existing tool might be used.
        Your goal is to output a JSON object adhering to the ToolAnalysisResult schema.

        If a new tool is needed, provide:
        - `needs_new_tool_generation`: `true`
        - `suggested_tool_name`: A concise, camelCase name for the tool (e.g., "readFile", "generateImage").
        - `tool_description`: A brief description of what the tool should accomplish.
        - `required_parameters`: A dictionary mapping parameter names to their Python type hints (e.g., {{"query": "str", "count": "int"}}).

        If no new tool is needed, or an existing general-purpose tool is sufficient, provide:
        - `needs_new_tool_generation`: `false`
        - `suggested_tool_name`: The name of the existing tool if applicable (e.g., "searchWeb").
        - `tool_description`: A brief explanation of why no new tool is needed or how an existing tool applies.
        - `required_parameters`: `null` or an empty dictionary.

        Your response MUST be a valid JSON object matching the ToolAnalysisResult schema.
        Strictly output only the JSON.

        Examples:
        User: "Can you summarize the content of the document 'report.pdf'?"
        Assistant:
        {{
            "needs_new_tool_generation": true,
            "suggested_tool_name": "summarizePdf",
            "tool_description": "A tool that reads and summarizes the content of a specified PDF file.",
            "required_parameters": {{"file_path": "str"}}
        }}

        User: "What's the current weather in London?"
        Assistant:
        {{
            "needs_new_tool_generation": false,
            "suggested_tool_name": "getCurrentWeather",
            "tool_description": "An existing tool for fetching current weather information by city.",
            "required_parameters": {{"location": "str"}}
        }}

        User: "Tell me a joke."
        Assistant:
        {{
            "needs_new_tool_generation": false,
            "suggested_tool_name": null,
            "tool_description": "This is a general conversational request and does not require a specific tool.",
            "required_parameters": null
        }}
        """

        # Prepare the messages for the LLM API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"Analyzing prompt: '{user_prompt}' with LLM model: {self.llm_model}")

        try:
            # Make the asynchronous call to the OpenAI API
            response = await self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                response_format={"type": "json_object"}, # IMPORTANT: Instructs OpenAI to return a JSON object
                temperature=0.4, # A lower temperature makes the output more deterministic and less creative
            )
            # Extract the content from the LLM's response
            raw_llm_response_content = response.choices[0].message.content
            logger.debug(f"Raw LLM analysis response: {raw_llm_response_content}")

            # Parse the raw JSON string into our Pydantic model for validation and structured access
            analysis_data = ToolAnalysisResult.parse_raw(raw_llm_response_content)
            # Store the raw response in the Pydantic model for debugging/inspection
            analysis_data.raw_llm_response = raw_llm_response_content

            logger.info(f"Prompt analysis complete. Needs new tool: {analysis_data.needs_new_tool_generation}")
            return analysis_data

        except openai.APIStatusError as e:
            # Handle API errors (e.g., invalid API key, rate limits)
            logger.error(f"OpenAI API error during prompt analysis: {e.status_code} - {e.response.text}")
            raise # Re-raise the exception after logging

        except openai.APIConnectionError as e:
            # Handle network/connection errors
            logger.error(f"OpenAI API connection error during prompt analysis: {e}")
            raise

        except Exception as e:
            # Catch any other unexpected errors, especially during JSON parsing
            logger.error(f"Unexpected error during prompt analysis: {e}", exc_info=True)
            # Return a default/safe analysis result if parsing fails
            return ToolAnalysisResult(
                needs_new_tool_generation=False,
                suggested_tool_name=None,
                tool_description=f"Error analyzing prompt: {e}",
                required_parameters=None,
                raw_llm_response=raw_llm_response_content if 'raw_llm_response_content' in locals() else None
            )
        

# Instantiate the analyzer for easy import as a singleton service
tool_analyzer_service = ToolAnalyzer()