# backend/app/services/tool_generator.py

import logging
import re
from typing import Optional, Dict
import openai
from backend.config.config import settings
from app.schemas.tool import MCPToolDefinition


logger = logging.getLogger(__name__)


class ToolGeneratorService:
    def __init__(self):
        self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.llm_model = settings.LLM_MODEL

    async def generate_tool_code(self, tool_definition: MCPToolDefinition) -> Optional[Dict[str, str]]:
        tool_name = tool_definition.name
        tool_description = tool_definition.description

        # Use placeholders inside string
        system_prompt = f'''
You are an expert Python agent system designer. Your task is to generate a tool for an autonomous AI OS.
The tool name is "{tool_name}" and its purpose is: "{tool_description}".

You MUST generate the code for a single Python file named: tool_{tool_name}.py
Location: app/tools/tool_{tool_name}.py

This file must contain:
def execute(input: dict) -> dict:
"""
Executes the tool logic. Takes a dictionary as input and returns a dictionary.
"""
import json
try:
# Tool logic here
result = ...
return {{ "status": "success", "output": result }}
except Exception as e:
return {{ "status": "error", "message": str(e) }}
Do NOT include any CLI runners, async wrappers, test cases, or explanations.
Output ONLY the Python code inside a triple-backtick block.
Make sure it's ready to be saved to app/tools/tool_{tool_name}.py
'''
        messages = [{"role": "system", "content": system_prompt}]
        logger.info(f"Generating code for tool: '{tool_name}'")

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.7,
            )
            raw_llm_response_content = response.choices[0].message.content
            logger.debug(f"LLM response: {raw_llm_response_content[:500]}")

            generated_code = self._extract_code_block(raw_llm_response_content)

            if not generated_code:
                logger.warning(f"No code block found in response for tool '{tool_name}'")
                return None

            return {
                "filename": f"tool_{tool_name}.py",
                "location": "app/tools/",
                "code": generated_code,
            }

        except openai.APIStatusError as e:
            logger.error(f"OpenAI API error: {e.status_code} - {e.response.text}")
            raise
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}")
            raise
        except Exception as e:
            logger.error(f"Tool generation failed: {e}", exc_info=True)
            return None

    def _extract_code_block(self, text: str) -> Optional[str]:
        match = re.search(r"``````", text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    # Instantiate the service as a singleton for easy import elsewhere
tool_generator_service = ToolGeneratorService()