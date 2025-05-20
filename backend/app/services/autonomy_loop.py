from app.core.config import settings
from app.core.clients import openai_client
from app.services.tool_registry import get_tool_definitions, call_tool
from app.utils.logger import trace_logger_instance
import json


def run_autonomous_agent(system_prompt: str, user_goal: str, max_steps: int = 5):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_goal}
    ]

    tools = get_tool_definitions()

    for step in range(max_steps):
        print(f"\n🔁 Step {step + 1}: Sending to OpenAI")

        response = openai_client.chat.completions.create( # USING THE SHARED CLIENT
        model=settings.LLM_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            # Append assistant tool call message
            messages.append({
                "role": "assistant",
                "tool_calls": message.tool_calls
            })

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                result = call_tool(name, arguments)

                # Ensure result is a string
                if not isinstance(result, str):
                    result = json.dumps(result)

                # Append tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

                # Trace log the tool call
                trace_logger_instance({
                    "step": step + 1,
                    "type": "tool_call",
                    "tool_name": name,
                    "tool_args": arguments,
                    "tool_result": result,
                    "tool_call_id": tool_call.id
                })

        else:
            # Final assistant response
            content = message.content
            if not isinstance(content, str):
                content = json.dumps(content)

            messages.append({
                "role": "assistant",
                "content": content
            })

            # Trace log final reply
            trace_logger_instance({
                "step": step + 1,
                "type": "final_reply",
                "content": content
            })

            return content

    return "⚠️ Max steps reached without resolution."
