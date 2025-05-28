# backend/app/services/llm_agent.py
from backend.config.config import settings
from backend.app.core.llm_clients import openai_client


def generate_agent_profile(prompt: str) -> str:
    """
    Takes a user prompt and generates a structured AI employee persona.
    Returns a stringified JSON-like response from the LLM.
    """
    system_instruction = (
        "You are an AI that creates virtual employee personas based on business needs. "
        "Given a user's description, return a JSON object with: "
        "name, personality, tools, and a system_prompt."
    )

    response = openai_client.chat.completions.create( # USING THE SHARED CLIENT
        model=settings.DEFAULT_GPT_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content
def get_agent_response(system_prompt: str, history: list) -> str:
    """
    Takes the agent's system prompt and a history of messages.
    Returns the assistant's next response.
    """
    messages = [{"role": "system", "content": system_prompt}] + history

    response = openai_client.chat.completions.create( # USING THE SHARED CLIENT
        model=settings.DEFAULT_GPT_MODEL,
        messages=messages,
        temperature=0.7
    )

    return response.choices[0].message.content
