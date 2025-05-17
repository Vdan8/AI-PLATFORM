# app/core/clients.py
from openai import OpenAI
from app.core.config import settings

# Shared OpenAI client instance
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
