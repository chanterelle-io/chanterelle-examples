import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(
  api_key=api_key
)

response = client.responses.create(
  model="gpt-5-nano",
  input="write a haiku about ai",
  store=True
)