import os
from typing import Dict, Any
from langchain.chat_models import init_chat_model

# Set your Google API key (ensure this is set securely in production)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "your-google-api-key")

# Initialize the Google Gemini LLM
llm = init_chat_model("google_genai:gemini-2.0-flash")

SYSTEM_PROMPT = (
    "You are a MongoDB query analysis agent. "
    "Given a user prompt, analyze and return a JSON object with: "
    "'intent' (string), 'collections' (list of strings), 'fields' (list of strings), "
    "'filters' (object), 'aggregation' (string or null), and any other useful metadata. "
    "Do NOT generate a MongoDB query, just the analysis. "
    "Example: {\"intent\": \"Find documents\", \"collections\": [\"users\"], \"fields\": [\"name\", \"email\"], \"filters\": {\"active\": true}, \"aggregation\": null}"
)

def analyze_user_prompt(prompt: str) -> Dict[str, Any]:
    """
    Analyze the user prompt and return a structured description for MongoDB querying using Google Gemini.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response = llm.invoke({"messages": messages})
    # Try to parse the response as JSON
    import json
    try:
        # If the LLM returns a code block, strip it
        content = response["content"] if isinstance(response, dict) else response
        if content.strip().startswith("```"):
            content = content.strip().strip("` ")
        return json.loads(content)
    except Exception:
        # Fallback: return the raw response
        return {"raw_response": response, "error": "Could not parse LLM output as JSON"}

class QueryAgent:
    """
    QueryAgent analyzes user prompts and produces a MongoDB-oriented query analysis using Google Gemini.
    """
    def __init__(self):
        pass

    def analyze(self, prompt: str) -> Dict[str, Any]:
        return analyze_user_prompt(prompt)
