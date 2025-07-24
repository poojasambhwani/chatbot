import os
from typing import Dict, Any, List
from langchain.chat_models import init_chat_model

# Set your Google API key (ensure this is set securely in production)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "your-google-api-key")

# Initialize the Google Gemini LLM
llm = init_chat_model("google_genai:gemini-2.0-flash")

SYSTEM_PROMPT = (
    "You are a response generation agent for a database chatbot. "
    "Given the user's original query analysis and the raw data fetched from the database, "
    "compose a clear, structured, and analytical response for the user. "
    "Use chain-of-thought reasoning: first summarize the user's intent and the data found, "
    "then provide insights, trends, or highlights if possible. "
    "Output should be user-friendly, concise, and informative. "
    "If the data is empty, explain that no results were found and suggest possible next steps."
)

class ResponseAgent:
    """
    ResponseAgent takes the user query analysis and database results, and uses LLM to generate a structured, analytical response.
    """
    def __init__(self):
        pass

    def generate_response(self, analysis: Dict[str, Any], data: List[Dict[str, Any]]) -> str:
        """
        Use LLM to generate a user-friendly, analytical response from the analysis and data.
        """
        import json
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"User Query Analysis: {json.dumps(analysis)}\n"
                f"Database Results: {json.dumps(data)}"
            )},
        ]
        response = llm.invoke({"messages": messages})
        return response["content"] if isinstance(response, dict) else response
