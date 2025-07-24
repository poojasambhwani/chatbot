import os
from typing import Dict, Any, List
from langchain.chat_models import init_chat_model
from db_conn.connection import get_mongo_client

# Set your Google API key (ensure this is set securely in production)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "your-google-api-key")

# Initialize the Google Gemini LLM
llm = init_chat_model("google_genai:gemini-2.0-flash")

SYSTEM_PROMPT = (
    "You are a MongoDB query generator. "
    "Given a structured analysis (intent, collections, fields, filters, aggregation, etc.), "
    "output ONLY a valid MongoDB query as a Python dictionary (no explanation, no code block, no extra text). "
    "If aggregation is required, output an aggregation pipeline as a list of dicts. "
    "Otherwise, output a find query as a dict with 'filter' and 'projection' keys. "
    "Example for find: {\"filter\": {\"active\": true}, \"projection\": {\"name\": 1, \"email\": 1}}. "
    "Example for aggregation: [{\"$match\": {\"active\": true}}, {\"$group\": {\"_id\": \"$role\", \"count\": {\"$sum\": 1}}}]."
)

def generate_mongo_query(analysis: Dict[str, Any]) -> Any:
    """
    Use LLM to generate a MongoDB query (find or aggregation) from the analysis.
    """
    import json
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analysis: {json.dumps(analysis)}"},
    ]
    response = llm.invoke({"messages": messages})
    # Try to parse the response as JSON or Python dict
    content = response["content"] if isinstance(response, dict) else response
    try:
        if content.strip().startswith("```)":
            content = content.strip().strip("` ")
        # Try JSON first
        return json.loads(content)
    except Exception:
        try:
            # Try eval as Python dict/list (safe only if LLM is trusted)
            return eval(content, {"__builtins__": {}})
        except Exception:
            return {"error": "Could not parse LLM output", "raw": content}

class DatabaseAgent:
    """
    DatabaseAgent takes the analysis from QueryAgent, uses LLM to generate a MongoDB query, executes it, and returns the results.
    """
    def __init__(self, db_name: str):
        self.client = get_mongo_client()
        self.db = self.client[db_name]

    def fetch(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use LLM to generate a MongoDB query from analysis, execute it, and return results.
        """
        collections = analysis.get("collections", [])
        if not collections:
            raise ValueError("No collection specified in analysis.")
        collection_name = collections[0]  # For now, use the first collection
        collection = self.db[collection_name]
        query_obj = generate_mongo_query(analysis)
        if isinstance(query_obj, list):  # Aggregation pipeline
            cursor = collection.aggregate(query_obj)
        elif isinstance(query_obj, dict):
            filter_ = query_obj.get("filter", {})
            projection = query_obj.get("projection")
            cursor = collection.find(filter_, projection)
        else:
            raise ValueError(f"Invalid query object from LLM: {query_obj}")
        return list(cursor)
