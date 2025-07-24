# LangGraph workflow builder for the chatbot pipeline
# This chains QueryAgent, DatabaseAgent, and ResponseAgent

from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, START, END

# Import agents
from agents.query_agent import QueryAgent
from agents.database_agent import DatabaseAgent
from agents.response_agent import ResponseAgent

# Define the state schema for the workflow
class ChatbotState(TypedDict):
    user_prompt: str
    analysis: Dict[str, Any]
    data: List[Dict[str, Any]]
    response: str

def build_chatbot_workflow(db_name: str):
    """
    Build and return a compiled LangGraph StateGraph for the chatbot pipeline.
    The graph expects an initial state: {"user_prompt": ...}
    """
    query_agent = QueryAgent()
    database_agent = DatabaseAgent(db_name=db_name)
    response_agent = ResponseAgent()

    def query_node(state: ChatbotState, config=None) -> Dict[str, Any]:
        analysis = query_agent.analyze(state["user_prompt"])
        return {"analysis": analysis}

    def db_node(state: ChatbotState, config=None) -> Dict[str, Any]:
        data = database_agent.fetch(state["analysis"])
        return {"data": data}

    def response_node(state: ChatbotState, config=None) -> Dict[str, Any]:
        response = response_agent.generate_response(state["analysis"], state["data"])
        return {"response": response}

    builder = StateGraph(ChatbotState)
    builder.add_node("query", query_node)
    builder.add_node("database", db_node)
    builder.add_node("response", response_node)
    builder.add_edge(START, "query")
    builder.add_edge("query", "database")
    builder.add_edge("database", "response")
    builder.add_edge("response", END)
    builder.set_entry_point("query")
    builder.set_finish_point("response")
    return builder.compile()

# Example usage (to be called from main.py):
# chatbot_graph = build_chatbot_workflow(db_name="your_db_name")
# result = chatbot_graph.invoke({"user_prompt": "Show me all active users."})
# print(result["response"])
