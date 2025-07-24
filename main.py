import os
from workflow.graph import build_chatbot_workflow

def main():
    # Load environment variables (ensure these are set in your environment)
    db_name = os.getenv("MONGODB_DB_NAME")
    mongo_uri = os.getenv("MONGODB_ATLAS_URI")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not db_name:
        raise ValueError("MONGODB_DB_NAME environment variable not set.")
    if not mongo_uri:
        raise ValueError("MONGODB_ATLAS_URI environment variable not set.")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    # Build the chatbot workflow
    chatbot_graph = build_chatbot_workflow(db_name=db_name)

    print("Welcome to the MongoDB Chatbot! Type your question and press Enter.")
    try:
        while True:
            user_prompt = input("You: ").strip()
            if not user_prompt:
                continue
            if user_prompt.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            try:
                result = chatbot_graph.invoke({"user_prompt": user_prompt})
                print(f"Bot: {result.get('response', '[No response]')}")
            except Exception as e:
                print(f"[Error] {e}")
    except KeyboardInterrupt:
        print("\nExiting chatbot.")

if __name__ == "__main__":
    main()
