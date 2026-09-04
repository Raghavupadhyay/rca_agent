import os
import sys
from langchain_core.messages import HumanMessage

# Add the project root to sys.path so we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.app.agent import build_agent, extract_context_from_messages

def load_env(filepath=".env"):
    """Simple parser to load .env without requiring python-dotenv"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filepath)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, val = line.strip().split('=', 1)
                    os.environ[key] = val

def main():
    # Load environment variables manually
    load_env()
    
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in environment.")
        print("Please add it to your .env file in the project root.")
        sys.exit(1)

    print("Building LangGraph agent...")
    app = build_agent()
    
    # State management for the session
    message_history = []
    context = {}
    
    print("\nAgent is ready! Type 'exit' to quit.")
    print("Try asking: 'How did Bangalore do on 2026-04-22?'")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
                
            # 1. Add user message to history
            message_history.append(HumanMessage(content=user_input))
            
            # 2. Update the sticky context based on the latest input
            context = extract_context_from_messages(message_history, context)
            
            # 3. Invoke the graph
            print("Agent is thinking (running tools)...")
            state_input = {
                "messages": message_history,
                "context": context
            }
            
            # The graph will run until it reaches the END node
            result = app.invoke(state_input, {"recursion_limit": 25})
            
            # 4. Update our history with the newly generated messages (AI & Tools)
            message_history = result["messages"]
            
            # 5. Print the final AI response
            final_message = message_history[-1].content
            print(f"\nAgent: {final_message}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
