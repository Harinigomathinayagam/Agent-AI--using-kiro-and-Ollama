# ============================================================
# Challenge 1 - Builders Skill Sprint
# Simple AI Chatbot using Strands SDK + Ollama (llama3.2:3b)
# ============================================================

from strands import Agent
from strands.models.ollama import OllamaModel

# Set up the local Ollama model
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b"
)

# Create the Agent with a system prompt
agent = Agent(
    model=ollama_model,
    system_prompt="You are a helpful AI assistant. Answer clearly and concisely."
)

print("=" * 50)
print("  AI Chatbot - Powered by Strands + Ollama")
print("  Type 'quit' or 'exit' to stop")
print("=" * 50)
print()

# Chat loop — keeps running until the user types 'quit' or 'exit'
while True:
    # Get input from the user
    user_input = input("You: ").strip()

    # Exit condition
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    # Skip empty input
    if not user_input:
        continue

    # Send message to the agent and print the response
    print("\nAssistant: ", end="", flush=True)
    agent(user_input)
    print()
