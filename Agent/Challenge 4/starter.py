# ============================================================
# Challenge 4 - Builders Skill Sprint
# Full Agent using Strands SDK + Ollama (llama3.2:3b)
# Tools: Calculator | Weather | Age Calculator
# Memory: Mem0 + FAISS (persistent, local)
# ============================================================
#
# SETUP — run these commands once before starting:
#
#   pip install strands-agents strands-agents-tools
#   pip install mem0ai
#   pip install faiss-cpu
#   pip install sentence-transformers
#
# This agent combines Challenge 2 (tools) + Challenge 3 (memory).
# It can do maths, check weather, calculate ages AND remember
# personal facts about you across sessions.
#
# Mem0 v2.x API rules (verified against installed version):
#   memory.add()     -> user_id= as top-level kwarg
#   memory.search()  -> filters={"user_id": ...}
#   memory.get_all() -> filters={"user_id": ...}
# ============================================================

import math
from datetime import date
from strands import Agent, tool
from strands.models.ollama import OllamaModel
from mem0 import Memory


# ============================================================
# SECTION 1: TOOLS
# ============================================================
# The @tool decorator turns a plain Python function into
# something the Strands agent can call automatically.
# The docstring is critical — the agent reads it to decide
# WHEN and HOW to use the tool.
# ============================================================


# ------------------------------------------------------------
# TOOL 1: CALCULATOR
# Handles any arithmetic or math expression the user asks.
# Uses eval() restricted to math module — safe, no arbitrary code.
# ------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Supports: +, -, *, /, ** (power), sqrt(), and basic math.
    Examples: '2 + 2', '10 * 5', 'sqrt(144)', '2 ** 8'
    """
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# ------------------------------------------------------------
# TOOL 2: WEATHER
# Simulated weather data — no API key needed.
# Swap in a real weather API call if you want live data.
# ------------------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    Returns temperature, condition, humidity, and wind speed.
    Available cities: London, New York, Tokyo, Sydney, Paris,
                      Dubai, Mumbai, Chennai, Bangalore, Delhi.
    """
    weather_data = {
        "london":    {"temp": 15, "condition": "Cloudy",        "humidity": 78, "wind": 12},
        "new york":  {"temp": 22, "condition": "Sunny",         "humidity": 55, "wind": 8},
        "tokyo":     {"temp": 28, "condition": "Humid",         "humidity": 85, "wind": 5},
        "sydney":    {"temp": 20, "condition": "Partly Cloudy", "humidity": 60, "wind": 14},
        "paris":     {"temp": 18, "condition": "Rainy",         "humidity": 80, "wind": 10},
        "dubai":     {"temp": 38, "condition": "Sunny",         "humidity": 40, "wind": 6},
        "mumbai":    {"temp": 32, "condition": "Humid",         "humidity": 90, "wind": 7},
        "chennai":   {"temp": 34, "condition": "Hot & Humid",   "humidity": 88, "wind": 9},
        "bangalore": {"temp": 26, "condition": "Pleasant",      "humidity": 65, "wind": 11},
        "delhi":     {"temp": 36, "condition": "Hot",           "humidity": 45, "wind": 8},
    }

    key = city.lower().strip()
    if key in weather_data:
        w = weather_data[key]
        return (
            f"Weather in {city.title()}:\n"
            f"  Temperature : {w['temp']}°C\n"
            f"  Condition   : {w['condition']}\n"
            f"  Humidity    : {w['humidity']}%\n"
            f"  Wind Speed  : {w['wind']} km/h"
        )
    else:
        available = ", ".join(c.title() for c in weather_data.keys())
        return (
            f"Weather data not available for '{city}'.\n"
            f"Available cities: {available}"
        )


# ------------------------------------------------------------
# TOOL 3: AGE CALCULATOR
# Calculates exact age and next milestone birthday (30, 40, 50…).
# ------------------------------------------------------------

@tool
def age_calculator(birth_year: int) -> str:
    """
    Calculate the current age of a person given their birth year.
    Also shows how many years until their next milestone birthday
    (30, 40, 50, etc.).
    Example: birth_year = 1995
    """
    current_year = date.today().year

    if birth_year < 1900 or birth_year > current_year:
        return (
            f"Invalid birth year: {birth_year}. "
            f"Please enter a year between 1900 and {current_year}."
        )

    age = current_year - birth_year
    next_milestone = ((age // 10) + 1) * 10
    years_to_milestone = next_milestone - age

    return (
        f"Age Calculation:\n"
        f"  Birth Year     : {birth_year}\n"
        f"  Current Year   : {current_year}\n"
        f"  Current Age    : {age} years old\n"
        f"  Next Milestone : {next_milestone} (in {years_to_milestone} year(s))"
    )


# ============================================================
# SECTION 2: MEM0 CONFIGURATION
# ============================================================
# FAISS = local vector database, no cloud required.
# embedding_model_dims = 384 matches all-MiniLM-L6-v2 output.
# path = folder where FAISS persists the index to disk.
# ============================================================

MEM0_CONFIG = {
    "vector_store": {
        "provider": "faiss",
        "config": {
            "embedding_model_dims": 384,
            "path": "./mem0_store"
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "ollama_base_url": "http://localhost:11434"
        }
    }
}


# ============================================================
# SECTION 3: INITIALISE MEM0
# ============================================================
# Loads or creates the FAISS index, downloads the embedder
# on first run (~90 MB), and connects to Ollama for fact extraction.
# ============================================================

print("Initialising memory store (first run downloads embedder ~90 MB)...")
memory = Memory.from_config(MEM0_CONFIG)
print("Memory store ready.\n")


# ============================================================
# SECTION 4: USER ID
# ============================================================
# All memories are stored under this ID.
# Change it to switch to a different user's memory profile.
# ============================================================

USER_ID = "thamarai"


# ============================================================
# SECTION 5: STRANDS AGENT WITH TOOLS
# ============================================================
# All three tools are passed to the agent.
# The system prompt tells it when to use tools and how to
# handle the Memory Context block injected each turn.
# ============================================================

ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b"
)

agent = Agent(
    model=ollama_model,
    tools=[calculator, get_weather, age_calculator],
    system_prompt=(
        "You are a helpful, friendly AI assistant with tools and persistent memory.\n\n"
        "TOOLS you can use:\n"
        "  1. calculator      - for any math or arithmetic questions\n"
        "  2. get_weather     - for weather information about a city\n"
        "  3. age_calculator  - for calculating someone's age from birth year\n\n"
        "MEMORY:\n"
        "  Each message may start with a 'Memory Context' block of stored facts.\n"
        "  Always use those facts in your reply.\n"
        "  When the user shares personal info (name, age, job, hobby, preference)\n"
        "  acknowledge it naturally."
    )
)


# ============================================================
# SECTION 6: MEMORY HELPER FUNCTIONS
# ============================================================
# Verified API signatures for Mem0 v2.0.4:
#
#   memory.add(messages, user_id=...)          <-- user_id top-level
#   memory.search(query, filters={user_id:..}) <-- user_id in filters
#   memory.get_all(filters={user_id:..})       <-- user_id in filters
# ============================================================

def save_to_memory(user_message: str, user_id: str) -> None:
    """
    Extract and persist facts from the user message into Mem0.
    add() takes user_id as a top-level kwarg (NOT inside filters).
    """
    memory.add(
        messages=[{"role": "user", "content": user_message}],
        user_id=user_id                 # top-level kwarg — correct for add()
    )


def get_memory_context(query: str, user_id: str, top_k: int = 5) -> str:
    """
    Search Mem0 for facts relevant to the current query.
    search() requires user_id inside filters={} (NOT top-level).
    Returns a formatted string or empty string if no memories yet.
    """
    results = memory.search(
        query=query,
        filters={"user_id": user_id},   # must be inside filters for search()
        limit=top_k
    )

    if not results or not results.get("results"):
        return ""

    facts = [item["memory"] for item in results["results"]]
    context = "Memory Context (facts I remember about you):\n"
    context += "\n".join(f"  - {fact}" for fact in facts)
    return context


def list_all_memories(user_id: str) -> None:
    """
    Print every stored memory for the given user.
    get_all() requires user_id inside filters={} (NOT top-level).
    """
    all_mems = memory.get_all(
        filters={"user_id": user_id}    # must be inside filters for get_all()
    )
    results = all_mems.get("results", [])
    if not results:
        print("  (no memories stored yet)")
        return
    for i, item in enumerate(results, 1):
        print(f"  {i}. {item['memory']}")


# ============================================================
# SECTION 7: INTERACTIVE CHAT LOOP
# ============================================================
# Pipeline every turn:
#   [1] save_to_memory()      - persist any new facts
#   [2] get_memory_context()  - retrieve relevant stored facts
#   [3] build enriched prompt - prepend memory to message
#   [4] agent()               - LLM replies, calls tools if needed
#
# Special commands:
#   "show memories" - dump all stored memories
#   "quit" / "exit" - end the session
# ============================================================

print("=" * 62)
print("  Challenge 4: Full Agent - Strands + Ollama + Mem0")
print("  Tools   : Calculator | Weather | Age Calculator")
print("  Memory  : Persistent across sessions (Mem0 + FAISS)")
print("  Type 'show memories' to see stored facts.")
print("  Type 'quit' or 'exit' to stop.")
print("=" * 62)
print()
print("Try asking:")
print("  - My name is Thamarai and I love cricket")
print("  - What is 128 * 256?")
print("  - What's the weather in Chennai?")
print("  - How old is someone born in 1998?")
print("  - What is my name?             <- tests memory recall")
print("  - What do you know about me?   <- tests memory recall")
print()

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye! Your memories are saved for next time.")
        break

    if not user_input:
        continue

    if user_input.lower() == "show memories":
        print("\nStored memories for this user:")
        list_all_memories(USER_ID)
        print()
        continue

    # STEP 1 — save the message (extracts & stores new facts)
    save_to_memory(user_input, USER_ID)

    # STEP 2 — retrieve relevant memories via similarity search
    memory_context = get_memory_context(user_input, USER_ID)

    # STEP 3 — prepend memory context to the prompt if we have any
    if memory_context:
        enriched_prompt = f"{memory_context}\n\nUser message: {user_input}"
    else:
        enriched_prompt = user_input

    # STEP 4 — call the agent (may invoke tools internally)
    print("\nAssistant: ", end="", flush=True)
    agent(enriched_prompt)
    print()
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye! Your memories are saved for next time.")
        break

    # ignore blank lines
    if not user_input:
        continue

    # debug: print all stored memories
    if user_input.lower() == "show memories":
        print("\nStored memories for this user:")
        list_all_memories(USER_ID)
        print()
        continue

    # ----------------------------------------------------------
    # STEP 1 — Save message to memory
    # We save BEFORE retrieving so the current message's facts
    # are persisted immediately and available on the next turn.
    # ----------------------------------------------------------
    save_to_memory(user_input, USER_ID)

    # ----------------------------------------------------------
    # STEP 2 — Retrieve relevant memories
    # Mem0 does a vector similarity search against the current
    # message and returns the most relevant stored facts.
    # ----------------------------------------------------------
    memory_context = get_memory_context(user_input, USER_ID)

    # ----------------------------------------------------------
    # STEP 3 — Build the enriched prompt
    # If we have memories, prepend them so the agent can use
    # them when forming its answer.
    # ----------------------------------------------------------
    if memory_context:
        enriched_prompt = f"{memory_context}\n\nUser message: {user_input}"
    else:
        enriched_prompt = user_input

    # ----------------------------------------------------------
    # STEP 4 — Call the agent
    # The agent reads the enriched prompt, decides whether to
    # call a tool (calculator / weather / age) or answer directly,
    # then streams the response to the terminal.
    # ----------------------------------------------------------
    print("\nAssistant: ", end="", flush=True)
    agent(enriched_prompt)
    print()
