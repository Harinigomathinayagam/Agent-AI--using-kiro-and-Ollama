# ============================================================
# Challenge 3 - Builders Skill Sprint
# Memory Agent using Strands SDK + Ollama (llama3.2:3b)
# Persistent Memory: Mem0 + FAISS (local vector store)
# ============================================================
#
# SETUP — run these commands before starting:
#
#   pip install strands-agents strands-agents-tools
#   pip install mem0ai
#   pip install faiss-cpu
#   pip install sentence-transformers
#
# HOW IT WORKS:
#   - Mem0 manages memory: it stores and retrieves facts
#   - FAISS is the local vector database (no cloud needed)
#   - sentence-transformers converts text to embeddings
#   - The agent searches memory before every reply
#   - Memories survive restarts (saved to ./mem0_store/)
#
# Mem0 v2.x API rules (verified against installed version):
#   memory.add()     -> user_id= as top-level kwarg
#   memory.search()  -> filters={"user_id": ...}
#   memory.get_all() -> filters={"user_id": ...}
# ============================================================

from strands import Agent
from strands.models.ollama import OllamaModel
from mem0 import Memory

# ============================================================
# SECTION 1: MEM0 CONFIGURATION
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
# SECTION 2: INITIALISE MEM0
# ============================================================
# Memory.from_config() reads our config and sets up:
#   1. The FAISS vector store (created fresh or loaded from disk)
#   2. The sentence-transformer embedder
#   3. The LLM Mem0 uses internally to extract facts
# ============================================================

print("Initialising memory store (first run may download embedder ~90 MB)...")
memory = Memory.from_config(MEM0_CONFIG)
print("Memory store ready.\n")

# ============================================================
# SECTION 3: FIXED USER ID
# ============================================================
# Mem0 stores memories per user so different users stay separate.
# For this single-user chat we hardcode one ID.
# Change this string to switch between different memory profiles.
# ============================================================

USER_ID = "thamarai"

# ============================================================
# SECTION 4: OLLAMA MODEL + STRANDS AGENT
# ============================================================
# Same OllamaModel setup as Challenges 1 & 2.
# The system_prompt tells the agent its role and that it has
# access to a memory context that will be injected each turn.
# ============================================================

ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b"
)

agent = Agent(
    model=ollama_model,
    system_prompt=(
        "You are a helpful, friendly AI assistant with persistent memory.\n"
        "At the start of each reply you will receive a 'Memory Context' block "
        "containing facts previously remembered about the user.\n"
        "Always use those facts when answering — never say you don't know "
        "something that appears in the memory context.\n"
        "If the user shares new personal information (name, age, job, hobby, "
        "preference, etc.) acknowledge it naturally in your reply."
    )
)

# ============================================================
# SECTION 5: HELPER — SAVE NEW MEMORY
# ============================================================
# After every user message we ask Mem0 to extract and store
# any new facts mentioned.
#
# IMPORTANT: add() takes user_id as a TOP-LEVEL kwarg.
# Do NOT put it inside filters={} — that causes TypeError.
# ============================================================

def save_to_memory(user_message: str, user_id: str) -> None:
    """Extract and persist facts from user_message into Mem0."""
    memory.add(
        messages=[{"role": "user", "content": user_message}],
        user_id=user_id                 # top-level kwarg — correct for add()
    )

# ============================================================
# SECTION 6: HELPER — BUILD MEMORY CONTEXT STRING
# ============================================================
# Before every agent call we search Mem0 for memories relevant
# to the current message and format them as plain text.
#
# IMPORTANT: search() requires user_id INSIDE filters={}.
# Passing it as a top-level kwarg raises ValueError.
# ============================================================

def get_memory_context(query: str, user_id: str, top_k: int = 5) -> str:
    """Search Mem0 for memories relevant to query, return formatted string."""
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

# ============================================================
# SECTION 7: HELPER — LIST ALL STORED MEMORIES (DEBUG)
# ============================================================
# Type "show memories" during chat to dump everything stored.
#
# IMPORTANT: get_all() also requires user_id INSIDE filters={}.
# ============================================================

def list_all_memories(user_id: str) -> None:
    """Print every memory stored for user_id."""
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
# SECTION 8: INTERACTIVE CHAT LOOP
# ============================================================
# Each iteration:
#   1. Get user input
#   2. Save the message to Mem0 (extract new facts)
#   3. Retrieve relevant memories for this message
#   4. Prepend the memory context to the message
#   5. Send the enriched message to the Strands agent
# ============================================================

print("=" * 58)
print("  Challenge 3: Memory Agent - Strands + Ollama + Mem0")
print("  Your conversations are remembered across sessions.")
print("  Type 'show memories' to see what I remember.")
print("  Type 'quit' or 'exit' to stop.")
print("=" * 58)
print()
print("Try saying:")
print("  - My name is Thamarai")
print("  - I am 28 years old and I love hiking")
print("  - What is my name?          <- tests recall")
print("  - What do you know about me?")
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

    # STEP 4 — call the Strands agent
    print("\nAssistant: ", end="", flush=True)
    agent(enriched_prompt)
    print()
