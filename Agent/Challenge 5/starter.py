# ============================================================
# Challenge 5 - Builders Skill Sprint
# MCP Chatbot using Strands SDK + Ollama (llama3.2:3b)
# ============================================================
#
# SETUP — run these commands once:
#
#   pip install strands-agents strands-agents-tools
#   pip install mcp
#
# No extra installs needed — 'mcp' comes with strands-agents.
#
# FILES IN THIS CHALLENGE:
#   mcp_server.py  — the local MCP server (4 tools)
#   starter.py     — this file (the agent + chat loop)
#
# HOW MCP WORKS HERE:
#   1. starter.py launches mcp_server.py as a subprocess
#   2. They communicate via stdin/stdout using MCP protocol
#   3. Strands MCPClient discovers the tools automatically
#   4. The agent calls tools exactly like @tool decorated ones
#
# MCP TOOLS AVAILABLE:
#   - calculator      : math expressions
#   - get_weather     : weather for a city
#   - age_calculator  : age from birth year
#   - unit_converter  : temperature / distance / weight
# ============================================================

import sys
import os
from strands import Agent
from strands.models.ollama import OllamaModel
from strands.tools.mcp import MCPClient
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# SECTION 1: LOCATE THE MCP SERVER FILE
# ============================================================
# mcp_server.py lives in the same folder as this file.
# We build the absolute path so the subprocess can find it
# regardless of which directory the user runs Python from.
# ============================================================

# __file__ is the path of starter.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_FILE = os.path.join(SCRIPT_DIR, "mcp_server.py")

# Verify the server file exists before trying to launch it
if not os.path.exists(SERVER_FILE):
    print(f"ERROR: mcp_server.py not found at {SERVER_FILE}")
    print("Make sure mcp_server.py is in the same folder as starter.py")
    sys.exit(1)


# ============================================================
# SECTION 2: OLLAMA MODEL
# ============================================================
# Same OllamaModel setup used in all previous challenges.
# The model receives tool descriptions from MCP automatically.
# ============================================================

ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b"
)


# ============================================================
# SECTION 3: MCP CLIENT + AGENT SETUP
# ============================================================
# MCPClient wraps the MCP server subprocess.
#
# How the transport works:
#   StdioServerParameters tells MCPClient to launch
#   mcp_server.py as a child process using Python.
#
#   stdio_client is the transport callable — it creates
#   the stdin/stdout pipes between this process and the server.
#
# MCPClient is used as an async context manager (with block).
# Inside the block:
#   - The server subprocess is running
#   - client.list_tools_sync() fetches all tool definitions
#   - We pass those tool objects to the Strands Agent
#
# When the with block ends, the subprocess is cleanly stopped.
# ============================================================

# Build the stdio transport parameters
# sys.executable = full path to the current Python interpreter
# This ensures we use the same Python environment (with mcp installed)
server_params = StdioServerParameters(
    command=sys.executable,         # e.g. C:\...\python.exe
    args=[SERVER_FILE],             # launch mcp_server.py
    env=None                        # inherit current environment
)

# The transport callable — MCPClient calls this to open the pipes
def mcp_transport():
    return stdio_client(server_params)


# ============================================================
# SECTION 4: INTERACTIVE CHAT LOOP (inside MCP context)
# ============================================================
# Everything that uses the MCP tools must run INSIDE the
# MCPClient context manager because:
#   - The server subprocess only lives inside the with block
#   - Tools are only valid while the server is running
#
# Inside the block we:
#   1. List all tools exposed by the MCP server
#   2. Create the Strands Agent with those tools
#   3. Run the interactive chat loop
# ============================================================

print("Starting MCP server subprocess...")

with MCPClient(mcp_transport) as client:

    # --------------------------------------------------------
    # Discover tools from the MCP server
    # list_tools_sync() sends a tools/list request to the server
    # and returns Strands-compatible tool objects.
    # The agent will read their docstrings to know when to call them.
    # --------------------------------------------------------
    mcp_tools = client.list_tools_sync()

    print(f"MCP server ready. {len(mcp_tools)} tools loaded:")
    for t in mcp_tools:
        # Each tool object has a .tool_name attribute
        print(f"  - {t.tool_name}")
    print()

    # --------------------------------------------------------
    # Create the Strands Agent with all MCP tools
    # The system prompt tells the agent what tools it has
    # and encourages it to use them for relevant questions.
    # --------------------------------------------------------
    agent = Agent(
        model=ollama_model,
        tools=mcp_tools,            # MCP tools plug in just like @tool ones
        system_prompt=(
            "You are a helpful AI assistant connected to a local MCP server.\n\n"
            "You have access to these MCP tools:\n"
            "  1. calculator      - evaluate math expressions\n"
            "  2. get_weather     - weather info for a city\n"
            "  3. age_calculator  - calculate age from birth year\n"
            "  4. unit_converter  - convert between units\n\n"
            "Always use the right tool when the user's question matches one.\n"
            "Give clear, friendly answers."
        )
    )

    # --------------------------------------------------------
    # Print the welcome banner and example prompts
    # --------------------------------------------------------
    print("=" * 62)
    print("  Challenge 5: MCP Chatbot - Strands + Ollama + MCP")
    print("  Local MCP server running with 4 tools.")
    print("  Type 'list tools' to see available MCP tools.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 62)
    print()
    print("Try asking:")
    print("  - What is sqrt(256) + 10 * 3?")
    print("  - What's the weather in Tokyo?")
    print("  - How old is someone born in 1992?")
    print("  - Convert 100 celsius to fahrenheit")
    print("  - Convert 10 km to miles")
    print()

    # --------------------------------------------------------
    # Main chat loop
    # --------------------------------------------------------
    while True:
        user_input = input("You: ").strip()

        # Exit
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        # Skip blank input
        if not user_input:
            continue

        # Debug: list all available MCP tools
        if user_input.lower() == "list tools":
            print("\nAvailable MCP tools:")
            for t in mcp_tools:
                print(f"  - {t.tool_name}")
            print()
            continue

        # Send the message to the agent.
        # The agent decides whether to call an MCP tool or answer directly.
        # Tool calls go back to mcp_server.py over the stdio pipe.
        print("\nAssistant: ", end="", flush=True)
        agent(user_input)
        print()

# ============================================================
# When the with block exits, MCPClient sends a shutdown
# signal to mcp_server.py and closes the stdio pipes.
# The subprocess terminates cleanly.
# ============================================================
print("\nMCP server stopped. Session ended.")
