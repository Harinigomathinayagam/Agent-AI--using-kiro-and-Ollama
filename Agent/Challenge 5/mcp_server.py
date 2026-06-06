# ============================================================
# Challenge 5 - MCP Server
# A local MCP server built with FastMCP.
# This file is launched as a subprocess by starter.py —
# you never run it directly.
#
# FastMCP exposes Python functions as MCP tools over stdio.
# The Strands MCPClient connects to it via stdin/stdout.
# ============================================================

import math
from datetime import date
from mcp.server.fastmcp import FastMCP

# Create the MCP server and give it a name
mcp = FastMCP("LocalToolsServer")


# ============================================================
# MCP TOOL 1: CALCULATOR
# ============================================================
# The @mcp.tool() decorator registers this function as an
# MCP tool. The docstring becomes the tool description that
# the agent reads to decide when to call it.
# ============================================================

@mcp.tool()
def calculator(expression: str) -> str:
    """
    Evaluate a math expression and return the result.
    Supports: +, -, *, /, ** (power), sqrt(), pi, e, etc.
    Examples: '2 + 2', '10 * 5', 'sqrt(144)', '2 ** 10'
    """
    try:
        # Restrict eval to math functions only — safe, no builtins
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# ============================================================
# MCP TOOL 2: WEATHER
# ============================================================
# Simulated weather data — no API key needed.
# In a real project you would call OpenWeatherMap or similar.
# ============================================================

@mcp.tool()
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.
    Returns temperature (°C), condition, humidity, and wind speed.
    Available: London, New York, Tokyo, Sydney, Paris,
               Dubai, Mumbai, Chennai, Bangalore, Delhi.
    """
    data = {
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
    if key in data:
        w = data[key]
        return (
            f"Weather in {city.title()}:\n"
            f"  Temperature : {w['temp']}°C\n"
            f"  Condition   : {w['condition']}\n"
            f"  Humidity    : {w['humidity']}%\n"
            f"  Wind Speed  : {w['wind']} km/h"
        )
    available = ", ".join(c.title() for c in data)
    return f"No data for '{city}'. Available cities: {available}"


# ============================================================
# MCP TOOL 3: AGE CALCULATOR
# ============================================================

@mcp.tool()
def age_calculator(birth_year: int) -> str:
    """
    Calculate a person's age from their birth year.
    Also shows years remaining until the next milestone birthday
    (30, 40, 50, 60, etc.).
    Example: birth_year = 1995
    """
    current_year = date.today().year
    if birth_year < 1900 or birth_year > current_year:
        return f"Invalid year {birth_year}. Use a year between 1900 and {current_year}."
    age = current_year - birth_year
    next_milestone = ((age // 10) + 1) * 10
    years_left = next_milestone - age
    return (
        f"Age Details:\n"
        f"  Birth Year     : {birth_year}\n"
        f"  Current Year   : {current_year}\n"
        f"  Age            : {age} years old\n"
        f"  Next Milestone : {next_milestone} (in {years_left} year(s))"
    )


# ============================================================
# MCP TOOL 4: UNIT CONVERTER
# ============================================================
# Extra tool to show MCP can expose any kind of helper.
# Converts between common units.
# ============================================================

@mcp.tool()
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a value between common units.
    Supported conversions:
      Temperature : celsius <-> fahrenheit, celsius <-> kelvin
      Distance    : km <-> miles, meters <-> feet
      Weight      : kg <-> pounds
    Example: value=100, from_unit='celsius', to_unit='fahrenheit'
    """
    f = from_unit.lower().strip()
    t = to_unit.lower().strip()
    key = (f, t)

    conversions = {
        ("celsius",    "fahrenheit") : lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius")    : lambda v: (v - 32) * 5/9,
        ("celsius",    "kelvin")     : lambda v: v + 273.15,
        ("kelvin",     "celsius")    : lambda v: v - 273.15,
        ("km",         "miles")      : lambda v: v * 0.621371,
        ("miles",      "km")         : lambda v: v * 1.60934,
        ("meters",     "feet")       : lambda v: v * 3.28084,
        ("feet",       "meters")     : lambda v: v * 0.3048,
        ("kg",         "pounds")     : lambda v: v * 2.20462,
        ("pounds",     "kg")         : lambda v: v * 0.453592,
    }

    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return (
        f"Conversion from '{from_unit}' to '{to_unit}' not supported.\n"
        f"Supported pairs: celsius/fahrenheit, celsius/kelvin, "
        f"km/miles, meters/feet, kg/pounds (and their reverses)."
    )


# ============================================================
# ENTRY POINT
# ============================================================
# mcp.run(transport="stdio") starts the server and listens
# on stdin/stdout for JSON-RPC messages from the MCP client.
#
# IMPORTANT: Do NOT run this file directly in a terminal.
#   WRONG : python mcp_server.py   <-- causes JSON parse errors
#   RIGHT : python starter.py      <-- starter.py launches this
#                                       as a managed subprocess
#
# When run directly, stdin receives empty input (no JSON),
# which causes the "EOF while parsing" validation error.
# The server is only meant to talk to an MCP client, not a human.
# ============================================================

if __name__ == "__main__":
    import sys

    # Safety check: warn if someone accidentally runs this directly.
    # A real MCP client will immediately send an "initialize" JSON
    # message on stdin. If stdin is a TTY (interactive terminal),
    # no client is connected — print a helpful message and exit.
    if sys.stdin.isatty():
        print("=" * 60)
        print("  mcp_server.py — Local MCP Server")
        print("=" * 60)
        print()
        print("  Do NOT run this file directly.")
        print()
        print("  This server communicates via stdin/stdout JSON-RPC.")
        print("  It must be launched as a subprocess by starter.py.")
        print()
        print("  To start the chatbot, run:")
        print("      python starter.py")
        print()
        sys.exit(0)

    # Started by an MCP client (e.g. MCPClient in starter.py)
    # stdin is a pipe, not a TTY — safe to start the server.
    mcp.run(transport="stdio")
