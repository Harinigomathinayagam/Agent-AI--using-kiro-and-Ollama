# ============================================================
# Challenge 2 - Builders Skill Sprint
# Tools Agent using Strands SDK + Ollama (llama3.2:3b)
# Tools: Calculator | Weather | Age Calculator
# ============================================================

import math
from datetime import date
from strands import Agent, tool
from strands.models.ollama import OllamaModel


# ============================================================
# TOOL 1: CALCULATOR
# ============================================================
# The @tool decorator turns a regular Python function into a
# tool the agent can call automatically when needed.
# The docstring is IMPORTANT — the agent reads it to understand
# what the tool does and when to use it.
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Supports: +, -, *, /, ** (power), sqrt(), and basic math.
    Example: '2 + 2', '10 * 5', 'sqrt(144)', '2 ** 8'
    """
    try:
        # Allow only safe math functions using math module
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# ============================================================
# TOOL 2: WEATHER
# ============================================================
# This is a simulated weather tool (no real API needed).
# In a real project you'd call something like OpenWeatherMap.
# The agent will call this when the user asks about weather.
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    Returns temperature, condition, humidity and wind speed.
    Example cities: London, New York, Tokyo, Sydney, Paris
    """
    # Simulated weather data — replace with a real API if you want
    weather_data = {
        "london":   {"temp": 15, "condition": "Cloudy",  "humidity": 78, "wind": 12},
        "new york": {"temp": 22, "condition": "Sunny",   "humidity": 55, "wind": 8},
        "tokyo":    {"temp": 28, "condition": "Humid",   "humidity": 85, "wind": 5},
        "sydney":   {"temp": 20, "condition": "Partly Cloudy", "humidity": 60, "wind": 14},
        "paris":    {"temp": 18, "condition": "Rainy",   "humidity": 80, "wind": 10},
        "dubai":    {"temp": 38, "condition": "Sunny",   "humidity": 40, "wind": 6},
        "mumbai":   {"temp": 32, "condition": "Humid",   "humidity": 90, "wind": 7},
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


# ============================================================
# TOOL 3: AGE CALCULATOR
# ============================================================
# Calculates a person's exact age from their birth year.
# The agent calls this when asked "how old is someone born in X"
# ============================================================

@tool
def age_calculator(birth_year: int) -> str:
    """
    Calculate the current age of a person given their birth year.
    Also tells how many years until their next milestone birthday
    (e.g., 30, 40, 50...).
    Example: birth_year = 1995
    """
    current_year = date.today().year

    if birth_year < 1900 or birth_year > current_year:
        return f"Invalid birth year: {birth_year}. Please enter a year between 1900 and {current_year}."

    age = current_year - birth_year

    # Find next milestone (every 10 years)
    next_milestone = ((age // 10) + 1) * 10
    years_to_milestone = next_milestone - age

    return (
        f"Age Calculation:\n"
        f"  Birth Year       : {birth_year}\n"
        f"  Current Year     : {current_year}\n"
        f"  Current Age      : {age} years old\n"
        f"  Next Milestone   : {next_milestone} (in {years_to_milestone} year(s))"
    )


# ============================================================
# AGENT SETUP
# ============================================================
# Connect all three tools to the agent.
# The agent will automatically decide which tool to call
# based on what the user asks.
# ============================================================

ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b"
)

agent = Agent(
    model=ollama_model,
    tools=[calculator, get_weather, age_calculator],
    system_prompt=(
        "You are a helpful assistant with access to three tools:\n"
        "1. calculator   — for any math or arithmetic questions\n"
        "2. get_weather  — for weather information about a city\n"
        "3. age_calculator — for calculating someone's age from birth year\n\n"
        "Always use the right tool when the user's question matches one of these. "
        "Give clear, friendly answers."
    )
)


# ============================================================
# INTERACTIVE CHAT LOOP
# ============================================================

print("=" * 55)
print("  Challenge 2: Tools Agent — Strands + Ollama")
print("  Tools: Calculator | Weather | Age Calculator")
print("  Type 'quit' or 'exit' to stop")
print("=" * 55)
print()
print("Try asking:")
print("  - What is 25 * 48?")
print("  - What's the weather in Tokyo?")
print("  - How old is someone born in 1990?")
print()

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    if not user_input:
        continue

    print("\nAssistant: ", end="", flush=True)
    agent(user_input)
    print()
