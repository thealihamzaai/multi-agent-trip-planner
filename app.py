import os
import re
from datetime import datetime
from pathlib import Path

import requests
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is missing. Add it to your .env file.")

llm = LLM(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.3)


@tool("Tavily Travel Search")
def tavily_search(query: str) -> str:
    """Search the web for current travel information and return source-linked results."""
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": 5,
        "include_answer": True,
        "search_depth": "advanced",
    }
    try:
        response = requests.post("https://api.tavily.com/search", json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return f"Search unavailable: {exc}"

    parts = []
    if data.get("answer"):
        parts.append(f"Summary: {data['answer']}")
    for idx, result in enumerate(data.get("results", [])[:5], 1):
        parts.append(
            f"{idx}. {result.get('title', 'Untitled')}\n"
            f"   {result.get('content', '')[:350]}\n"
            f"   Source: {result.get('url', 'N/A')}"
        )
    return "\n\n".join(parts) if parts else "No relevant travel results found."


@tool("Budget Calculator")
def calculate_budget(expenses: str) -> str:
    """Calculate a total from comma-separated label:value expense pairs."""
    try:
        total = 0.0
        breakdown = []
        for item in expenses.split(","):
            if ":" not in item:
                continue
            label, value = item.split(":", 1)
            amount = float(re.sub(r"[^0-9.\-]", "", value.strip()))
            total += amount
            breakdown.append(f"- {label.strip()}: {amount:.2f}")
        if not breakdown:
            return "No valid expenses were provided."
        return "\n".join(breakdown) + f"\n\nEstimated total: {total:.2f}"
    except ValueError:
        return "Budget calculation failed because one or more values were not numeric."


def build_agents():
    researcher = Agent(
        role="Destination Researcher",
        goal="Research current, practical destination information with source links.",
        backstory="A careful travel researcher who distinguishes verified facts from estimates.",
        tools=[tavily_search],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    budget = Agent(
        role="Travel Budget Analyst",
        goal="Build a realistic travel budget using sourced estimates and transparent assumptions.",
        backstory="A travel-cost analyst focused on clear assumptions rather than false precision.",
        tools=[tavily_search, calculate_budget],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    planner = Agent(
        role="Itinerary Planner",
        goal="Create a practical day-by-day itinerary aligned with the traveler's preferences.",
        backstory="An itinerary specialist who balances attractions, pacing, logistics, and free time.",
        tools=[tavily_search],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    booking = Agent(
        role="Booking Research Assistant",
        goal="Identify reputable places to compare flights and stays without claiming live availability unless verified.",
        backstory="A cautious booking-research assistant who treats prices and availability as time-sensitive.",
        tools=[tavily_search],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    return researcher, budget, planner, booking


def safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return value.strip("_") or "trip"


def plan_trip(destination: str, duration: str, budget_range: str, preferences: str) -> dict:
    researcher, budget_analyst, itinerary_planner, booking_assistant = build_agents()

    research_task = Task(
        description=(
            f"Research {destination} for a {duration} trip. Cover best time to visit, major attractions, "
            "local culture, practical transport, safety considerations, and a few lesser-known options. "
            "Use current sources and include source URLs. Do not invent facts."
        ),
        expected_output="A concise, source-linked destination research brief.",
        agent=researcher,
    )
    budget_task = Task(
        description=(
            f"Prepare a realistic budget outline for {destination} for {duration}. User budget: {budget_range}. "
            f"Travel preferences: {preferences}. Use sourced estimates where possible, state assumptions, "
            "and avoid presenting changing prices as guaranteed."
        ),
        expected_output="A practical cost breakdown with assumptions and savings notes.",
        agent=budget_analyst,
        context=[research_task],
    )
    itinerary_task = Task(
        description=(
            f"Create a day-by-day itinerary for {destination} lasting {duration}. Preferences: {preferences}. "
            "Keep the schedule realistic, group nearby activities, and allow sensible travel/rest time."
        ),
        expected_output="A clear day-by-day itinerary with practical pacing.",
        agent=itinerary_planner,
        context=[research_task, budget_task],
    )
    booking_task = Task(
        description=(
            f"Research reputable options for comparing flights and accommodation for {destination}. "
            f"Trip duration: {duration}. Budget: {budget_range}. Provide source links. Clearly state that "
            "prices and availability must be rechecked before booking. Do not claim a booking has been made."
        ),
        expected_output="A short list of booking-research options with links and verification caveats.",
        agent=booking_assistant,
        context=[research_task, budget_task, itinerary_task],
    )

    crew = Crew(
        agents=[researcher, budget_analyst, itinerary_planner, booking_assistant],
        tasks=[research_task, budget_task, itinerary_task, booking_task],
        process=Process.sequential,
        verbose=False,
    )
    crew.kickoff()

    return {
        "trip_details": {
            "destination": destination,
            "duration": duration,
            "budget": budget_range,
            "preferences": preferences,
        },
        "research": getattr(research_task.output, "raw", str(research_task.output)),
        "budget": getattr(budget_task.output, "raw", str(budget_task.output)),
        "itinerary": getattr(itinerary_task.output, "raw", str(itinerary_task.output)),
        "booking_research": getattr(booking_task.output, "raw", str(booking_task.output)),
    }


def save_report(result: dict, output_dir: str = "outputs") -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    details = result["trip_details"]
    path = Path(output_dir) / f"trip_plan_{safe_slug(details['destination'])}.md"
    content = f"""# Multi-Agent Trip Plan

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Trip details

- **Destination:** {details['destination']}
- **Duration:** {details['duration']}
- **Budget:** {details['budget']}
- **Preferences:** {details['preferences']}

## Destination research

{result['research']}

## Budget analysis

{result['budget']}

## Itinerary

{result['itinerary']}

## Booking research

{result['booking_research']}

## Important note

Travel prices, schedules, entry requirements, and availability can change. Recheck all critical information with official providers before booking or traveling.
"""
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    print("\nMulti-Agent Trip Planner\n")
    destination = input("Destination: ").strip() or "Bali, Indonesia"
    duration = input("Duration (e.g. 7 days): ").strip() or "7 days"
    budget_range = input("Budget (e.g. 2000 USD): ").strip() or "2000 USD"
    preferences = input("Travel preferences: ").strip() or "culture, food, and nature"

    result = plan_trip(destination, duration, budget_range, preferences)
    print("\n--- Destination Research ---\n" + result["research"])
    print("\n--- Budget ---\n" + result["budget"])
    print("\n--- Itinerary ---\n" + result["itinerary"])
    print("\n--- Booking Research ---\n" + result["booking_research"])
    path = save_report(result)
    print(f"\nTrip plan saved to: {path}")


if __name__ == "__main__":
    main()
