# Multi-Agent Trip Planner

A CrewAI-based travel planning workflow that coordinates specialized agents for destination research, budget planning, itinerary design, and booking research.

## What it does

- Researches a destination using live Tavily web search
- Builds a travel budget with explicit assumptions
- Creates a day-by-day itinerary
- Researches reputable flight and accommodation comparison options
- Saves the final plan as a Markdown report

## Agent workflow

1. **Destination Researcher** — current destination facts, attractions, logistics, and safety
2. **Budget Analyst** — cost estimates and savings opportunities
3. **Itinerary Planner** — realistic day-by-day schedule
4. **Booking Research Assistant** — source-linked places to compare flights and stays

## Tech stack

- Python
- CrewAI
- OpenAI-compatible LLM via CrewAI
- Tavily Search API
- Requests

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Add your API keys to `.env`, then run:

```bash
python app.py
```

## Safety and limitations

- Travel prices, schedules, visa/entry rules, and availability change frequently.
- Booking research is informational only; the app does not make reservations.
- Verify critical information with airlines, hotels, official tourism authorities, and government sources.
- Never commit `.env` or API keys to GitHub.

## Portfolio context

This project demonstrates multi-agent orchestration, web-grounded research, context passing between tasks, tool use, planning, and structured report generation with CrewAI.
