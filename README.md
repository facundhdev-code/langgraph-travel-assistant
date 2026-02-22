# LangGraph Travel Assistant

Multi-agent travel planning assistant built with LangGraph and LangChain. Uses a planner-executor architecture to orchestrate specialized AI agents that research destinations, activities, and travel logistics to generate personalized travel itineraries.

## Architecture

```
START → Planner → Executor ──→ web_researcher (flights, weather, visa)
                     │     ──→ activities_researcher (things to do, food, culture)
                     │
                     └───────→ Synthesizer → END (final itinerary)
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/facundhdev-code/langgraph-travel-assistant.git
cd langgraph-travel-assistant
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Run:
```bash
python -m src.main
```

## Tech Stack

- **LangGraph** — Multi-agent orchestration
- **LangChain** — LLM framework
- **OpenAI GPT-4o-mini** — Language model
- **Tavily** — Web search tool
