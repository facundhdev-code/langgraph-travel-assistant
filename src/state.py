"""Travel Assistant - State definition."""
from typing import Any, Dict, List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    plan: Dict[str, Any]
    current_step: int
    agent_query: str
    replan_flag: bool
    replan_attempts: Dict[int, int]
    final_answer: str
    travel_date: str
    trip_duration: str
    rag_context:str