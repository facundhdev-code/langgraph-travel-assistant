"""Travel Assistant - Agent configuration."""
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from .prompts import (
    PLANNER_PROMPT,
    EXECUTOR_PROMPT,
    SYNTHESIZER_PROMPT
)
  
# --- STRUCUTRED OUTPUT SCHEMAS ---

class PlanStep(BaseModel):
    step: int = Field(description='Step number')
    agent: str = Field(description='Agent to execute: web_researcher or activities_researcher')
    query: str = Field(description='Query to pass to the agent')
    description: str = Field(description='What this step researches')
    
class Plan(BaseModel):
    destination: str = Field(description='Travel destination')
    trip_summary: str = Field(description='Brief trip summary')
    steps: List[PlanStep] = Field(description='Ordered research steps')
    
class ExecutorDecision(BaseModel):
    agent_query: str = Field(description='Refined query for the agent')
    needs_replan: bool = Field(description='Whether to trigger a replan')
    replan_reason: str = Field(description='Reason for replanning if needed')
    
# --- TOOLS ---

search_tools = [TavilySearch(max_results=3)]
tool_node = ToolNode(search_tools)
    
# --- LLM ---

def get_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(model='gpt-4o-mini', temperature=temperature)

# --- AGENT FACTORIES ---
def create_planner_agent():
    llm = get_llm()
    return PLANNER_PROMPT | llm.with_structured_output(Plan)

def create_executor_agent():
    llm = get_llm()
    return EXECUTOR_PROMPT | llm.with_structured_output(ExecutorDecision)

def create_web_researcher_llm():
    llm = get_llm()
    return llm.bind_tools(search_tools)

def create_activities_researcher_llm():
    llm = get_llm()
    return llm.bind_tools(search_tools)

def create_synthesizer_agent():
    llm = get_llm(temperature=0.3)
    return SYNTHESIZER_PROMPT | llm