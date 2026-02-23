"""Travel Assistant - Agent configuration."""
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_agent

from .prompts import (
    PLANNER_PROMPT,
    EXECUTOR_PROMPT,
    WEB_RESEARCHER_SYSTEM,
    ACTIVITIES_RESEARCHER_SYSTEM,
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

def create_web_researcher_agent():
    llm = get_llm()
    tools = [TavilySearchResults(max_results=3)]
    return create_agent(llm, tools, prompt=WEB_RESEARCHER_SYSTEM)

def create_activities_researcher_agent():
    llm = get_llm()
    tools = [TavilySearchResults(max_results=3)]
    return create_agent(llm, tools, prompt=ACTIVITIES_RESEARCHER_SYSTEM)

def create_synthesizer_agent():
    llm = get_llm(temperature=0.3)
    return SYNTHESIZER_PROMPT | llm