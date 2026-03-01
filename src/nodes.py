"""Travel Assistant - Graph nodes."""
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .rag import retrieve_context

from .state import State
from .agents import (
    create_planner_agent,
    create_executor_agent,
    create_activities_researcher_llm,
    create_web_researcher_llm,
    create_synthesizer_agent,
    tool_node
)

from .prompts import WEB_RESEARCHER_SYSTEM, ACTIVITIES_RESEARCHER_SYSTEM
MAX_REPLAN_ATTEMPTS = 2

# --- NODES --- 
def planner_node(state:State) -> dict:
    planner = create_planner_agent()
    travel_date = state.get('travel_date', '') or 'not specified'
    trip_duration = state.get('trip_duration', '') or 'not specified'
    plan = planner.invoke({'user_query': state['user_query'], 'travel_date': travel_date, 'trip_duration':trip_duration })
    
    return {
        'plan': plan.model_dump(),
        'current_step': 0,
        'replan_flag': False
    }

def executor_node(state: State) -> dict:
    executor = create_executor_agent()
    plan = state['plan']
    current_step = state['current_step']
    plan_step = plan['steps'][current_step]
    
    previous_results = '\n'.join(
        msg.content
        for msg in state['messages']
        if isinstance(msg, AIMessage) and msg.content
    )
    
    decision = executor.invoke({
        'messages': state['messages'],
        'current_step': current_step + 1,
        'plan_step' : str(plan_step),
        'previous_results': previous_results or 'No previous results yet.'
    })
    
    updates: dict = {
        'agent_query': decision.agent_query,
        'replan_flag': decision.needs_replan
    }
    
    if decision.needs_replan:
        replan_attempts = dict(state.get('replan_attempts', {}))
        replan_attempts[str(current_step)] = replan_attempts.get(str(current_step),0) + 1
        updates['replan_attempts'] = replan_attempts
        
    return updates

def web_researcher_node(state:State) -> dict:
    llm = create_web_researcher_llm()
    messages = [
        SystemMessage(content=WEB_RESEARCHER_SYSTEM)]
    if state.get('rag_context'):
        messages.append(SystemMessage(content=f"Curated destination info:\n{state['rag_context']}"))
    messages +=  [*state['messages'], HumanMessage(content=state['agent_query'])]  
    response = llm.invoke(messages)
    
    return {'messages': [response]}

def activities_researcher_node(state:State) -> dict: 
    llm = create_activities_researcher_llm()
    messages = [
        SystemMessage(content=ACTIVITIES_RESEARCHER_SYSTEM)]
    if state.get('rag_context'):
         messages.append(SystemMessage(content=f"Curated destination info:\n{state['rag_context']}"))
         
    messages += [*state['messages'], HumanMessage(content=state['agent_query'])]
    response = llm.invoke(messages)
    
    return {'messages': [response]}

def advance_step_node(state:State) -> dict:
    return {'current_step': state['current_step'] + 1}

def rag_retriever_node(state:State) -> Dict:
    context = retrieve_context(state['user_query'])
    return {'rag_context': context}

def synthesizer_node(state: State) -> dict:
    synthesizer = create_synthesizer_agent()
    travel_date = state.get('travel_date', '') or 'not specified'
    trip_duration = state.get('trip_duration', '') or 'not specified'
    research_summary = "\n\n".join(
        f'Research {i + 1}: \n{msg.content}'
        for i, msg in enumerate(state['messages'])
        if isinstance(msg, AIMessage) and msg.content
    )
    
    result = synthesizer.invoke({
        'messages': state['messages'],
        'user_query':state['user_query'],
        'travel_date': travel_date,
        'trip_duration': trip_duration,
        'research_summary': research_summary
    })
    
    return {'final_answer': result.content}

# --- ROUTING FUNCTIONS ---

def route_after_executor(state: State,) -> Literal['web_researcher', 'activities_researcher', 'planner']:
    if state.get('replan_flag'):
        current_step = state['current_step']
        attempts = state.get('replan_attempts', {}).get(str(current_step), 0)
        if attempts <= MAX_REPLAN_ATTEMPTS:
            return 'planner'
    plan = state['plan']
    current_step = state['current_step']
    return plan['steps'][current_step]['agent']

def should_use_tools(state:State) -> Literal['tool_node', 'advance_step']:
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return 'tool_node'
    return 'advance_step'

def route_after_tool(state:State) -> Literal['web_researcher', 'activities_researcher']:
    plan = state['plan']
    current_step = state['current_step']
    return plan['steps'][current_step]['agent']

def route_after_step(state:State) -> Literal['executor', 'synthesizer']:
    plan = state['plan']
    current_step = state['current_step']
    if current_step >= len(plan['steps']):
        return 'synthesizer'
    return 'executor'