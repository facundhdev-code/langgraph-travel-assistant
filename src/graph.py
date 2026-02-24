"""Travel Assistant - Graph assembly and compilation."""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import State
from .agents import tool_node
from .nodes import (
    planner_node,
    executor_node,
    web_researcher_node,
    activities_researcher_node,
    advance_step_node,
    synthesizer_node,
    route_after_executor,
    should_use_tools,
    route_after_tool,
    route_after_step
)


def build_graph(checkpointer=None):
    builder = StateGraph(State)
    
    # -- NODES --
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("web_researcher", web_researcher_node)
    builder.add_node("activities_researcher", activities_researcher_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("advance_step", advance_step_node)
    builder.add_node("synthesizer", synthesizer_node)
    
    # -- EDGES --
    builder.add_edge(START, 'planner')
    builder.add_edge('planner', 'executor')
    
    builder.add_conditional_edges(
        'executor',
        route_after_executor,
        ['web_researcher', 'activities_researcher', 'planner']
    )
    
    builder.add_conditional_edges(
        'web_researcher',
        should_use_tools,
        ['tool_node', 'advance_step']
    )
    
    builder.add_conditional_edges(
        'activities_researcher',
        should_use_tools,
        ['tool_node', 'advance_step']
    )
    
    builder.add_conditional_edges(
        'tool_node',
        route_after_tool,
        ['web_researcher', 'activities_researcher']
    )
    
    builder.add_conditional_edges(
        'advance_step',
        route_after_step,
        ['executor', 'synthesizer']
    )
    
    builder.add_edge('synthesizer', END)
    
    return builder.compile(checkpointer=checkpointer)

graph = build_graph(checkpointer=MemorySaver())