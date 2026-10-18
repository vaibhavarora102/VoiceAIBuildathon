from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from models import HiringAgentState
from agent.nodes import analyzer_node, form_builder_node, route_after_analysis


def build_hiring_graph():
    """Builds and compiles the multi-turn LangGraph recruitment agent."""
    builder = StateGraph(HiringAgentState)
    builder.add_node("analyzer", analyzer_node)
    builder.add_node("form_builder", form_builder_node)

    builder.add_edge(START, "analyzer")
    builder.add_conditional_edges(
        "analyzer",
        route_after_analysis,
        {
            "form_builder": "form_builder",
            END: END
        }
    )
    builder.add_edge("form_builder", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


hiring_graph = build_hiring_graph()
