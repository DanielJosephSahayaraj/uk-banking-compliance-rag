from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
import operator

from nodes import (
    guardrail_node,
    cache_check_node,
    router_node,
    date_tool_node,
    web_search_node,
    rewrite_node,
    hyde_node,
    retrieve_node,
    summarize_history_node,
    generate_node,
    compliance_check_node,
    critique_node,
    retry_node,
    document_compliance_node,
    direct_answer_node,
    end_node
)


# ──────────────────────────────────────────
# STATE DEFINITION
# ──────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    rewrite_query: Optional[str]
    hyde_answer: Optional[str]
    retrieved_chunks: list
    context: str
    final_response: str
    critique: str
    compliance: Optional[dict]
    messages: Annotated[list[BaseMessage], operator.add]
    document_text: Optional[str]


# ──────────────────────────────────────────
# BUILD GRAPH
# ──────────────────────────────────────────
workflow = StateGraph(AgentState)

# ── Add all nodes ──
workflow.add_node("guardrail",          guardrail_node)
workflow.add_node("cache_check",        cache_check_node)
workflow.add_node("router",             router_node)
workflow.add_node("date_tool",          date_tool_node)
workflow.add_node("web_search",         web_search_node)
workflow.add_node("rewrite",            rewrite_node)
workflow.add_node("hyde",               hyde_node)
workflow.add_node("retrieve",           retrieve_node)
workflow.add_node("summarize_history",  summarize_history_node)
workflow.add_node("generate",           generate_node)
workflow.add_node("compliance_check",   compliance_check_node)
workflow.add_node("critique",           critique_node)
workflow.add_node("retry",              retry_node)
workflow.add_node("document_compliance",document_compliance_node)
workflow.add_node("direct_answer",      direct_answer_node)
workflow.add_node("end",                end_node)

# ── Entry point ──
workflow.set_entry_point("guardrail")

# ── Edges ──
workflow.add_conditional_edges(
    "guardrail",
    lambda s: s["next"],
    {
        "end":        "end",
        "cache_check": "cache_check"
    }
)

workflow.add_conditional_edges(
    "cache_check",
    lambda s: s["next"],
    {
        "end":      "end",
        "router":   "router",
        "date_tool": "date_tool"
    }
)

workflow.add_conditional_edges(
    "router",
    lambda s: s["next"],
    {
        "rewrite":              "rewrite",
        "direct_answer":        "direct_answer",
        "web_search":           "web_search",
        "document_compliance":  "document_compliance",
        "date_tool":            "date_tool"
    }
)

workflow.add_edge("rewrite",            "hyde")
workflow.add_edge("hyde",               "retrieve")
workflow.add_edge("retrieve",           "summarize_history")
workflow.add_edge("summarize_history",  "generate")
workflow.add_edge("generate",           "compliance_check")

workflow.add_conditional_edges(
    "compliance_check",
    lambda s: s["next"],
    {
        "critique": "critique",
        "end":      "end"
    }
)

workflow.add_conditional_edges(
    "critique",
    lambda s: s["next"],
    {
        "end":   "end",
        "retry": "retry"
    }
)

workflow.add_edge("retry",              "end")
workflow.add_edge("direct_answer",      "end")
workflow.add_edge("date_tool",          "end")
workflow.add_edge("web_search",         "end")
workflow.add_edge("document_compliance","end")

# ── Compile ──
app = workflow.compile()
print("Graph compiled successfully!")