"""
agent.py — LangGraph ReAct agent wiring tools, system prompt, and MCP.

Graph shape:
  user message → agent node (reason + pick tool) → tool node → loop → final reply

MCP filesystem server is started once at process startup (via lifespan in main.py)
and its tools are injected here.
"""

import os
import json
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from backend.app.tools import ALL_TOOLS

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an operations analyst agent for Darkstore's quick-commerce delivery operations.
You help ops users understand store performance and diagnose why OR2A SLAs were breached on a given date.

## Your Role
- Answer questions at city, store, or hour granularity
- Run root-cause analysis (RCA) using the provided playbook tools
- Maintain context across turns within a session (city, store, date, hour)

## Context Tracking Rules
- When a user references a new store but no date, inherit the date from previous turns
- When a user says "what about STORE_X", switch store but keep city and date
- When a user says "morning hours", interpret as hours 6–11 unless otherwise specified
- When a user says "evening hours", interpret as hours 17–22

## Tool Usage Rules — Follow These Strictly
1. For "how did X do?" or performance overview questions → use `query_performance`
2. For "why did STORE_X underperform?" or day-level root cause → use `run_store_day_rca`
3. For "why was hour X bad?" or single-hour root cause → use `run_hour_rca`
4. For "walk me through hours X to Y" or time-range questions → use `run_hour_range_rca`
5. For "which stores are in city X?" → use `list_stores`
6. For ad-hoc metric questions not covered above → use `query_sql` with a SELECT query
7. For metric definitions, threshold explanations, or "what is OR2A/pileup/etc.?" → use `read_file` to read from docs/
8. NEVER reason about root causes from memory — always call `run_store_day_rca` or `run_hour_rca`
9. NEVER fabricate numbers — if a tool returns no data, say so clearly

## RCA Output Format
When presenting RCA results, always follow this structure for each problem hour:

### [Store] — Hour [X] — avg OR2A: [Y] min (threshold: [Z] min)
1. Demand Spike: [YES/NO] — [total_orders] orders vs [order_projection] projected ([+/-%]%)
2. Pileup: [YES/NO] — [pileup_count] orders carried from previous hour [+ SUSTAINED if applicable]
3. Supply:
   a. Booking: [booked_size] of [current_size] slots booked ([X]%)
   b. Utilization: man_hour ratio [X] ([noshow_count] no-shows)
**Summary**: OR2A was [X] min over threshold due to [list of triggered flags in plain language].

Always answer all three checks even when the answer is NO.
The summary line must name every contributing factor in one sentence.

## Thresholds (for reference when explaining)
- OR2A SLA: ≤ 0 min (client-specific)
- Demand spike: actual orders > 110% of forecast
- Booking gap: < 90% of available slots filled
- Utilization gap: man_hour ratio < 0.85
- Pileup sustained: pileup in 3+ consecutive hours

## Tone
Be concise and direct. Ops users are busy. Lead with the headline finding, then detail.
When the user asks a follow-up, acknowledge the context switch naturally ("Looking at STORE_102 now...").
"""


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: dict


# ---------------------------------------------------------------------------
# Build the graph (called once per process, tools injected at build time)
# ---------------------------------------------------------------------------

def build_agent(extra_tools: list[BaseTool] | None = None):
    """
    Construct and compile the LangGraph ReAct agent using Gemini.
    """
    all_tools = ALL_TOOLS + (extra_tools or [])

    # CHANGED: Use ChatGoogleGenerativeAI
    # Note: Gemini models usually use "gemini-1.5-pro" or "gemini-1.5-flash"
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-1.5-pro"),
        temperature=0,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
    ).bind_tools(all_tools)

    tool_node = ToolNode(all_tools)

    def agent_node(state: AgentState):
        messages = list(state["messages"])
        ctx = state.get("context", {})

        context_parts = []
        if ctx.get("date"):
            context_parts.append(f"date={ctx['date']}")
        if ctx.get("city"):
            context_parts.append(f"city={ctx['city']}")
        if ctx.get("store"):
            context_parts.append(f"store={ctx['store']}")
        if ctx.get("hour") is not None:
            context_parts.append(f"hour={ctx['hour']}")

        system_content = SYSTEM_PROMPT
        if context_parts:
            system_content += f"\n\n## Current Session Context\n{', '.join(context_parts)}"

        # Gemini handles SystemMessages well, but ensure they are the first message
        full_messages = [SystemMessage(content=system_content)] + messages
        response = llm.invoke(full_messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# Context extraction helper
# ---------------------------------------------------------------------------

def extract_context_from_messages(messages: list[BaseMessage], current_context: dict) -> dict:
    """
    After a completed agent turn, scan the last user message for explicit
    context cues and update the sticky context dict.
    This is a lightweight heuristic — the LLM handles ambiguous cases via
    the system prompt context tracking rules.
    """
    import re

    ctx = dict(current_context)

    # Find the last human message
    last_human = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_human = m.content
            break

    if not last_human:
        return ctx

    text = last_human.lower()

    # Date: YYYY-MM-DD
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", last_human)
    if date_match:
        ctx["date"] = date_match.group()

    # Store: STORE_XXX (case-insensitive)
    store_match = re.search(r"store[_\s]?\d+", text)
    if store_match:
        # Normalise to STORE_XXX format
        digits = re.search(r"\d+", store_match.group())
        if digits:
            ctx["store"] = f"STORE_{digits.group()}"

    # Hour: "hour 10", "at 10", "hour 10am"
    hour_match = re.search(r"hour\s+(\d{1,2})", text)
    if hour_match:
        ctx["hour"] = int(hour_match.group(1))
    else:
        # "morning" / "evening" clears specific hour — range handled by agent
        if "morning" in text or "evening" in text or "afternoon" in text:
            ctx["hour"] = None

    return ctx