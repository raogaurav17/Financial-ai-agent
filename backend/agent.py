"""Optional LangGraph/Gemini integration."""

from typing import Annotated, Any, TypedDict

from config import GOOGLE_API_KEY
from finance import assess_risk, fetch_market_data, predict_trends, rebalance_portfolio, simulate_portfolio


class AgentUnavailableError(RuntimeError):
    """Raised when an LLM request is made without Gemini configuration."""


class AgentServiceError(RuntimeError):
    """Raised when the configured LLM service cannot complete a request."""


def ensure_available() -> None:
    if not GOOGLE_API_KEY:
        raise AgentUnavailableError("LLM requests require GOOGLE_API_KEY to be configured.")


def _graph():
    ensure_available()
    from langchain_core.messages import SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    class AgentState(TypedDict):
        messages: Annotated[list[Any], add_messages]

    tools = [fetch_market_data, assess_risk, predict_trends, simulate_portfolio, rebalance_portfolio]
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=GOOGLE_API_KEY)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        messages = [SystemMessage(content="You are a helpful financial assistant. Use finance tools for factual data, then provide a concise explanation. Never return an empty response."), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def should_continue(state: AgentState):
        return "call_tool" if getattr(state["messages"][-1], "tool_calls", None) else "end"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", should_continue, {"call_tool": "tools", "end": END})
    builder.add_edge("tools", "agent")
    return builder.compile()


def invoke(query: str) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage
    try:
        return _graph().invoke({"messages": [HumanMessage(content=query)]})
    except AgentUnavailableError:
        raise
    except Exception as exc:
        raise AgentServiceError(f"LLM service failed: {exc}") from exc


def stream(query: str):
    from langchain_core.messages import HumanMessage
    try:
        yield from _graph().stream({"messages": [HumanMessage(content=query)]})
    except AgentUnavailableError:
        raise
    except Exception as exc:
        raise AgentServiceError(f"LLM service failed: {exc}") from exc
