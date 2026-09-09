"""Graph wiring: a minimal tool-calling agent loop.

    START -> agent -> (tools -> agent)* -> END

`agent` calls the model. If the reply contains tool calls, `tools_condition`
routes to the `tools` node, which executes them and loops back so the model can
read the results. Otherwise the graph ends.
"""

import os

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.state import AgentState
from app.tools import TOOLS

# Override with OPENAI_MODEL to use a different model.
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")

SYSTEM_PROMPT = (
    "You are a concise assistant with tools. Use them instead of relying on "
    "memory:\n"
    "- calculator / word_count for arithmetic and text statistics.\n"
    "- web_search for anything current, or any fact you are unsure of. Cite the "
    "URLs you used.\n"
    "- database_schema then database_query for questions about customers, "
    "products or orders. Always read the schema before writing SQL.\n"
    "- search_documents for anything that might be in the user's own loaded "
    "files. Prefer it over web_search, quote what you find, and name the file."
)


def build_model() -> ChatOpenAI:
    return ChatOpenAI(model=MODEL, temperature=0).bind_tools(TOOLS)


def build_graph(checkpointer=None):
    """Compile the agent graph.

    Pass a checkpointer (e.g. `InMemorySaver()`) to keep conversation history
    per `thread_id`; without one each invocation starts fresh.
    """
    model = build_model()

    def agent(state: AgentState) -> dict:
        # print(f"  state[messages]-------------{state['messages']}")
        response = model.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")
    # tools_condition sends us to "tools" when the model requested a tool call,
    # and to END otherwise.
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


# Module-level graph for `langgraph dev` / LangGraph Studio.
graph = build_graph(checkpointer=InMemorySaver())
