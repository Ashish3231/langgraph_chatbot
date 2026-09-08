"""Graph state definition."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State passed between nodes.

    `messages` uses the `add_messages` reducer, so nodes return only the new
    messages they produced and LangGraph appends them to the running list.
    """

    messages: Annotated[list, add_messages]
