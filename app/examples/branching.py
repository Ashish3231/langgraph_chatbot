"""A graph with branching: one router picks between three paths, then they merge.

                         ┌──> compute ──┐
    START ──> classify ──┼──> analyze ──┼──> report ──> END
                         └──> reject  ──┘

`add_conditional_edges` takes a *router* — a plain function that reads the state
and returns the name of the next node. Only the branch it names runs; the others
are skipped. All three branches then `add_edge` into `report`, so the paths
rejoin without any extra bookkeeping.
"""

import re
import sys
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.tools import calculator, word_count

MATH_RE = re.compile(r"^[\d\s+\-*/%().]+$")


def append(left: list, right: list) -> list:
    """Reducer: nodes return the steps they took and LangGraph concatenates them.

    Same idea as `add_messages` in `app/state.py`, minus the message handling.
    """
    return left + right


class RouteState(TypedDict, total=False):
    """State for the routing example.

    `trail` carries a reducer, so every node can append to it without reading
    the previous value; the other keys are plain overwrites.
    """

    query: str
    kind: str
    result: str
    trail: Annotated[list[str], append]


def classify(state: RouteState) -> dict:
    """Label the query. This node only *records* the decision..."""
    query = state["query"].strip()
    if not query:
        kind = "empty"
    elif MATH_RE.match(query) and any(c.isdigit() for c in query):
        kind = "math"
    else:
        kind = "text"
    return {"kind": kind, "trail": [f"classified as {kind}"]}


def route(state: RouteState) -> str:
    """...and this router turns it into an edge.

    A router returns a node name (or END). Keeping it separate from `classify`
    means the decision is visible in the state and the routing stays a pure
    function of it.
    kind       → next node

    "math"     → "compute"
    "text"     → "analyze"
    anything   → "reject"
    """
    return {"math": "compute", "text": "analyze"}.get(state["kind"], "reject")


def compute(state: RouteState) -> dict:
    """Math branch — reuses the `calculator` tool from app/tools.py."""
    result = calculator.invoke({"expression": state["query"]})
    return {"result": result, "trail": ["compute"]}


def analyze(state: RouteState) -> dict:
    """Text branch — reuses the `word_count` tool."""
    result = word_count.invoke({"text": state["query"]})
    return {"result": result, "trail": ["analyze"]}


def reject(state: RouteState) -> dict:
    """Fallback branch for anything the classifier could not place."""
    return {"result": "nothing to do — give me an expression or some text", "trail": ["reject"]}


def report(state: RouteState) -> dict:
    """Join point: whichever branch ran, control lands here."""
    return {"trail": [f"reported via {' -> '.join(state['trail'][1:])}"]}


def build_graph():
    """Compile the branching graph."""
    builder = StateGraph(RouteState)
    for node in (classify, compute, analyze, reject, report):
        builder.add_node(node.__name__, node)

    builder.add_edge(START, "classify")
    # The third argument maps the router's return values to nodes. It is
    # optional when the router already returns node names, but passing it states
    # the reachable destinations up front, so the diagram (and anyone reading
    # this) sees the branches without having to infer them from `route`.
    builder.add_conditional_edges(
        "classify",
        route,
        {"compute": "compute", "analyze": "analyze", "reject": "reject"},
    )

    # Fan back in: every branch continues to the same node.
    for branch in ("compute", "analyze", "reject"):
        builder.add_edge(branch, "report")
    builder.add_edge("report", END)

    return builder.compile()


graph = build_graph()


def main() -> None:
    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "3 * (14 + 2) / 8",
        "how many words are in this sentence",
        "",
    ]

    for query in queries:
        state = graph.invoke({"query": query})
        print(f"{query!r}\n  {state['kind']}: {state['result']}\n  trail: {state['trail']}\n")


if __name__ == "__main__":
    main()
