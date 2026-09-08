"""A simple state graph: three nodes in a straight line.

    START -> clean -> tokenize -> summarize -> END

No model, no tools — just how state flows. Each node receives the whole state
and returns a dict of *only the keys it changed*; LangGraph merges that into the
running state. Keys without a reducer (unlike `messages` in `app/state.py`) are
overwritten by whatever the node returns.
"""

import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class PipelineState(TypedDict, total=False):
    """State for the text pipeline.

    `total=False` means a node may return a subset of these keys. `text` is the
    input; the rest are filled in as the graph runs.
    """

    text: str
    words: list[str]
    stats: dict
    summary: str


def clean(state: PipelineState) -> dict:
    """Collapse whitespace so later nodes see a predictable string."""
    return {"text": " ".join(state["text"].split())}


def tokenize(state: PipelineState) -> dict:
    """Split into words and record a couple of counts."""
    words = [w.strip(".,!?;:\"'").lower() for w in state["text"].split()]
    words = [w for w in words if w]
    return {
        "words": words,
        "stats": {
            "words": len(words),
            "unique": len(set(words)),
            "characters": len(state["text"]),
        },
    }


def summarize(state: PipelineState) -> dict:
    """Turn the stats into a one-line report."""
    stats = state["stats"]
    longest = max(state["words"], key=len, default="")
    return {
        "summary": (
            f"{stats['words']} words ({stats['unique']} unique), "
            f"{stats['characters']} characters; longest word: {longest!r}"
        )
    }


def build_graph():
    """Compile the linear pipeline."""
    builder = StateGraph(PipelineState)
    builder.add_node("clean", clean)
    builder.add_node("tokenize", tokenize)
    builder.add_node("summarize", summarize)

    # A plain `add_edge` is an unconditional hand-off: the next node always runs.
    builder.add_edge(START, "clean")
    builder.add_edge("clean", "tokenize")
    builder.add_edge("tokenize", "summarize")
    builder.add_edge("summarize", END)

    return builder.compile()


graph = build_graph()

DEFAULT_TEXT = "LangGraph  models an agent as a  graph of nodes over shared state."


def main() -> None:
    text = " ".join(sys.argv[1:]) or DEFAULT_TEXT

    # `stream` yields one chunk per node, so you can watch the state fill in.
    for chunk in graph.stream({"text": text}, stream_mode="updates"):
        for node, update in chunk.items():
            print(f"{node:>10} -> {update}")

    print("\n" + graph.invoke({"text": text})["summary"])


if __name__ == "__main__":
    main()
