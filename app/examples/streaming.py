"""Streaming: watching a graph run instead of waiting for the final answer.

    START -> retrieve -> respond -> END

`graph.invoke(...)` hands you one dict when everything is finished. `graph.stream(...)`
returns a *generator* — a lazy sequence you loop over, which yields items while
the graph is still running. `stream_mode` decides what those items are:

    "updates"   what each node returned, as it finishes   -> progress, by node
    "values"    the whole state after each step           -> a running snapshot
    "messages"  model output token by token               -> the typing effect
    "custom"    whatever a node writes itself             -> your own progress
    "debug"     every internal event (very verbose)

Pass a *list* of modes and each item becomes a `(mode, chunk)` pair instead, so
one loop can do several of these at once — that is what `app/main.py` does.

Nothing here needs an API key: the "model" is `GenericFakeChatModel`, which
replays a canned reply one word at a time, the way a real one would.

    python -m app.examples.streaming            # run every mode in turn
    python -m app.examples.streaming messages   # or just one
"""

import sys
import time
from typing import Annotated, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# The canned reply. A real model would produce these words one at a time; the
# fake one splits on spaces and yields them with a short pause, so the streaming
# below is genuine streaming, not a print statement pretending.
ANSWER = (
    "Streaming lets you show work in progress instead of a spinner: "
    "tokens as the model writes them, and a note each time a node finishes."
)

DOCUMENTS = ["handbook.md", "support-faq.md", "onboarding.md"]


class RagState(TypedDict, total=False):
    """State for the example.

    `messages` carries the `add_messages` reducer (same as `app/state.py`), so
    nodes return only what they added. `documents` has no reducer, so whatever a
    node returns replaces the old value.
    """

    messages: Annotated[list, add_messages]
    documents: list[str]


def retrieve(state: RagState) -> dict:
    """Pretend to search some files, reporting progress while it works.

    `get_stream_writer()` returns a function that publishes anything you give it
    to `stream_mode="custom"`. This is the escape hatch for progress a node
    knows about but the state does not: LangGraph only emits `updates` once a
    node *returns*, and a slow node has plenty to say before then.
    """
    writer = get_stream_writer()
    found = []
    for number, name in enumerate(DOCUMENTS, start=1):
        time.sleep(0.3)  # stand-in for a network call
        writer({"searched": f"{number}/{len(DOCUMENTS)}", "file": name})
        found.append(name)
    return {"documents": found}


def respond(state: RagState) -> dict:
    """Answer using the retrieved files.

    Note this calls `model.invoke`, not `model.stream`, and tokens still reach
    `stream_mode="messages"`. LangGraph attaches a streaming callback to every
    model call inside a node, and LangChain chat models notice that callback and
    switch to streaming internally — so nodes are written the plain way and the
    caller decides whether to stream. `app/graph.py` never mentions streaming
    for exactly this reason.
    """
    # A fresh iterator per call: GenericFakeChatModel consumes one message per
    # invocation and would raise StopIteration on a second one.
    model = GenericFakeChatModel(messages=iter([AIMessage(ANSWER)]))
    question = state["messages"][-1].text
    sources = ", ".join(state["documents"])
    reply = model.invoke(f"Question: {question}\nSources: {sources}")
    return {"messages": [reply]}


def build_graph():
    builder = StateGraph(RagState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("respond", respond)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "respond")
    builder.add_edge("respond", END)

    return builder.compile()


graph = build_graph()

QUESTION = "why bother streaming?"


def start():
    """The input for one run — a new dict each time, so runs stay independent."""
    return {"messages": [HumanMessage(QUESTION)]}


def demo_updates() -> None:
    """One chunk per node, containing only what that node returned."""
    for chunk in graph.stream(start(), stream_mode="updates"):
        for node, update in chunk.items():
            print(f"  {node:>8} -> {update}")


def demo_values() -> None:
    """One chunk per step: the entire state as it stands after that step."""
    for state in graph.stream(start(), stream_mode="values"):
        documents = state.get("documents", [])
        print(f"  {len(state['messages'])} message(s), {len(documents)} document(s)")


def demo_messages() -> None:
    """Model output as it is produced.

    Each chunk is a `(message_chunk, metadata)` pair. The metadata says which
    node produced it — essential in a real agent, where several different model
    calls (an answer, a summariser, a router) share one stream.
    """
    print("  ", end="", flush=True)
    for message, metadata in graph.stream(start(), stream_mode="messages"):
        if metadata["langgraph_node"] == "respond":
            print(message.text, end="", flush=True)
    print()


def demo_custom() -> None:
    """Only what the nodes wrote themselves via `get_stream_writer()`."""
    for item in graph.stream(start(), stream_mode="custom"):
        print(f"  searching... {item['searched']}  {item['file']}")


def demo_combined() -> None:
    """Several modes at once: every chunk arrives as a `(mode, chunk)` pair."""
    tokens = 0
    for mode, chunk in graph.stream(start(), stream_mode=["custom", "updates", "messages"]):
        if mode == "custom":
            print(f"  [custom]   {chunk['file']}")
        elif mode == "updates":
            # `chunk` is {node_name: what_it_returned}; here we only want the name.
            print(f"  [updates]  {', '.join(chunk)} finished")
        elif mode == "messages":
            message, _ = chunk
            tokens += 1
            if tokens <= 4:
                print(f"  [messages] {message.text!r}")
            elif tokens == 5:
                print("  [messages] ... one chunk per token, all the way to the end")


DEMOS = {
    "updates": demo_updates,
    "values": demo_values,
    "messages": demo_messages,
    "custom": demo_custom,
    "combined": demo_combined,
}


def main() -> None:
    wanted = sys.argv[1:] or list(DEMOS)
    for name in wanted:
        demo = DEMOS.get(name)
        if demo is None:
            print(f"unknown mode {name!r}; try: {', '.join(DEMOS)}")
            continue
        print(f"===== stream_mode={name} =====")
        demo()
        print()


if __name__ == "__main__":
    main()
