"""Interactive CLI for the agent.

    python -m app.main                 # chat loop
    python -m app.main "what is 17*23" # single question
    python -m app.main --no-stream ... # wait for the whole answer instead

The answer is printed token by token as the model writes it, and each tool call
is announced as the agent decides to make one. `app/examples/streaming.py`
shows the stream modes on their own, without a model.
"""

import sys
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph


def format_tool_call(call: dict) -> str:
    """Render one requested tool call as a short single line."""
    args = ", ".join(f"{name}={value!r}" for name, value in call["args"].items())
    if len(args) > 70:
        args = args[:69] + "…"
    return f"{call['name']}({args})"


def stream_once(graph, config, text: str, prefix: str = "") -> None:
    """Run one turn, printing output as it is produced.

    `prefix` is the "bot> " label. The function prints it itself, because it
    also has to know whether the cursor is sitting mid-line when the first tool
    call needs announcing.

    Two stream modes at once, so a list is passed and every chunk arrives as a
    `(mode, chunk)` pair:

    - "messages" gives the model's output token by token, together with metadata
      saying which node produced it — we only want the `agent` node's words.
    - "updates" gives whatever a node returned once it finishes, which is how we
      notice that the model asked for a tool before the tool runs.

    A tool-calling turn produces no text (only the call), so the two never fight
    over the same line.
    """
    print(prefix, end="", flush=True)
    at_line_start = not prefix  # so tool notes and tokens do not collide mid-line

    for mode, chunk in graph.stream(
        {"messages": [HumanMessage(text)]}, config, stream_mode=["messages", "updates"]
    ):
        if mode == "messages":
            message, metadata = chunk
            # Tool results are messages too; skip them and print only the answer.
            if metadata.get("langgraph_node") == "agent" and message.text:
                print(message.text, end="", flush=True)
                at_line_start = False
        elif mode == "updates":
            for node, update in chunk.items():
                if node != "agent":
                    continue
                for call in update["messages"][-1].tool_calls:
                    if not at_line_start:
                        print()
                    print(f"  · {format_tool_call(call)}", flush=True)
                    at_line_start = True

    if not at_line_start:
        print()


def run_once(graph, config, text: str, prefix: str = "") -> None:
    """Run one turn the blocking way: nothing is printed until it is finished."""
    result = graph.invoke({"messages": [HumanMessage(text)]}, config)
    print(prefix + result["messages"][-1].text)


def main() -> None:
    args = sys.argv[1:]
    # `--no-stream` is removed from the list so the rest can be the question.
    streaming = "--no-stream" not in args
    args = [arg for arg in args if arg != "--no-stream"]
    turn = stream_once if streaming else run_once

    # InMemorySaver keeps history for the life of the process. Swap in
    # langgraph-checkpoint-sqlite or -postgres to persist across runs.
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    if args:
        turn(graph, config, " ".join(args))
        return

    print("Chat with the agent. Ctrl-C or 'exit' to quit.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        turn(graph, config, text, prefix="bot> ")


if __name__ == "__main__":
    main()
