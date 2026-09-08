"""Interactive CLI for the agent.

    python -m app.main                 # chat loop
    python -m app.main "what is 17*23" # single question
"""

import sys
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph


def run_once(graph, config, text: str) -> None:
    result = graph.invoke({"messages": [HumanMessage(text)]}, config)
    print(result["messages"][-1].text)


def main() -> None:
    # InMemorySaver keeps history for the life of the process. Swap in
    # langgraph-checkpoint-sqlite or -postgres to persist across runs.
    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    if len(sys.argv) > 1:
        run_once(graph, config, " ".join(sys.argv[1:]))
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
        print("bot> ", end="", flush=True)
        run_once(graph, config, text)


if __name__ == "__main__":
    main()
