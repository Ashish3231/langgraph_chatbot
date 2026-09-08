"""Render the graphs in this project as diagrams.

    python -m app.examples.visualize                 # Mermaid to stdout
    python -m app.examples.visualize --write out/    # also write .mmd files
    python -m app.examples.visualize --png out/      # .mmd + .png (needs network)

Every compiled graph exposes `get_graph()`, and the returned object can draw
itself. `draw_mermaid()` is pure Python and always available; `draw_ascii()`
needs `grandalf`, and `draw_mermaid_png()` posts to mermaid.ink, so both are
attempted and skipped with a note if unavailable.
"""

import sys
from pathlib import Path

from app.examples import branching, simple


def graphs() -> dict:
    """The graphs to render, by output name.

    `app.graph` is imported lazily: it constructs an OpenAI client at import
    time, so it is only pulled in when a key is actually configured.
    """
    rendered = {"simple": simple.graph, "branching": branching.graph}
    try:
        from app.graph import graph as agent_graph
    except Exception as exc:  # missing key, missing package — not fatal here
        print(f"note: skipping the agent graph ({type(exc).__name__}: {exc})\n", file=sys.stderr)
    else:
        rendered["agent"] = agent_graph
    return rendered


def render(name: str, graph, out_dir: Path | None, png: bool) -> None:
    drawable = graph.get_graph()
    mermaid = drawable.draw_mermaid()

    print(f"===== {name} =====")
    print(mermaid)

    try:
        print(drawable.draw_ascii())
    except (ImportError, ModuleNotFoundError):
        print("(install `grandalf` for an ASCII rendering)\n")

    if out_dir is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    mmd_path = out_dir / f"{name}.mmd"
    mmd_path.write_text(mermaid)
    print(f"wrote {mmd_path}")

    if png:
        try:
            png_path = out_dir / f"{name}.png"
            png_path.write_bytes(drawable.draw_mermaid_png())
            print(f"wrote {png_path}")
        except Exception as exc:
            print(f"png skipped for {name} ({type(exc).__name__}: {exc})", file=sys.stderr)


def main() -> None:
    args = sys.argv[1:]
    png = "--png" in args
    out_dir = None
    for flag in ("--write", "--png"):
        if flag in args:
            index = args.index(flag) + 1
            out_dir = Path(args[index]) if index < len(args) else Path("diagrams")

    for name, graph in graphs().items():
        render(name, graph, out_dir, png)


if __name__ == "__main__":
    main()
