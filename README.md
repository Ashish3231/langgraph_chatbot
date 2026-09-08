# LangGraph Agent

A minimal LangGraph project: a tool-calling agent backed by an OpenAI model.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```

The database tools need a running PostgreSQL server. On macOS:

```bash
brew install postgresql@17
brew services start postgresql@17
.venv/bin/python -m app.db      # creates the database, tables and sample rows
```

That last command is optional — the tools run it themselves on first use. The
default connection URL is `postgresql://localhost/langgraph_demo`; set
`DATABASE_URL` in `.env` to point at a different server.

`.env` is loaded automatically on import (see [app/__init__.py](app/__init__.py)),
so there is no need to `export` anything.

## Run

```bash
.venv/bin/python -m app.main                  # interactive chat
.venv/bin/python -m app.main "what is 17*23"  # one-shot
```

## Layout

| File | Purpose |
|---|---|
| [app/state.py](app/state.py) | `AgentState` — the state passed between nodes |
| [app/tools.py](app/tools.py) | Tool definitions — calculator, web search, database |
| [app/db.py](app/db.py) | PostgreSQL connection + the sample schema and rows |
| [app/graph.py](app/graph.py) | Graph wiring and the compiled `graph` object |
| [app/main.py](app/main.py) | CLI entry point |
| [app/examples/](app/examples/) | Standalone graph examples (no API key needed) |
| [langgraph.json](langgraph.json) | Config for `langgraph dev` / LangGraph Studio |

## Tools

Defined in [app/tools.py](app/tools.py). A tool is a normal Python function
wrapped with `@tool`: the type hints become the arguments the model must supply,
and **the docstring is the description the model reads to decide when to call
it** — so the docstring is part of the program, not just a comment.

| Tool | Arguments | What it does |
|---|---|---|
| `calculator` | `expression` | Arithmetic: `+ - * / ** %` and parentheses |
| `word_count` | `text` | Word and character count |
| `web_search` | `query`, `max_results` | DuckDuckGo search — no API key needed |
| `database_schema` | — | Lists the tables, columns and foreign keys |
| `database_query` | `sql` | Runs a read-only `SELECT` against PostgreSQL |

Every tool returns a **string** (the result is fed back to the model as text)
and never raises — each one catches its own errors and returns a readable
message, because an unhandled exception would stop the whole graph.

### Calculator

The obvious implementation, `eval(expression)`, is a security hole: `eval` runs
any Python, so `__import__("os").system(...)` is a valid "expression". Instead
the text is parsed into a syntax tree with `ast.parse` and walked by hand,
evaluating only the node types in the `_OPS` table. Anything else is refused:

```
>>> calculator.invoke({"expression": "3 * (14 + 2) / 8"})
'6.0'
>>> calculator.invoke({"expression": '__import__("os")'})
"Error evaluating '__import__(\"os\")': unsupported expression element: Call(...)"
```

### Web search

A model's knowledge stops at its training cut-off, so anything recent has to
come from outside. This uses DuckDuckGo through the [`ddgs`](https://pypi.org/project/ddgs/)
package — no API key, no account.

Results are returned as a compact numbered list of title / URL / summary rather
than raw JSON: the model reads text, and text costs far fewer tokens. Snippets
are trimmed to 300 characters and `max_results` is clamped to 1–10, so one call
cannot flood the conversation.

`ddgs` is imported in a `try/except ImportError`, so the app still runs without
it; only `web_search` reports the problem, and only when called.

### Database

Two tools instead of one, on purpose:

```
database_schema   ->  "what tables and columns exist?"
database_query    ->  "run this SELECT"
```

A model cannot write correct SQL without knowing the schema, and one combined
tool would force it to guess. Splitting a job into a *discovery* step and an
*action* step is a pattern worth reusing for any tool that touches a system the
model cannot see.

The data lives in [app/db.py](app/db.py): three tables (`customers`, `products`,
`orders`) in a PostgreSQL database named `langgraph_demo`. Postgres is a
**server**, not a file — it runs as its own process and you reach it over a
connection URL:

```
postgresql://user:password@host:port/database_name
postgresql://localhost/langgraph_demo      <- the default here
```

`ensure_database()` creates the database, the tables and the sample rows if they
are missing, and every step checks first, so it is safe to re-run. It is called
automatically the first time a tool opens a connection, or by hand with
`python -m app.db`.

Two Postgres details the code calls out, because they trip people up:

- **`CREATE DATABASE` cannot run inside a transaction.** psycopg opens one for
  you by default, so that one connection is made with `autocommit=True`.
- **Placeholders are `%s`, not `?`** (SQLite's style), and they only work for
  *values*. Table and database names need `psycopg.sql.Identifier` instead.
  Either way the rule is the same: never build SQL with f-strings.

Money is stored as `NUMERIC(10, 2)`, not a float. Binary floating point cannot
represent `0.1` exactly, and those errors accumulate over a `SUM()`.

**Safety has two layers**, which is worth understanding because the SQL is
written by a language model:

1. `database_query` rejects anything not starting with `SELECT`/`WITH`, and
   rejects a `;` inside the query so a second statement cannot be smuggled in.
   This layer exists to give a *clear error message*.
2. The connection sets `read_only = True`, so **Postgres itself** refuses every
   write. This layer is the *actual guarantee*. A data-modifying CTE such as
   `WITH gone AS (DELETE FROM orders RETURNING *) SELECT * FROM gone` starts
   with `WITH` and sails past layer 1 — the server still rejects it:

   ```
   Error running query: cannot execute SELECT in a read-only transaction
   ```

   In production you would go one step further and connect as a role that was
   only ever `GRANT`ed `SELECT`.

A `statement_timeout` of 10 seconds is set on the connection, so an accidentally
expensive query is aborted by the server rather than hanging the agent. Results
are capped at 50 rows and rendered as an aligned text table:

```
.venv/bin/python -m app.main "who spent the most, and on what?"

name              | city     | spent
------------------+----------+-------
Linus Torvalds    | Portland | 878.98
Alan Turing       | London   | 734.75
Ada Lovelace      | London   | 697.49
```

## How the graph works

```
START ──> agent ──(tool calls?)──> tools ──┐
            │                              │
            │<─────────────────────────────┘
            └──(no tool calls)──> END
```

The `agent` node calls the model. `tools_condition` inspects the reply: if it
contains tool calls, control goes to the `tools` node, which executes them and
loops back so the model can read the results; otherwise the graph ends.

State uses the `add_messages` reducer, so each node returns only the messages it
produced and LangGraph appends them to the running conversation.

## Examples

Three self-contained graphs that run without an API key — every node is a plain
Python function, so the graph mechanics are visible on their own.

```bash
.venv/bin/python -m app.examples.simple                  # linear pipeline
.venv/bin/python -m app.examples.branching               # conditional routing
.venv/bin/python -m app.examples.visualize               # draw every graph
```

### Simple state graph — [app/examples/simple.py](app/examples/simple.py)

```
START ──> clean ──> tokenize ──> summarize ──> END
```

Three nodes joined by plain `add_edge` calls. Each node returns a dict of only
the keys it changed and LangGraph merges it into the state; keys without a
reducer are overwritten. `main()` streams with `stream_mode="updates"` so you can
watch the state fill in one node at a time.

### Graph with branching — [app/examples/branching.py](app/examples/branching.py)

```
                     ┌──> compute ──┐
START ──> classify ──┼──> analyze ──┼──> report ──> END
                     └──> reject  ──┘
```

`classify` records a decision in the state; a separate `route` function reads it
and returns the name of the next node, wired up with `add_conditional_edges`.
Only the named branch runs. All three branches then `add_edge` into `report`, so
the paths rejoin with no extra bookkeeping. The `trail` key carries a list
reducer, so each node appends its step instead of overwriting.

The math and text branches call `calculator` and `word_count` from
[app/tools.py](app/tools.py) directly — a `@tool` is still an ordinary callable
via `.invoke({...})`, no model in the loop.

```bash
.venv/bin/python -m app.examples.branching "3 * (14 + 2) / 8"   # -> compute
.venv/bin/python -m app.examples.branching "how many words"     # -> analyze
```

### Graph visualization — [app/examples/visualize.py](app/examples/visualize.py)

Any compiled graph can draw itself via `graph.get_graph()`:

```bash
.venv/bin/python -m app.examples.visualize                # Mermaid to stdout
.venv/bin/python -m app.examples.visualize --write diagrams/   # + .mmd files
.venv/bin/python -m app.examples.visualize --png diagrams/     # + .png files
```

This renders the two examples above *and* the agent graph from
[app/graph.py](app/graph.py) (skipped with a note if the model client cannot be
constructed).

| Renderer | Requires |
|---|---|
| `draw_mermaid()` | nothing — always available |
| `draw_ascii()` | `pip install grandalf` |
| `draw_mermaid_png()` | network access (renders via mermaid.ink) |

The last two are attempted and skipped with a note when unavailable, so the
Mermaid text always comes through. Paste it into any Mermaid viewer, or drop it
into a `mermaid` fenced code block in Markdown.

## Adding a tool

Add a function to [app/tools.py](app/tools.py) and include it in `TOOLS`:

```python
@tool
def reverse(text: str) -> str:
    """Reverse a string."""
    return text[::-1]

TOOLS = [calculator, word_count, web_search, database_schema, database_query, reverse]
```

The docstring is the description the model sees, so keep it accurate. Both the
model binding and the `ToolNode` read from `TOOLS`, so nothing else needs to
change.

## Memory

`build_graph()` takes a checkpointer. The CLI passes `InMemorySaver()`, which
keeps history per `thread_id` for the life of the process. To persist across
runs, install `langgraph-checkpoint-sqlite` and swap in `SqliteSaver`.

## Notes on the model

The default is `gpt-4.1`, overridable via the `OPENAI_MODEL` environment
variable:

```bash
export OPENAI_MODEL=gpt-4o
```

Set `temperature` and other parameters in `build_model()` in
[app/graph.py](app/graph.py).
