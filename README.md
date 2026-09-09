# LangGraph Agent

A minimal LangGraph project: a tool-calling agent backed by an OpenAI model. It
can do arithmetic, search the web, query a PostgreSQL database, and answer
questions about your own documents using a vector search.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```

Two keys: `OPENAI_API_KEY` for the chat model, and `HF_TOKEN` for the document
embeddings (free — create one at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), "Read"
access is enough). Web search needs none; it uses DuckDuckGo.

The database and document tools need a running PostgreSQL server with the
[pgvector](https://github.com/pgvector/pgvector) extension available. On macOS:

```bash
brew install postgresql@17 pgvector
brew services start postgresql@17
.venv/bin/python -m app.db      # creates the database, tables and sample rows
```

That last command is optional — the tools run it themselves on first use. The
default connection URL is
`postgresql://postgres:postgres@localhost:5432/chatbot`; set `DATABASE_URL` in
`.env` to point at a different server.

To load the sample documents into the vector database:

```bash
.venv/bin/python -m app.loader             # loads ./documents
.venv/bin/python -m app.loader my/notes    # or any other folder or file
```

Loading sends each chunk to Hugging Face to be embedded, so a big folder takes
a moment.

`.env` is loaded automatically on import (see [app/__init__.py](app/__init__.py)),
so there is no need to `export` anything.

## Run

```bash
.venv/bin/python -m app.main                  # interactive chat
.venv/bin/python -m app.main "what is 17*23"  # one-shot
.venv/bin/python -m app.main --no-stream ...  # wait for the finished answer
```

The answer appears word by word as the model writes it, and each tool call is
announced as it is requested — see [Streaming](#streaming).

## Layout

| File | Purpose |
|---|---|
| [app/state.py](app/state.py) | `AgentState` — the state passed between nodes |
| [app/tools.py](app/tools.py) | Tool definitions — calculator, web search, database, documents |
| [app/db.py](app/db.py) | PostgreSQL connection + the sample schema and rows |
| [app/vectorstore.py](app/vectorstore.py) | pgvector table, embeddings and similarity search |
| [app/loader.py](app/loader.py) | Reads files from disk, chunks them, stores them |
| [documents/](documents/) | Sample files to load — drop your own in here |
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
| `search_documents` | `query`, `max_results` | Finds passages in your loaded documents by meaning |

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
`orders`) in a PostgreSQL database named `chatbot`. Postgres is a
**server**, not a file — it runs as its own process and you reach it over a
connection URL:

```
postgresql://user:password@host:port/database_name
postgresql://postgres:postgres@localhost:5432/chatbot    <- the default here
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

### Documents (vector search)

`database_query` answers questions about rows and numbers. `search_documents`
answers questions about **prose** — whatever you loaded from disk. This is the
"R" in RAG (retrieval-augmented generation): rather than hoping the model
memorised your files, the agent fetches the few paragraphs that are actually
relevant and reads them before answering.

A keyword search only finds the words you typed. A **vector search** finds text
that *means* the same thing, so "how much time off do I get?" matches a
paragraph that only ever says "annual leave". Three steps make that work:

1. An **embedding model** turns a piece of text into a list of 384 numbers.
   Texts about similar things get similar numbers. We use
   `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face, called through
   their hosted Inference API — so nothing heavyweight is installed locally, but
   loading and searching both need a network connection and `HF_TOKEN`.
2. Those vectors are stored in Postgres in a real `vector(384)` column, which
   the **pgvector** extension adds.
3. To answer a question we embed the question the same way and ask Postgres for
   the rows whose vectors are closest to it.

"Closest" is cosine distance, written `<=>` in pgvector — `0` means identical
direction, `1` means unrelated — so similarity is `1 - distance`:

```sql
SELECT source, content, 1 - (embedding <=> %(query)s::vector) AS similarity
FROM documents
ORDER BY embedding <=> %(query)s::vector
LIMIT 4
```

The `ORDER BY` is what uses the HNSW index; without one, Postgres compares the
question against every row in the table.

Vectors from two different embedding models are not comparable, so the model
name and the column width have to agree. Change `EMBEDDING_MODEL` in
[app/vectorstore.py](app/vectorstore.py) and the table is dropped and rebuilt
the next time you run the loader, with a message saying so — set
`EMBEDDING_DIMENSIONS` to match the new model (768 for `all-mpnet-base-v2`, for
example) or Postgres will reject the vectors.

**Loading** ([app/loader.py](app/loader.py)) is read → split → store. Files are
cut into ~1000-character chunks with a 150-character overlap, because the agent
should be handed a couple of relevant paragraphs rather than fifty pages — and
because the embedding of a whole document is an average of everything in it,
which is fuzzy and matches nothing well. The overlap keeps a sentence that falls
on a boundary readable in at least one chunk.

`.txt`, `.md` and `.pdf` are supported. Re-loading a file you already loaded is
safe: a `UNIQUE (source, chunk_index)` constraint plus a delete-then-insert in
one transaction replaces its old chunks instead of storing a second copy.

Search results below a similarity of `0.2` are dropped — pgvector always returns
the *nearest* rows, even when the nearest thing in the database is not close at
all, so without a floor an unrelated question gets confident nonsense back.

```
.venv/bin/python -m app.main "how many days of paid time off do employees get?"

Employees receive 28 days of paid annual leave per year, plus public holidays.
Up to 5 unused days may be carried into the following year, and any days beyond
that are lost on December 31st. This information is from the company handbook.
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

## Streaming

`graph.invoke(...)` returns one dict when everything has finished. On a question
that needs two tool calls that is several seconds of nothing. `graph.stream(...)`
returns a **generator** — a lazy sequence you loop over — that yields while the
graph is still running. `stream_mode` decides what it yields:

| `stream_mode` | Each chunk is | Use it for |
|---|---|---|
| `"updates"` | `{node_name: what_that_node_returned}` | progress, one node at a time |
| `"values"` | the whole state after each step | a running snapshot |
| `"messages"` | `(message_chunk, metadata)` — model output token by token | the typing effect |
| `"custom"` | whatever a node wrote itself | progress from inside a slow node |
| `"debug"` | every internal event | debugging |

Pass a **list** of modes and each chunk arrives as a `(mode, chunk)` pair
instead, so one loop can do several at once. That is what
[app/main.py](app/main.py) does:

```python
for mode, chunk in graph.stream(
    {"messages": [HumanMessage(text)]}, config, stream_mode=["messages", "updates"]
):
    if mode == "messages":
        message, metadata = chunk
        if metadata.get("langgraph_node") == "agent" and message.text:
            print(message.text, end="", flush=True)   # the answer, as it is written
    elif mode == "updates":
        ...                                           # "the model asked for a tool"
```

```
you> who spent the most, and on what?
bot>
  · database_schema()
  · database_query(sql='SELECT c.name, SUM(o.total) ...')
Linus Torvalds spent the most, 878.98 across three orders...
```

Two details worth knowing:

- **`metadata["langgraph_node"]` matters.** Tool results are messages too, and a
  bigger app may run several model calls (an answer, a router, a summariser)
  in one graph. Without that filter they all land in the same stream.
- **Nodes do not mention streaming.** [app/graph.py](app/graph.py) calls
  `model.invoke(...)`, not `model.stream(...)`, and tokens still come through:
  LangGraph attaches a streaming callback to model calls inside a node, and
  LangChain chat models switch to streaming internally when they see it. Nodes
  stay written the plain way and the *caller* decides whether to stream.

A tool-calling turn produces no text of its own (only the call), so the token
stream and the tool announcements never fight over the same line.

### Progress from inside a node — `stream_mode="custom"`

`updates` only fires when a node *returns*, and a node that spends ten seconds
fetching documents has plenty to say before then. `get_stream_writer()` returns
a function that publishes anything you hand it to the `custom` stream:

```python
from langgraph.config import get_stream_writer

def retrieve(state):
    writer = get_stream_writer()
    for number, name in enumerate(files, start=1):
        writer({"searched": f"{number}/{len(files)}", "file": name})
    ...
```

## Examples

Four self-contained graphs that run without an API key — every node is a plain
Python function (the streaming one uses a fake model), so the graph mechanics
are visible on their own.

```bash
.venv/bin/python -m app.examples.simple                  # linear pipeline
.venv/bin/python -m app.examples.branching               # conditional routing
.venv/bin/python -m app.examples.visualize               # draw every graph
.venv/bin/python -m app.examples.streaming               # every stream mode
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

### Streaming — [app/examples/streaming.py](app/examples/streaming.py)

```
START ──> retrieve ──> respond ──> END
```

Every mode from [Streaming](#streaming) above, run one after another on the same
two-node graph so the output can be compared side by side:

```bash
.venv/bin/python -m app.examples.streaming            # all of them
.venv/bin/python -m app.examples.streaming messages   # or one: updates, values,
                                                      # messages, custom, combined
```

`retrieve` reports progress with `get_stream_writer()`; `respond` calls a
`GenericFakeChatModel`, which replays a canned sentence one word at a time — so
the token streaming is real streaming, and the whole example needs no API key.

### Graph visualization — [app/examples/visualize.py](app/examples/visualize.py)

Any compiled graph can draw itself via `graph.get_graph()`:

```bash
.venv/bin/python -m app.examples.visualize                # Mermaid to stdout
.venv/bin/python -m app.examples.visualize --write diagrams/   # + .mmd files
.venv/bin/python -m app.examples.visualize --png diagrams/     # + .png files
```

This renders the examples above *and* the agent graph from
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

TOOLS = [calculator, word_count, web_search, database_schema, database_query,
         search_documents, reverse]
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
