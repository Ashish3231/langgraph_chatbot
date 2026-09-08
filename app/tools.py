"""Tools the agent can call.

A "tool" here is an ordinary Python function wrapped with LangChain's `@tool`
decorator. The decorator reads three things and hands them to the model:

  * the function name        -> the tool's name
  * the type hints           -> the arguments the model must supply
  * the docstring            -> the description the model uses to decide *when*
                                to call it

So the docstring is not a comment for humans only; it is part of the program.
Keep it accurate and say when the tool should be used.

Two rules that apply to every tool below:

  1. Return a string. The result is fed back to the model as text.
  2. Never raise. An unhandled exception stops the whole graph, so each tool
     catches its own failures and returns a readable error message instead —
     the model can then apologise, retry, or try something else.
"""

import ast
import operator

import psycopg
from langchain_core.tools import tool

from app.db import connect_read_only

# ddgs (the DuckDuckGo search client) is an optional dependency. Importing it
# inside a try/except means the rest of the app still works if it is missing;
# only `web_search` reports the problem, and only when it is actually called.
try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - depends on what is installed
    DDGS = None


# ---------------------------------------------------------------------------
# 1. Calculator
# ---------------------------------------------------------------------------
# Language models are unreliable at arithmetic, so we hand the work to Python.
#
# The obvious implementation is `eval(expression)` — and it is a serious
# security hole, because `eval` runs *any* Python, not just maths. A string like
# `__import__("os").system("rm -rf ~")` is a valid expression.
#
# Instead we parse the text into a syntax tree with `ast.parse` and then walk
# that tree ourselves, evaluating only the node types we explicitly allow.
# Anything else — a function call, a name, an attribute — falls through to a
# ValueError. This is the standard safe-calculator pattern in Python.

_OPS = {
    ast.Add: operator.add,  # a + b
    ast.Sub: operator.sub,  # a - b
    ast.Mult: operator.mul,  # a * b
    ast.Div: operator.truediv,  # a / b
    ast.Pow: operator.pow,  # a ** b
    ast.Mod: operator.mod,  # a % b
    ast.USub: operator.neg,  # -a
    ast.UAdd: operator.pos,  # +a
}


def _eval(node: ast.AST) -> float:
    """Evaluate one node of a parsed arithmetic expression.

    The function calls itself for the sub-parts of an expression (this is
    recursion): to work out `2 * (3 + 4)` it evaluates the left side, evaluates
    the right side, then applies the operator between them.
    """
    # A plain number, e.g. the `3` in `3 + 4`.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    # A binary operation: something on the left, something on the right.
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    # A unary operation, e.g. the minus in `-5`.
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    # Anything we did not explicitly allow above is refused.
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. "3 * (14 + 2) / 8".

    Supports + - * / ** % and parentheses. Use this instead of doing arithmetic
    in your head.
    """
    try:
        # mode="eval" tells the parser to expect a single expression rather than
        # a whole program; `.body` is the expression node at the top of the tree.
        return str(_eval(ast.parse(expression, mode="eval").body))
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return f"Error evaluating {expression!r}: {exc}"


@tool
def word_count(text: str) -> str:
    """Count the words and characters in a piece of text."""
    return f"{len(text.split())} words, {len(text)} characters"


# ---------------------------------------------------------------------------
# 2. Web search
# ---------------------------------------------------------------------------
# The model's knowledge stops at its training cut-off, so anything recent has to
# come from outside. This tool queries DuckDuckGo through the `ddgs` package,
# which needs no API key.
#
# We return the results as readable text rather than raw JSON: the model reads
# text, and a compact numbered list costs far fewer tokens than a JSON dump.

SNIPPET_LIMIT = 300  # characters of each result's summary to keep


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return the top results (title, URL and a summary).

    Use this for current events, recent releases, prices, documentation, or any
    fact you are not certain about. `max_results` defaults to 5 and is capped
    at 10.
    """
    if DDGS is None:
        return "Error: web search is unavailable — install it with `pip install ddgs`."

    # Clamp the model's request into a sane range: min() caps the top,
    # max() stops a zero or negative value from asking for nothing.
    max_results = max(1, min(max_results, 10))

    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as exc:
        # Network calls fail in many different ways (no connection, timeout,
        # rate limit, upstream change), so here a broad `except` is the honest
        # choice — the tool must return text no matter what went wrong.
        return f"Error searching for {query!r}: {exc}"

    if not results:
        return f"No results found for {query!r}."

    # Build the reply one line-group at a time, then join. Repeatedly adding to
    # a string with += creates a new string each time; a list plus "\n".join is
    # the idiomatic Python way to assemble multi-line text.
    lines = [f"Top {len(results)} results for {query!r}:"]
    for position, result in enumerate(results, start=1):
        # .get() returns "" instead of raising if a key is missing, which
        # protects us from changes in what the search API sends back.
        title = result.get("title", "(no title)")
        url = result.get("href", "")
        snippet = result.get("body", "").strip()
        if len(snippet) > SNIPPET_LIMIT:
            snippet = snippet[:SNIPPET_LIMIT].rstrip() + "..."
        lines.append(f"\n{position}. {title}\n   {url}\n   {snippet}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Database (PostgreSQL)
# ---------------------------------------------------------------------------
# Two tools, deliberately kept separate:
#
#   database_schema  -- "what tables and columns exist?"
#   database_query   -- "run this SELECT"
#
# The model cannot write correct SQL without knowing the schema, and a single
# combined tool would force it to guess. With two, the natural sequence is:
# look at the schema, then write a query against it. That splitting of a job
# into a discoverable step and an acting step is a pattern worth reusing.
#
# The connection setup lives in app/db.py; here we only turn results into text.

MAX_ROWS = 50  # never hand the model an unbounded result set


def _format_rows(rows: list[dict]) -> str:
    """Render query results as an aligned plain-text table."""
    if not rows:
        return "(query returned no rows)"

    # Each row is a dict keyed by column name (see `dict_row` in app/db.py).
    # Those names may be computed ones such as "sum" that exist in no table.
    headers = list(rows[0].keys())

    # Convert every value to a string once, so we can measure and print it.
    # None means SQL NULL; showing it as "NULL" is clearer than an empty gap.
    table = [
        [("NULL" if value is None else str(value)) for value in row.values()]
        for row in rows
    ]

    # Column width = the longest cell in that column, header included. zip(*table)
    # transposes the table: it turns a list of rows into a list of columns.
    widths = [
        max(len(header), *(len(cell) for cell in column))
        for header, column in zip(headers, zip(*table))
    ]

    def render(cells) -> str:
        # str.ljust pads a value with spaces so the columns line up.
        return " | ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in table)])


# Postgres describes itself through `information_schema`, a set of standard
# views every SQL database is supposed to provide. Unlike SQLite there is no
# stored "CREATE TABLE" text to print, so we rebuild a readable description
# from the catalogue instead.
COLUMNS_SQL = """
SELECT table_name, column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position
"""

# Foreign keys are what tell the model which columns can be JOINed, so they are
# worth including. This walks the constraint catalogue: each FOREIGN KEY
# constraint is joined to the columns on both sides of the relationship.
FOREIGN_KEYS_SQL = """
SELECT tc.table_name,
       kcu.column_name,
       ccu.table_name  AS foreign_table,
       ccu.column_name AS foreign_column
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
     ON kcu.constraint_name = tc.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
     ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
ORDER BY tc.table_name, kcu.column_name
"""


@tool
def database_schema() -> str:
    """List the tables, columns and foreign keys in the PostgreSQL database.

    Call this before `database_query` so you know what you can select from and
    which columns can be joined.
    """
    try:
        connection = connect_read_only()
    except psycopg.Error as exc:
        return f"Error connecting to the database: {exc}"

    try:
        columns = connection.execute(COLUMNS_SQL).fetchall()
        foreign_keys = connection.execute(FOREIGN_KEYS_SQL).fetchall()
    except psycopg.Error as exc:
        return f"Error reading the schema: {exc}"
    finally:
        # `finally` runs whether or not the code above raised, so the connection
        # is always closed.
        connection.close()

    if not columns:
        return "The database contains no tables."

    # Group the flat list of columns by table name. `setdefault` returns the
    # list already stored under that key, creating an empty one the first time.
    tables: dict[str, list[str]] = {}
    for column in columns:
        nullable = "" if column["is_nullable"] == "YES" else " NOT NULL"
        tables.setdefault(column["table_name"], []).append(
            f"  {column['column_name']} {column['data_type']}{nullable}"
        )

    blocks = [f"{name}\n" + "\n".join(lines) for name, lines in tables.items()]

    if foreign_keys:
        links = [
            f"  {fk['table_name']}.{fk['column_name']}"
            f" -> {fk['foreign_table']}.{fk['foreign_column']}"
            for fk in foreign_keys
        ]
        blocks.append("foreign keys\n" + "\n".join(links))

    return "\n\n".join(blocks)


@tool
def database_query(sql: str) -> str:
    """Run a read-only SQL SELECT against the PostgreSQL database.

    PostgreSQL syntax. Only SELECT (or WITH ... SELECT) statements are allowed,
    one at a time. Call `database_schema` first if you are unsure of the tables.
    Results are capped at 50 rows, so add your own LIMIT or an aggregate such as
    COUNT(*) when a query could match many rows.
    """
    statement = sql.strip().rstrip(";").strip()

    # First guard: a friendly, specific message for the model. `startswith`
    # accepts a tuple, so this reads as "starts with either of these".
    if not statement.lower().startswith(("select", "with")):
        return "Error: only SELECT queries are allowed."
    # Second guard: one statement per call, so nothing can be smuggled in after
    # a semicolon.
    if ";" in statement:
        return "Error: run one statement at a time (no ';' inside the query)."

    try:
        connection = connect_read_only()
    except psycopg.Error as exc:
        return f"Error connecting to the database: {exc}"

    try:
        # Read one row past the cap so we can tell "exactly 50" from "more
        # than 50" and say so.
        rows = connection.execute(statement).fetchmany(MAX_ROWS + 1)
    except psycopg.Error as exc:
        # The real protection is the read-only connection in app/db.py: even a
        # DELETE that slipped past the checks above is refused by Postgres
        # itself rather than changing data. The checks exist for clearer errors.
        return f"Error running query: {exc}"
    finally:
        connection.close()

    truncated = len(rows) > MAX_ROWS
    output = _format_rows(rows[:MAX_ROWS])
    if truncated:
        output += f"\n\n(showing the first {MAX_ROWS} rows; add a LIMIT to narrow this down)"
    return output


# Both the model binding and the ToolNode in app/graph.py read this list, so a
# tool added here becomes available to the agent with no other changes.
TOOLS = [calculator, word_count, web_search, database_schema, database_query]
