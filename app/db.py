"""The PostgreSQL database that the `database_*` tools read.

Postgres is a *server*, not a file: it runs as its own process and you reach it
over a network connection, even when that server is on your own machine. That
gives us three things a single-file database cannot:

  * real types — `NUMERIC` stores money exactly, `DATE` stores a real date
  * real permissions — a session can be put into read-only mode by the server
  * many clients at once — several agents can query the same data safely

Connecting is done with a **connection URL**:

    postgresql://user:password@host:port/database_name
    postgresql://postgres:postgres@localhost:5432/chatbot   <- our default

Set `DATABASE_URL` in your .env to point somewhere else.

`ensure_database()` creates the database, the tables and the sample rows if they
are not there yet, so a fresh clone works with no manual SQL. It is safe to call
repeatedly — every step checks first.
"""

import os
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

# The database we read. Override in .env to use a different server.
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/chatbot"
)

# A query written by a language model could accidentally be an expensive one, so
# the server is told to abort anything still running after 10 seconds. Passing
# it as a connection option means it applies to every query on the connection.
STATEMENT_TIMEOUT_MS = 10_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    city        TEXT    NOT NULL,
    signup_date DATE    NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id       INTEGER        PRIMARY KEY,
    name     TEXT           NOT NULL,
    category TEXT           NOT NULL,
    -- NUMERIC is exact. Never use a float for money: 0.1 + 0.2 != 0.3 in
    -- binary floating point, and those tiny errors accumulate over a SUM().
    price    NUMERIC(10, 2) NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    product_id  INTEGER NOT NULL REFERENCES products(id),
    quantity    INTEGER NOT NULL,
    order_date  DATE    NOT NULL
);
"""

# Sample rows. Each tuple lines up with the column order above.
CUSTOMERS = [
    (1, "Ada Lovelace", "London", "2024-01-15"),
    (2, "Grace Hopper", "New York", "2024-02-03"),
    (3, "Alan Turing", "London", "2024-02-20"),
    (4, "Katherine Johnson", "Hampton", "2024-03-11"),
    (5, "Linus Torvalds", "Portland", "2024-05-02"),
]

PRODUCTS = [
    (1, "Mechanical Keyboard", "peripherals", "129.00"),
    (2, "27-inch Monitor", "displays", "349.50"),
    (3, "USB-C Dock", "peripherals", "89.99"),
    (4, "Standing Desk", "furniture", "599.00"),
    (5, "Desk Lamp", "furniture", "45.25"),
]

ORDERS = [
    # (id, customer_id, product_id, quantity, order_date)
    (1, 1, 1, 2, "2024-03-01"),
    (2, 1, 3, 1, "2024-03-01"),
    (3, 2, 2, 1, "2024-03-05"),
    (4, 3, 4, 1, "2024-03-18"),
    (5, 3, 5, 3, "2024-03-18"),
    (6, 4, 1, 1, "2024-04-02"),
    (7, 5, 2, 2, "2024-04-20"),
    (8, 5, 3, 2, "2024-04-20"),
    (9, 2, 5, 1, "2024-05-09"),
    (10, 1, 2, 1, "2024-06-14"),
]


def _database_name() -> str:
    """Pull the database name out of DATABASE_URL.

    In `postgresql://postgres:postgres@localhost:5432/chatbot` the parsed
    `.path` is "/chatbot", so we drop the leading slash.
    """
    return urlsplit(DATABASE_URL).path.lstrip("/")


def _admin_url() -> str:
    """The same server, but pointing at the built-in `postgres` database.

    You cannot create a database while connected to it, so we connect to the
    maintenance database that every Postgres server ships with, and create ours
    from there.
    """
    parts = urlsplit(DATABASE_URL)
    return urlunsplit(parts._replace(path="/postgres"))


def _create_database_if_missing() -> None:
    """CREATE DATABASE, but only when it does not already exist."""
    # autocommit=True is required here: Postgres refuses to run CREATE DATABASE
    # inside a transaction block, and psycopg opens one for you by default.
    with psycopg.connect(_admin_url(), autocommit=True) as connection:
        # pg_database is the server's catalogue of every database on it.
        # The %s placeholder keeps the value separate from the SQL — this is
        # what prevents SQL injection. Never build SQL with f-strings.
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (_database_name(),)
        ).fetchone()
        if not exists:
            # Identifiers (table and database names) cannot use %s placeholders,
            # so psycopg provides sql.Identifier to quote them safely instead.
            connection.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(_database_name()))
            )


def _create_tables_and_seed() -> None:
    """Create the tables if needed and insert the sample rows once."""
    # `with psycopg.connect(...)` commits when the block ends without an error
    # and rolls back if one is raised, then closes the connection.
    with psycopg.connect(DATABASE_URL) as connection:
        # Every CREATE here is "IF NOT EXISTS", so this is safe to re-run.
        connection.execute(SCHEMA)

        # ON CONFLICT DO NOTHING makes the inserts idempotent: rows whose id is
        # already present are skipped instead of raising a duplicate-key error.
        connection.cursor().executemany(
            "INSERT INTO customers VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            CUSTOMERS,
        )
        connection.cursor().executemany(
            "INSERT INTO products VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            PRODUCTS,
        )
        connection.cursor().executemany(
            "INSERT INTO orders VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            ORDERS,
        )


def ensure_database() -> None:
    """Make sure the database, tables and sample rows all exist."""
    _create_database_if_missing()
    _create_tables_and_seed()


# Set to True after the first successful setup, so the "does the database
# exist?" check runs once per process instead of before every single query.
_database_ready = False


def connect_read_only() -> psycopg.Connection:
    """Open a connection the server itself will not let us write through.

    `read_only = True` makes psycopg start every transaction with
    READ ONLY, so an INSERT, UPDATE, DROP or anything else that modifies data is
    rejected by Postgres with a `ReadOnlySqlTransaction` error. That is a real
    guarantee enforced by the server, not a convention — which matters here,
    because the SQL we run was written by a language model.

    For a production system you would go one step further and connect as a role
    that was only ever GRANTed SELECT.
    """
    # `global` lets this function assign to the module-level variable above
    # rather than creating a new local one with the same name.
    global _database_ready
    if not _database_ready:
        ensure_database()
        _database_ready = True

    connection = psycopg.connect(
        DATABASE_URL,
        # row_factory decides what a row looks like in Python. dict_row gives
        # dictionaries keyed by column name, so you can read row["name"]
        # instead of counting positions.
        row_factory=dict_row,
        options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
    )
    connection.read_only = True
    return connection


if __name__ == "__main__":
    # `python -m app.db` sets the database up by hand, which is handy the first
    # time or after you have dropped it.
    ensure_database()
    print(f"Database ready at {DATABASE_URL}")
