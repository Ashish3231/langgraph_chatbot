"""The sample database that the `database_*` tools read.

SQLite is part of Python's standard library (`import sqlite3`), so there is no
server to install and no password to configure: the entire database is a single
file on disk, `data/demo.db`.

The file is created and filled with sample rows the first time anything asks for
a connection, so a fresh clone of this project can answer questions like
"which customer spent the most?" straight away. Delete `data/demo.db` and it is
rebuilt on the next run.
"""

import sqlite3
from pathlib import Path

# __file__ is this file's path. .resolve() makes it absolute, .parent walks up a
# directory. app/db.py -> app/ -> the project root -> data/demo.db.
# Building the path this way means the tools find the database no matter which
# directory you run `python -m app.main` from.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "demo.db"

# The tables, written as plain SQL. Keeping the schema in one string (rather than
# scattered across Python code) means you can paste it into any SQLite client.
SCHEMA = """
CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    city        TEXT    NOT NULL,
    signup_date TEXT    NOT NULL   -- ISO dates: SQLite has no date type
);

CREATE TABLE products (
    id       INTEGER PRIMARY KEY,
    name     TEXT    NOT NULL,
    category TEXT    NOT NULL,
    price    REAL    NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    product_id  INTEGER NOT NULL REFERENCES products(id),
    quantity    INTEGER NOT NULL,
    order_date  TEXT    NOT NULL
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
    (1, "Mechanical Keyboard", "peripherals", 129.00),
    (2, "27-inch Monitor", "displays", 349.50),
    (3, "USB-C Dock", "peripherals", 89.99),
    (4, "Standing Desk", "furniture", 599.00),
    (5, "Desk Lamp", "furniture", 45.25),
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


def _create_database() -> None:
    """Create `data/demo.db` and insert the sample rows.

    Only called when the file does not exist yet — see `ensure_database()`.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # `with sqlite3.connect(...)` commits the transaction when the block ends
    # without an error, and rolls it back if an exception is raised. It does not
    # close the connection, so we still call .close() afterwards.
    connection = sqlite3.connect(DB_PATH)
    try:
        with connection:
            # executescript runs several statements separated by semicolons.
            connection.executescript(SCHEMA)
            # executemany runs one statement once per tuple in the list. The "?"
            # placeholders are filled in by SQLite, which is what keeps values
            # and SQL code separate (no string formatting into SQL, ever).
            connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", CUSTOMERS)
            connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", PRODUCTS)
            connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", ORDERS)
    finally:
        connection.close()


def ensure_database() -> Path:
    """Make sure the database file exists, then return its path."""
    if not DB_PATH.exists():
        _create_database()
    return DB_PATH


def connect_read_only() -> sqlite3.Connection:
    """Open the database in a mode where writes are rejected by SQLite itself.

    The `file:...?mode=ro` URI form is a real guarantee, not a convention: an
    INSERT, UPDATE, DROP or anything else that modifies data raises
    `sqlite3.OperationalError` before it can touch the file. That matters here
    because the SQL we run is written by a language model.
    """
    ensure_database()
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    # Rows come back as objects you can index by column name (row["name"])
    # instead of only by position (row[0]) — easier to read when formatting.
    connection.row_factory = sqlite3.Row
    return connection
