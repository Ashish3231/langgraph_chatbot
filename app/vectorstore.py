"""The vector database: storing documents so the agent can search them by meaning.

A normal `WHERE content LIKE '%holiday%'` search only finds the exact word you
typed. A **vector search** finds text that *means* the same thing, so a question
about "time off" can match a paragraph that only ever says "annual leave".

How that works, in three steps:

  1. An **embedding model** turns a piece of text into a list of numbers (a
     "vector") — 1536 of them for the model we use, which is Google's Gemini
     embedding model. Texts about similar things end up with similar numbers.
  2. We store those vectors in Postgres using the **pgvector** extension, which
     adds a real `vector` column type and distance operators.
  3. To answer a question we embed the question the same way and ask Postgres
     for the rows whose vectors are closest to it.

"Closest" is measured with cosine distance, written `<=>` in pgvector: 0 means
identical direction, 1 means unrelated. Similarity is just `1 - distance`, which
is friendlier to read.

Documents are loaded into this table by `app/loader.py`, and read back out by
the `search_documents` tool in `app/tools.py`.
"""

import os

import psycopg
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.db import DATABASE_URL, ensure_database

# Google's Gemini embedding model, called through the Gemini Developer API.
# The maths happens on Google's servers, so nothing large is installed here.
# That needs a free API key in `GOOGLE_API_KEY` (create one at
# https://aistudio.google.com/apikey).
#
# Every text stored here must be embedded by the *same* model — vectors from two
# different models are not comparable, so switching models means reloading your
# documents. `ensure_documents_table` below notices and handles that for you.
EMBEDDING_MODEL = "gemini-embedding-001"

# How many numbers we ask this model to produce per text.
#
# This one is unusual: it is a **Matryoshka** model, meaning it was trained so
# that you can cut the vector short and it still works — the most important
# information is packed into the earliest numbers. So we get to *choose* the
# size, via the `output_dimensionality` argument passed below.
#
# The model's own default is 3072, but we ask for 1536 on purpose, because
# **pgvector's HNSW index refuses any column wider than 2000 dimensions**:
#
#     column cannot have more than 2000 dimensions for hnsw index
#
# At 3072 the table would still be created and searches would still return the
# right answers — but the CREATE INDEX in SCHEMA below would fail, so every
# search would fall back to comparing the question against every single row.
# 1536 keeps the index (and so the speed) while losing very little accuracy.
EMBEDDING_DIMENSIONS = 1536

# Gemini embeds text differently depending on what the text is *for*: a stored
# passage and the question that should find it get embedded with different
# settings, which measurably improves retrieval. These are the two names the API
# expects; they are passed on each call in `add_chunks` and `search` below.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"  # text we are storing
TASK_QUERY = "RETRIEVAL_QUERY"        # a question we are searching with

# The table itself. `CREATE EXTENSION IF NOT EXISTS vector` runs separately in
# `ensure_documents_table` below, because pgvector has to be loaded into the
# database before a `vector` column can be created.
SCHEMA = f"""
CREATE TABLE IF NOT EXISTS documents (
    -- BIGSERIAL is an auto-incrementing id: Postgres fills it in for us.
    id          BIGSERIAL PRIMARY KEY,
    -- Which file this text came from, and its position within that file.
    source      TEXT      NOT NULL,
    chunk_index INTEGER   NOT NULL,
    content     TEXT      NOT NULL,
    embedding   vector({EMBEDDING_DIMENSIONS}) NOT NULL,
    -- One row per (file, position), so re-loading a file replaces its rows
    -- instead of quietly storing a second copy of everything.
    UNIQUE (source, chunk_index)
);

-- Without an index Postgres compares the question against every single row.
-- That is fine for a few thousand rows and slow for a million. HNSW is
-- pgvector's index for "find me the nearest vectors quickly"; `vector_cosine_ops`
-- tells it we will be measuring distance with the cosine operator `<=>`.
CREATE INDEX IF NOT EXISTS documents_embedding_index
    ON documents USING hnsw (embedding vector_cosine_ops);
"""


# Asks Postgres how wide the existing embedding column is. `to_regclass` returns
# the table's internal id, or NULL when the table does not exist yet — which is
# why this is safe to run on a fresh database. For a pgvector column `atttypmod`
# is simply the number of dimensions.
DIMENSIONS_SQL = """
SELECT atttypmod
FROM pg_attribute
WHERE attrelid = to_regclass('public.documents') AND attname = 'embedding'
"""


def ensure_documents_table() -> None:
    """Create the extension, the table and the index if they are missing.

    If the table was built for a *different* embedding model, it is dropped and
    rebuilt: old vectors have the wrong length, and even if they fitted they
    would be meaningless next to vectors from the new model.
    """
    ensure_database()  # creates the `chatbot` database itself, if needed
    with psycopg.connect(DATABASE_URL) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")

        row = connection.execute(DIMENSIONS_SQL).fetchone()
        if row and row[0] != EMBEDDING_DIMENSIONS:
            print(
                f"The documents table holds {row[0]}-dimension vectors but "
                f"{EMBEDDING_MODEL} produces {EMBEDDING_DIMENSIONS}. "
                "Dropping it - re-run the loader to rebuild it."
            )
            connection.execute("DROP TABLE documents")

        connection.execute(SCHEMA)


# Building the client contacts nothing, but it *does* need GOOGLE_API_KEY to be
# set, so we create it on first use rather than at import time. That way
# importing this module never fails just because a key is missing, and the one
# client is reused for the whole process.
_client = None


def _embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the one shared embedding client, creating it the first time."""
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            # Raising a clear error here beats a confusing 400 from the API.
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Create a free key at "
                "https://aistudio.google.com/apikey and add it to .env"
            )
        _client = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )
    return _client


def _to_vector_literal(vector: list[float]) -> str:
    """Format a Python list the way pgvector wants to read it: "[0.1,0.2,...]".

    psycopg does not know about the `vector` type, so we send the value as text
    and let Postgres convert it with a `::vector` cast in the SQL below.
    """
    return "[" + ",".join(str(number) for number in vector) + "]"


def add_chunks(source: str, chunks: list[str]) -> int:
    """Embed the chunks of one document and store them, replacing any old copy.

    Returns how many chunks were stored.
    """
    if not chunks:
        return 0

    # One API call for the whole list is much faster (and cheaper) than one call
    # per chunk. The results come back in the same order we sent them.
    #
    # task_type tells Gemini these are passages to be *found later*, and
    # output_dimensionality asks for the 1536-wide vectors our column expects.
    vectors = _embeddings().embed_documents(
        chunks,
        task_type=TASK_DOCUMENT,
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )

    # zip() walks two lists side by side; enumerate() adds the position number.
    rows = [
        (source, index, chunk, _to_vector_literal(vector))
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    with psycopg.connect(DATABASE_URL) as connection:
        # Delete first so a shortened document does not leave stale chunks
        # behind. Both statements share one transaction, so either the whole
        # replacement happens or none of it does.
        connection.execute("DELETE FROM documents WHERE source = %s", (source,))
        connection.cursor().executemany(
            """
            INSERT INTO documents (source, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s::vector)
            """,
            rows,
        )
    return len(rows)


# A note if you read Google's docs: they say vectors shorter than the full
# 3072 should be re-normalised before use. That matters for *dot product*
# comparisons, but not here — cosine distance divides by each vector's
# length anyway, so scaling a vector cannot change the ranking.
#
# The SQL that does the actual searching.
#   embedding <=> %s::vector   -> cosine distance between the row and the question
#   ORDER BY that distance     -> closest rows first (this is what uses the index)
SEARCH_SQL = """
SELECT source,
       chunk_index,
       content,
       1 - (embedding <=> %(query)s::vector) AS similarity
FROM documents
ORDER BY embedding <=> %(query)s::vector
LIMIT %(limit)s
"""


def search(question: str, limit: int = 4) -> list[dict]:
    """Return the stored chunks whose meaning is closest to `question`."""
    from app.db import connect_read_only  # imported here to keep the top tidy

    # The mirror image of add_chunks: same model and width, but task_type says
    # this text is a question doing the searching rather than a passage being
    # searched.
    vector = _to_vector_literal(
        _embeddings().embed_query(
            question,
            task_type=TASK_QUERY,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        )
    )

    connection = connect_read_only()
    try:
        # Named placeholders (%(query)s) let us reuse one value twice without
        # passing it twice — the query vector appears in SELECT and ORDER BY.
        return connection.execute(
            SEARCH_SQL, {"query": vector, "limit": limit}
        ).fetchall()
    finally:
        connection.close()


def stats() -> list[dict]:
    """One row per loaded file: its name and how many chunks it has."""
    with psycopg.connect(DATABASE_URL) as connection:
        cursor = connection.execute(
            """
            SELECT source, COUNT(*) AS chunks
            FROM documents
            GROUP BY source
            ORDER BY source
            """
        )
        # Rows here are plain tuples, so turn them into dicts for readability.
        return [{"source": source, "chunks": chunks} for source, chunks in cursor]
