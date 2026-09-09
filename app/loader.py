"""Document loader: read files from disk and put them in the vector database.

    python -m app.loader                # load everything in ./documents
    python -m app.loader path/to/folder # load a different folder
    python -m app.loader notes.md       # load a single file

The job has three steps, and each one is a small function below:

  1. **Read** the file into plain text (.txt, .md and .pdf are supported).
  2. **Split** that text into chunks of roughly a thousand characters. We do not
     store whole documents because the agent should be handed a couple of
     relevant paragraphs, not fifty pages — and because an embedding of a whole
     document is an average of everything in it, which is fuzzy and matches
     nothing well.
  3. **Store** the chunks, which embeds them and writes them to Postgres
     (see app/vectorstore.py).

Re-running the loader on a file you have already loaded is safe: its old chunks
are replaced.
"""

import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vectorstore import add_chunks, ensure_documents_table, stats

# Roughly how big each chunk should be, in characters. Overlap repeats the tail
# of one chunk at the start of the next so a sentence split across the boundary
# is still readable in at least one of them.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Which file types we know how to read. `.suffix` gives ".md" for "notes.md".
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}

# pypdf is only needed for PDFs, so a missing install should not stop .md files
# from loading. Same optional-import pattern as `ddgs` in app/tools.py.
try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - depends on what is installed
    PdfReader = None


def read_file(path: Path) -> str:
    """Return the text content of one file, or "" if we cannot read it."""
    if path.suffix.lower() in TEXT_SUFFIXES:
        # errors="ignore" skips the odd byte that is not valid UTF-8 instead of
        # crashing on a file that was saved with a different encoding.
        return path.read_text(encoding="utf-8", errors="ignore")

    if path.suffix.lower() in PDF_SUFFIXES:
        if PdfReader is None:
            print(f"  skipped {path} - install pypdf to read PDFs")
            return ""
        # A PDF is a list of pages; join their text with blank lines between.
        pages = [page.extract_text() or "" for page in PdfReader(path).pages]
        return "\n\n".join(pages)

    return ""


def split_text(text: str) -> list[str]:
    """Cut a long document into overlapping chunks.

    RecursiveCharacterTextSplitter tries to break on paragraph breaks first,
    then single newlines, then sentences, then spaces — so chunks end at natural
    boundaries wherever possible instead of mid-word.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


def find_files(target: Path) -> list[Path]:
    """List the files to load: the file itself, or everything under a folder."""
    if target.is_file():
        return [target]
    # rglob("*") walks the folder and all its sub-folders. `sorted` just makes
    # the output order predictable.
    return sorted(path for path in target.rglob("*") if path.is_file())


def load(target: Path) -> None:
    """Load one file or a whole folder into the vector database."""
    if not target.exists():
        print(f"Nothing to load: {target} does not exist.")
        return

    ensure_documents_table()

    total_chunks = 0
    for path in find_files(target):
        text = read_file(path).strip()
        if not text:
            # Either an unsupported file type or an empty/unreadable one.
            continue

        chunks = split_text(text)
        # `str(path)` is what we store as the source, so answers can cite it.
        stored = add_chunks(str(path), chunks)
        total_chunks += stored
        print(f"  {path} -> {stored} chunks")

    if total_chunks == 0:
        print(f"No readable documents found in {target}.")
        return

    print(f"\nStored {total_chunks} chunks. The database now holds:")
    for row in stats():
        print(f"  {row['source']}: {row['chunks']} chunks")


def main() -> None:
    # sys.argv is the list of words typed on the command line; argv[0] is the
    # program itself, so a given path would be argv[1].
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("documents")
    print(f"Loading {target} ...")
    try:
        load(target)
    except RuntimeError as exc:
        # Raised by app/vectorstore.py when GOOGLE_API_KEY is missing. A one-line
        # message is far more useful here than a stack trace.
        print(f"\n{exc}")
        # A non-zero exit code is how a command-line program says "this failed",
        # which matters if you ever run the loader from a script.
        sys.exit(1)


if __name__ == "__main__":
    main()
