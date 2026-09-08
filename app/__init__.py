"""Load .env before any submodule reads os.environ.

`app.graph` resolves OPENAI_MODEL and constructs the client at import time, so
the .env has to be loaded here rather than inside main() to guarantee ordering.
"""

from dotenv import load_dotenv

load_dotenv()
