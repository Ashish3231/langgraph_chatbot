"""Standalone LangGraph examples.

These run without an API key: every node is a plain Python function, so the
graph mechanics (state, reducers, edges, routing) are visible on their own.

    python -m app.examples.simple "some text"
    python -m app.examples.branching "3 * (14 + 2)"
    python -m app.examples.visualize
"""
