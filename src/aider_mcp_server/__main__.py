"""Main entry point for the Aider MCP Server."""

# Import standard library modules first
import sys

# Import the main function from a separate module
# to avoid the circular import issues that trigger the RuntimeWarning
from aider_mcp_server.templates.initialization.cli import main

# This creates a clean entry point for the package when run with python -m
if __name__ == "__main__":
    sys.exit(main())
