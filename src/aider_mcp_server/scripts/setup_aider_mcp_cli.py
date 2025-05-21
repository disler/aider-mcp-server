#!/usr/bin/env python3
"""CLI entry point for the aider-mcp setup script."""

import sys
from aider_mcp_server.setup_aider_mcp import main

def cli_main():
    """Entry point for the setup-aider-mcp command."""
    sys.exit(main())

if __name__ == "__main__":
    cli_main()