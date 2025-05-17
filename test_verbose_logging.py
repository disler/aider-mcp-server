#!/usr/bin/env python
"""
Test script to demonstrate verbose logging functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aider_mcp_server.atoms.logging import get_logger


async def main():
    """Test verbose logging functionality."""
    # Test 1: Normal logging
    print("=== Test 1: Normal logging ===")
    normal_logger = get_logger("test.normal", log_dir=None)
    normal_logger.info("This is a normal info message")
    normal_logger.debug("This is a normal debug message (won't show)")
    normal_logger.error("This is a normal error message")

    print("\n=== Test 2: DEBUG level without verbose ===")
    debug_logger = get_logger("test.debug", log_dir=None, level=10)  # 10 = DEBUG
    debug_logger.info("This is an info message")
    debug_logger.debug("This is a debug message (will show)")
    debug_logger.verbose("This is a verbose message (won't show)")

    print("\n=== Test 3: DEBUG level with verbose ===")
    verbose_logger = get_logger("test.verbose", log_dir=None, level=10, verbose=True)
    verbose_logger.info("This is an info message")
    verbose_logger.debug("This is a debug message with verbose formatting")
    verbose_logger.verbose("This is a verbose-only message (will show)")

    print("\n=== Test 4: Environment variable test ===")
    import os

    os.environ["MCP_LOG_LEVEL"] = "VERBOSE"
    env_logger = get_logger("test.env", log_dir=None)
    env_logger.info("This is an info message")
    env_logger.debug("This is a debug message (verbose from env)")
    env_logger.verbose("This is a verbose message (will show)")


if __name__ == "__main__":
    asyncio.run(main())
