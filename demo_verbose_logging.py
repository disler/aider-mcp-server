#!/usr/bin/env python
"""
Demo script to show how verbose logging helps understand internal operations.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aider_mcp_server.atoms.logging import get_logger
from aider_mcp_server.atoms.event_types import EventTypes


def get_verbose_logger(name):
    """Factory function to create verbose loggers for testing."""
    return get_logger(name, level=logging.DEBUG, verbose=True)  # DEBUG with verbose


async def demo_logging_functionality():
    """Demonstrate different logging scenarios with verbose mode."""
    print("=== Demo: Verbose Logging Features ===\n")
    
    # Test 1: Standard logger without verbose
    print("1. Standard logger (no verbose):")
    standard_logger = get_logger("demo.standard", level=logging.DEBUG)
    standard_logger.info("Standard info message")
    standard_logger.debug("Standard debug message")
    
    print("\n2. Verbose logger (with file/line info):")
    verbose_logger = get_verbose_logger("demo.verbose")
    verbose_logger.info("Verbose info message - note the file path and line number")
    verbose_logger.debug("Verbose debug message - detailed formatting")
    verbose_logger.verbose("Verbose-only message - only shows in verbose mode")
    
    print("\n3. Simulating component lifecycle logging:")
    # Simulate what ApplicationCoordinator would log
    app_logger = get_verbose_logger("ApplicationCoordinator")
    app_logger.verbose("ApplicationCoordinator initializing...")
    app_logger.verbose("Initial ApplicationCoordinator object created.")
    app_logger.verbose("Initializing TransportAdapterRegistry...")
    app_logger.verbose("TransportAdapterRegistry initialized and set.")
    app_logger.verbose("Initializing EventCoordinator...")
    app_logger.verbose("EventCoordinator initialized.")
    app_logger.info("ApplicationCoordinator instance created and initialized successfully.")
    
    print("\n4. Simulating request processing:")
    req_logger = get_verbose_logger("RequestProcessor")
    request_id = "req-12345"
    operation = "aider_ai_code"
    req_logger.verbose(f"Processing request {request_id} for operation '{operation}'")
    req_logger.verbose(f"Looking up handler for operation '{operation}'")
    req_logger.verbose(f"Handler found for operation '{operation}'")
    req_logger.verbose(f"Permission check passed for operation '{operation}'")
    req_logger.verbose(f"Starting handler execution for '{operation}'")
    req_logger.verbose(f"Handler completed successfully for '{operation}'")
    
    print("\n5. Simulating event flow:")
    event_logger = get_verbose_logger("EventCoordinator")
    transport_id = "transport-001"
    event_type = EventTypes.STATUS
    event_logger.verbose(f"Subscribing transport {transport_id} to event type {event_type.value}")
    event_logger.verbose(f"Broadcasting event {event_type.value} with data")
    event_logger.verbose(f"Sending event {event_type.value} to transport {transport_id}")
    
    print("\n6. Session management example:")
    session_logger = get_verbose_logger("SessionManager")
    session_logger.verbose(f"Created new session for transport '{transport_id}'")
    session_logger.verbose(f"Updated session for transport '{transport_id}'")
    session_logger.verbose(f"Permission check for transport '{transport_id}': Required 'EXECUTE', Has: True")
    
    print("\n=== Demo Complete ===")
    print("\nVerbose logging benefits:")
    print("- See exact file paths and line numbers")
    print("- Track component initialization order")
    print("- Debug request processing flow")
    print("- Monitor event subscriptions and broadcasts")
    print("- Understand session lifecycle")
    print("- Only shows when --verbose flag is used")


async def main():
    """Run the demo."""
    await demo_logging_functionality()


if __name__ == "__main__":
    print("Running verbose logging demo...")
    print("This demonstrates how verbose mode helps debug internal operations.\n")
    asyncio.run(main())