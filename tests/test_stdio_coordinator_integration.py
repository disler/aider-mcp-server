"""Test STDIO transport coordinator integration for Phase 1.2."""

import asyncio
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from aider_mcp_server.atoms.types.event_types import EventTypes
from ...pages.application.coordinator import ApplicationCoordinator
from aider_mcp_server.organisms.transports.stdio.stdio_transport_adapter import StdioTransportAdapter


@pytest_asyncio.fixture
async def clean_coordinator():
    """Provide a clean coordinator instance for each test."""
    # Reset singleton state for testing
    ApplicationCoordinator._instance = None
    ApplicationCoordinator._initialized = False

    # Get fresh instance
    coordinator = await ApplicationCoordinator.getInstance()
    await coordinator.__aenter__()

    yield coordinator

    # Clean up after test
    await coordinator.__aexit__(None, None, None)

    # Reset singleton state for next test
    ApplicationCoordinator._instance = None
    ApplicationCoordinator._initialized = False


@pytest.mark.asyncio
async def test_stdio_auto_discovery_and_registration(clean_coordinator):
    """Test that STDIO transport automatically discovers and registers with coordinators."""
    coordinator = clean_coordinator
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create STDIO transport without providing coordinator - should auto-discover
        stdio_adapter = StdioTransportAdapter(
            coordinator=None,  # This should trigger auto-discovery
            discovery_file=temp_path / "coordinator_discovery.json",
        )

        try:
            # Initialize the adapter - this should discover and connect to coordinator
            await stdio_adapter.initialize()

            # Verify that the adapter has a coordinator
            assert stdio_adapter._coordinator is not None, "STDIO adapter should have discovered coordinator"

            # Verify that the adapter is registered with the coordinator
            transport_id = stdio_adapter.transport_id
            assert transport_id in coordinator._transports, f"Transport {transport_id} should be registered"

            # Verify event subscription
            # The adapter should have subscribed to AIDER events
            # Note: This depends on the coordinator's internal event system implementation

        finally:
            # Clean up
            await stdio_adapter.shutdown()


@pytest.mark.asyncio
async def test_stdio_coordinator_event_broadcasting(clean_coordinator):
    """Test that STDIO transport can receive events from coordinator."""
    coordinator = clean_coordinator
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create STDIO transport with explicit coordinator
        stdio_adapter = StdioTransportAdapter(
            coordinator=coordinator, discovery_file=temp_path / "coordinator_discovery.json"
        )

        try:
            # Initialize the adapter
            await stdio_adapter.initialize()

            # Verify coordinator connection
            assert stdio_adapter._coordinator is coordinator

            # Test event broadcasting
            test_event_data = {"test": True, "message": "Phase 1.2 integration test", "timestamp": 1234567890}

            # Broadcast an AIDER event through the coordinator
            await coordinator.broadcast_event(event_type=EventTypes.AIDER_SESSION_STARTED, data=test_event_data)

            # Give some time for event processing
            await asyncio.sleep(0.1)

            # The test passes if no exceptions are raised during event broadcasting

        finally:
            # Clean up
            await stdio_adapter.shutdown()


@pytest.mark.asyncio
async def test_stdio_cross_transport_communication(clean_coordinator):
    """Test cross-transport communication setup between STDIO and potential SSE transports."""
    coordinator = clean_coordinator
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create two STDIO transports to simulate cross-transport communication
        stdio_adapter1 = StdioTransportAdapter(
            coordinator=coordinator, discovery_file=temp_path / "coordinator_discovery.json"
        )

        stdio_adapter2 = StdioTransportAdapter(
            coordinator=coordinator, discovery_file=temp_path / "coordinator_discovery.json"
        )

        try:
            # Initialize both adapters
            await stdio_adapter1.initialize()
            await stdio_adapter2.initialize()

            # Verify both are registered with the same coordinator
            assert stdio_adapter1._coordinator is coordinator
            assert stdio_adapter2._coordinator is coordinator

            transport_id1 = stdio_adapter1.transport_id
            transport_id2 = stdio_adapter2.transport_id

            assert transport_id1 in coordinator._transports
            assert transport_id2 in coordinator._transports
            assert transport_id1 != transport_id2  # Should have unique IDs

            # Test cross-transport event flow
            test_event_data = {
                "source": transport_id1,
                "destination": transport_id2,
                "phase": "1.2",
                "cross_transport_test": True,
            }

            # Broadcast event from coordinator (simulating AIDER tool event)
            await coordinator.broadcast_event(event_type=EventTypes.AIDER_RATE_LIMIT_DETECTED, data=test_event_data)

            # Give time for event processing
            await asyncio.sleep(0.1)

        finally:
            # Clean up
            await stdio_adapter1.shutdown()
            await stdio_adapter2.shutdown()
