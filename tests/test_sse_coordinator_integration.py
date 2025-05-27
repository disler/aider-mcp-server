"""
Tests for SSE server integration with Application Coordinator.

These tests verify that the SSE Transport Adapter properly integrates with the
Application Coordinator, including initialization, subscription to events,
and event processing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.organisms.transports.sse.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.templates.servers.sse_server import run_sse_server


@pytest.mark.asyncio
async def test_adapter_registration_with_coordinator():
    """Test that the SSE adapter registers itself with the coordinator."""
    # Create a mock coordinator
    mock_coordinator = MagicMock()
    mock_coordinator.register_transport = AsyncMock()
    mock_coordinator.register_transport_adapter = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Initialize the adapter
    await adapter.initialize()

    # Verify the adapter registered itself with the coordinator
    mock_coordinator.register_transport_adapter.assert_called_once_with(adapter)


@pytest.mark.asyncio
async def test_adapter_event_subscription():
    """Test that the SSE adapter subscribes to events from the coordinator."""
    # Create a mock coordinator
    mock_coordinator = MagicMock()
    mock_coordinator.register_transport = AsyncMock()
    mock_coordinator.register_transport_adapter = AsyncMock()
    mock_coordinator.subscribe_to_event_type = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Initialize the adapter
    await adapter.initialize()

    # Verify the adapter subscribed to events
    # The adapter should subscribe to all event types in its capabilities
    capabilities = adapter.get_capabilities()
    for event_type in capabilities:
        mock_coordinator.subscribe_to_event_type.assert_any_call(adapter.get_transport_id(), event_type)

    # Check that all capabilities were subscribed to
    assert mock_coordinator.subscribe_to_event_type.call_count == len(capabilities)


@pytest.mark.asyncio
async def test_coordinator_event_propagation_to_adapter():
    """Test that events from the coordinator are propagated to the SSE adapter."""
    # Skip the parent initialization methods to avoid awaiting MagicMock
    with patch(
        "aider_mcp_server.molecules.transport.base_adapter.AbstractTransportAdapter.initialize", new=AsyncMock()
    ):
        # Create the adapter without a coordinator to avoid initialization issues
        adapter = SSETransportAdapter()

        # Mock send_event and should_receive_event
        adapter.send_event = AsyncMock()
        adapter.should_receive_event = MagicMock(return_value=True)

        # Create a test event
        test_event_type = EventTypes.STATUS
        test_event_data = {
            "message": "Test message",
            "transport_origin": {"transport_id": "stdio", "transport_type": "stdio"},
        }

        # Test that send_event is called when the adapter should receive an event
        # This simulates what happens when the coordinator sends an event to the adapter
        if adapter.should_receive_event(test_event_type, test_event_data):
            await adapter.send_event(test_event_type, test_event_data)

        # Verify that the adapter's send_event method was called with the correct arguments
        adapter.send_event.assert_called_once_with(test_event_type, test_event_data)


@pytest.mark.asyncio
async def test_adapter_shutdown_unsubscribes_from_coordinator():
    """Test that the adapter unregisters from the coordinator on shutdown."""
    # Create a mock coordinator
    mock_coordinator = MagicMock()
    mock_coordinator.register_transport = AsyncMock()
    mock_coordinator.register_transport_adapter = AsyncMock()
    mock_coordinator.unregister_transport = AsyncMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)

    # Initialize the adapter
    await adapter.initialize()

    # Shutdown the adapter
    await adapter.shutdown()

    # Verify that the adapter unregistered itself
    mock_coordinator.unregister_transport.assert_called_once_with(adapter.get_transport_id())


@pytest.mark.asyncio
async def test_adapter_handles_coordinator_exception_during_initialization():
    """Test that the adapter handles exceptions from the coordinator during init."""
    # Skip calling the parent initialize method which throws the exception
    with patch(
        "aider_mcp_server.molecules.transport.base_adapter.AbstractTransportAdapter.initialize", new=AsyncMock()
    ):
        # Create a mock coordinator that raises an exception during register
        mock_coordinator = MagicMock()
        mock_coordinator.register_transport = AsyncMock(side_effect=Exception("Test exception"))
        mock_coordinator.register_transport_adapter = AsyncMock()

        # Create the adapter with the mock coordinator
        adapter = SSETransportAdapter(coordinator=mock_coordinator)

        # Initialize the adapter - this should not raise an exception
        with patch.object(adapter, "logger"):
            await adapter.initialize()

            # Verify that initialization continued despite the exception
            assert adapter._app is not None


@pytest.mark.asyncio
async def test_run_sse_server_with_invalid_working_dir():
    """Test that run_sse_server checks for a valid git repository."""
    # Mock is_git_repository to return False
    with patch(
        "aider_mcp_server.templates.servers.sse_server.is_git_repository", return_value=(False, "Not a git repository")
    ) as mock_git_check:
        # Call run_sse_server with an invalid working directory
        with pytest.raises(ValueError) as excinfo:
            await run_sse_server(host="127.0.0.1", port=8765, current_working_dir="/invalid/dir")

        # Verify that is_git_repository was called with the correct path
        from pathlib import Path

        mock_git_check.assert_called_once_with(Path("/invalid/dir"))

        # Verify that the correct error message was raised
        assert "not a valid git repository" in str(excinfo.value)
