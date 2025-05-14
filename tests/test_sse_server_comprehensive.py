import asyncio
from unittest.mock import MagicMock, patch

import pytest

from aider_mcp_server.sse_server import serve_sse
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.transport_coordinator import ApplicationCoordinator


@pytest.mark.asyncio
async def test_sse_connection_establishment_and_termination():
    # Mocking setup
    mock_coordinator_instance = MagicMock(spec=ApplicationCoordinator)
    mock_asyncio_event_cls = patch("aider_mcp_server.sse_server.asyncio.Event").start()
    mock_event_instance_returned = mock_asyncio_event_cls.return_value
    mock_event_instance_returned.is_set = MagicMock(return_value=False)
    mock_event_instance_returned.wait = MagicMock(side_effect=wait_side_effect)
    mock_event_instance_returned.set = MagicMock(side_effect=set_side_effect)

    host, port, editor_model, cwd, heartbeat = (
        "127.0.0.1",
        8888,
        "test_model",
        "/test/repo",
        20.0,
    )

    with patch(
        "aider_mcp_server.sse_server.is_git_repository", return_value=(True, None)
    ):
        await serve_sse(host, port, editor_model, cwd, heartbeat_interval=heartbeat)

    # Verify coordinator was used correctly
    mock_coordinator_instance.__aenter__.assert_awaited_once()

    # Verify coordinator cleanup was called
    mock_coordinator_instance.__aexit__.assert_awaited_once()


def wait_side_effect(mock_event_instance_returned):
    while not mock_event_instance_returned.is_set():
        await asyncio.sleep(0.001)


def set_side_effect(mock_event_instance_returned):
    mock_event_instance_returned.is_set.return_value = True
