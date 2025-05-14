import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch, ANY  # Added ANY import

import pytest

from aider_mcp_server.security import Permissions
from aider_mcp_server.sse_server import serve_sse
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.transport_coordinator import ApplicationCoordinator


# Helper functions
def setup_test_loggers():
    mock_server_logger = MagicMock()
    mock_adapter_logger = MagicMock()
    return mock_server_logger, mock_adapter_logger


def setup_mock_coordinator():
    coordinator_instance = MagicMock(spec=ApplicationCoordinator)
    coordinator_instance.register_transport = MagicMock()
    coordinator_instance.register_handler = AsyncMock()
    coordinator_instance.__aenter__ = AsyncMock(return_value=coordinator_instance)
    coordinator_instance.__aexit__ = AsyncMock()
    return coordinator_instance


def setup_event_handling():
    mock_asyncio_event_cls = patch("aider_mcp_server.sse_server.asyncio.Event").start()
    mock_event_instance_returned = mock_asyncio_event_cls.return_value
    mock_event_instance_returned.is_set = MagicMock(return_value=False)
    mock_event_instance_returned.wait = AsyncMock(side_effect=wait_side_effect)
    mock_event_instance_returned.set = MagicMock(side_effect=set_side_effect)
    return mock_asyncio_event_cls, mock_event_instance_returned


async def wait_side_effect(mock_event_instance_returned):
    while not mock_event_instance_returned.is_set():
        await asyncio.sleep(0.001)


def set_side_effect(mock_event_instance_returned):
    mock_event_instance_returned.is_set.return_value = True


# Test function
@pytest.mark.asyncio
async def test_serve_sse_startup_and_run():
    mock_server_logger, mock_adapter_logger = setup_test_loggers()
    coordinator_instance = setup_mock_coordinator()
    mock_asyncio_event_cls, mock_event_instance_returned = setup_event_handling()

    host, port, editor_model, cwd, heartbeat = (
        "127.0.0.1",
        8888,
        "test_model",
        "/test/repo",
        20.0,
    )

    with patch("aider_mcp_server.sse_server.is_git_repository", return_value=(True, None)):
        await serve_sse(host, port, editor_model, cwd, heartbeat_interval=heartbeat)

    mock_coordinator_instance = coordinator_instance
    mock_coordinator_instance.__aenter__.assert_awaited_once()

    expected_handler_calls = [
        call("aider_ai_code", ANY, required_permission=Permissions.EXECUTE_AIDER),
        call("list_models", ANY),
    ]
    assert mock_coordinator_instance.register_handler.call_count == 2
    mock_coordinator_instance.register_handler.assert_has_calls(expected_handler_calls)

    assert mock_coordinator_instance.register_transport.call_count >= 1
    adapter_instance_registered = mock_coordinator_instance.register_transport.call_args_list[0][0][1]
    assert isinstance(adapter_instance_registered, SSETransportAdapter)
    assert adapter_instance_registered.transport_id.startswith("sse_")

    mock_coordinator_instance.__aexit__.assert_awaited_once()
