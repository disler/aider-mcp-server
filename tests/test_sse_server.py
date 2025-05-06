import asyncio
import signal
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest
from aider_mcp_server.sse_server import Starlette, uvicorn
from starlette.routing import Route  # Added import for Route

import tests.monkey_patch_sse  # Apply monkey patches for SSE tests

import aider_mcp_server.sse_server as sse_server_module
from aider_mcp_server.security import Permissions
from aider_mcp_server.sse_server import serve_sse
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.transport_adapter import LoggerProtocol
from aider_mcp_server.transport_coordinator import ApplicationCoordinator

# Helper functions

def setup_test_loggers():
    mock_server_logger = MagicMock(spec=LoggerProtocol)
    mock_server_logger.info = MagicMock()
    mock_server_logger.debug = MagicMock()
    mock_server_logger.warning = MagicMock()
    mock_server_logger.error = MagicMock()

    mock_adapter_logger = MagicMock(spec=LoggerProtocol)
    mock_adapter_logger.info = MagicMock()
    mock_adapter_logger.debug = MagicMock()
    mock_adapter_logger.warning = MagicMock()
    mock_adapter_logger.error = MagicMock()

    return mock_server_logger, mock_adapter_logger

def setup_mock_coordinator():
    coordinator_instance = MagicMock(spec=ApplicationCoordinator)
    coordinator_instance.register_transport = MagicMock()
    coordinator_instance.unregister_transport = MagicMock()
    coordinator_instance.subscribe_to_event_type = MagicMock()
    coordinator_instance.is_shutting_down = MagicMock(return_value=False)

    coordinator_instance.start_request = AsyncMock()
    coordinator_instance.fail_request = AsyncMock()
    coordinator_instance.shutdown = AsyncMock()
    coordinator_instance.wait_for_initialization = AsyncMock()
    coordinator_instance.register_handler = AsyncMock()

    coordinator_instance.__aenter__ = AsyncMock(return_value=coordinator_instance)
    coordinator_instance.__aexit__ = AsyncMock()

    patcher_sse_server = patch(
        "aider_mcp_server.sse_server.ApplicationCoordinator.getInstance",
        new_callable=AsyncMock,
        return_value=coordinator_instance
    )

    mock_get_instance_sse = patcher_sse_server.start()

    return coordinator_instance, mock_get_instance_sse

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
    coordinator_instance, mock_get_instance_sse = setup_mock_coordinator()
    mock_asyncio_event_cls, mock_event_instance_returned = setup_event_handling()

    host, port, editor_model, cwd, heartbeat = "127.0.0.1", 8888, "test_model", "/test/repo", 20.0

    # Patch is_git_repository directly where it's used in sse_server
    with patch("aider_mcp_server.sse_server.is_git_repository", return_value=(True, None)), \
         patch.object(sse_server_module, "_test_handle_shutdown_signal", AsyncMock()) as mock_handle_shutdown_signal_func:
        await serve_sse(host, port, editor_model, cwd, heartbeat_interval=heartbeat)

    # Verify that coordinator was used correctly
    mock_coordinator_instance = mock_get_instance_sse.return_value
    mock_coordinator_instance.__aenter__.assert_awaited_once()

    # Verify the handlers were registered
    expected_handler_calls = [
        call("aider_ai_code", ANY, required_permission=Permissions.EXECUTE_AIDER),
        call("list_models", ANY)
    ]
    mock_coordinator_instance.register_handler.assert_has_awaits(expected_handler_calls, any_order=True)

    # Verify the SSE adapter was registered
    mock_coordinator_instance.register_transport.assert_called_once()
    adapter_instance_registered = mock_coordinator_instance.register_transport.call_args[0][1]
    assert isinstance(adapter_instance_registered, SSETransportAdapter)
    assert adapter_instance_registered.transport_id.startswith("sse_")

    # Verify coordinator cleanup was called
    mock_coordinator_instance.__aexit__.assert_awaited_once()
