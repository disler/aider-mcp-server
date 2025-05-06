import asyncio
import signal
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest
from starlette.routing import Route

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

    with patch.object(sse_server_module, "_test_handle_shutdown_signal", AsyncMock()) as mock_handle_shutdown_signal_func:
        await serve_sse(host, port, editor_model, cwd, heartbeat_interval=heartbeat)

    mock_get_instance_sse.stop()

    mock_coordinator_instance = mock_get_instance_sse.return_value

    mock_coordinator_instance.__aenter__.assert_awaited_once()

    expected_handler_calls = [
        call("aider_ai_code", ANY, required_permission=Permissions.EXECUTE_AIDER),
        call("list_models", ANY)
    ]
    mock_coordinator_instance.register_handler.assert_has_awaits(expected_handler_calls, any_order=True)

    mock_coordinator_instance.register_transport.assert_called_once()
    adapter_instance_registered = mock_coordinator_instance.register_transport.call_args[0][1]
    assert isinstance(adapter_instance_registered, SSETransportAdapter)
    assert adapter_instance_registered.transport_id.startswith("sse_")

    mock_uvicorn_config_cls = patch("aider_mcp_server.sse_server.UvicornConfig").start()
    mock_starlette_app = MagicMock()
    mock_starlette_cls = patch("aider_mcp_server.sse_server.Starlette").start()
    mock_starlette_cls.assert_called_once()
    call_args, call_kwargs = mock_starlette_cls.call_args
    assert "routes" in call_kwargs
    routes_arg = call_kwargs["routes"]
    assert isinstance(routes_arg, list)
    assert len(routes_arg) == 2
    assert isinstance(routes_arg[0], Route)
    assert routes_arg[0].path == "/sse"
    assert isinstance(routes_arg[1], Route)
    assert routes_arg[1].path == "/message"
    assert routes_arg[1].methods == {"POST"}

    mock_uvicorn_config_cls.assert_called_once_with(
        app=mock_starlette_app,
        host=host,
        port=port,
        log_config=None,
        handle_signals=False
    )

    mock_uvicorn_server_cls = patch("aider_mcp_server.sse_server.UvicornServer").start()
    mock_uvicorn_server_cls.assert_called_once_with(mock_uvicorn_config_cls)

    mock_get_loop = patch("aider_mcp_server.sse_server.asyncio.get_event_loop").start()
    mock_get_loop.assert_called_once()
    mock_create_shutdown_wrapper = patch("aider_mcp_server.sse_server.create_shutdown_wrapper").start()
    assert mock_create_shutdown_wrapper.call_count == 2
    mock_create_shutdown_wrapper.assert_any_call(signal.SIGINT, mock_handle_shutdown_signal_func, mock_event_instance_returned)
    mock_create_shutdown_wrapper.assert_any_call(signal.SIGTERM, mock_handle_shutdown_signal_func, mock_event_instance_returned)

    mock_loop = mock_get_loop.return_value
    if hasattr(mock_loop, "add_signal_handler") and mock_loop.add_signal_handler.called:
        mock_loop.add_signal_handler.assert_any_call(signal.SIGINT, signal.SIG_IGN)
        mock_loop.add_signal_handler.assert_any_call(signal.SIGTERM, signal.SIG_IGN)
    else:
        mock_signal_signal = patch("aider_mcp_server.sse_server.signal.signal").start()
        mock_signal_signal.assert_any_call(signal.SIGINT, signal.SIG_IGN)
        mock_signal_signal.assert_any_call(signal.SIGTERM, signal.SIG_IGN)

    mock_uvicorn_server_instance = mock_uvicorn_server_cls.return_value
    mock_uvicorn_server_instance.serve.assert_awaited_once()
    mock_event_instance_returned.wait.assert_awaited_once()

    assert mock_uvicorn_server_instance.should_exit is True

    mock_coordinator_instance.__aexit__.assert_awaited_once()

    if hasattr(mock_loop, "remove_signal_handler") and mock_loop.remove_signal_handler.called:
        mock_loop.remove_signal_handler.assert_any_call(signal.SIGINT)
        mock_loop.remove_signal_handler.assert_any_call(signal.SIGTERM)
    else:
        mock_signal_signal.assert_any_call(signal.SIGINT, signal.SIG_DFL)
        mock_signal_signal.assert_any_call(signal.SIGTERM, signal.SIG_DFL)

    mock_handle_shutdown_signal_func.assert_not_awaited()
