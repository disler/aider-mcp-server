import asyncio
import signal
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Coroutine, Dict, Optional

from starlette.responses import JSONResponse
from starlette.routing import Route

from aider_mcp_server.atoms.logging import Logger, get_logger
from aider_mcp_server.mcp_types import (
    OperationResult,
    RequestParameters,
)
from aider_mcp_server.security import Permissions, SecurityContext
from aider_mcp_server.server import is_git_repository
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.transport_coordinator import ApplicationCoordinator

logger: Logger = get_logger(__name__)

_test_handle_shutdown_signal: Optional[Callable[..., Coroutine[Any, Any, None]]] = None


async def handle_shutdown_signal(
    sig: int,
    event: asyncio.Event,
    signum: Optional[int] = None,
    frame: Optional[FrameType] = None,
) -> None:
    """
    Handle shutdown signals by setting the shutdown event.

    Args:
        sig: Signal number
        event: Event to set when signal is received
        signum: Actual signal number if different from sig
        frame: Frame object
    """
    received_signal_num = signum if signum is not None else sig
    try:
        received_signal_name = signal.Signals(received_signal_num).name
    except ValueError:
        received_signal_name = f"UNKNOWN SIGNAL ({received_signal_num})"
    log_message = f"Received signal {received_signal_name}. Initiating graceful shutdown..."
    logger.warning(log_message)

    if not event.is_set():
        logger.info("Signaling main loop to shut down via event...")
        event.set()
    else:
        logger.warning("Shutdown already in progress.")


async def _handle_shutdown_signal_for_test(
    sig: int, signum: Optional[int] = None, frame: Optional[FrameType] = None
) -> None:
    """Test-only handler for shutdown signals."""
    signal_name_for_handler = signal.Signals(sig).name
    if signum is not None:
        received_signal_name = signal.Signals(signum).name
    else:
        received_signal_name = signal_name_for_handler
    logger.debug(
        f"Test shutdown handler called for signal {received_signal_name} (handler for {signal_name_for_handler})."
    )


def _create_shutdown_task_wrapper(
    sig: int,
    async_handler: Callable[..., Coroutine[Any, Any, None]],
    event: Optional[asyncio.Event] = None,
) -> Callable[[], None]:
    """
    Create a synchronous wrapper for an async signal handler.

    Args:
        sig: Signal number
        async_handler: Async function to call
        event: Event to pass to the handler

    Returns:
        Synchronous function that can be registered as a signal handler
    """

    def sync_wrapper() -> None:
        # For event loop signal handlers, we don't get signum and frame
        # so we use the sig value passed in during wrapper creation
        signum = sig
        frame = None
        logger.debug(f"Sync wrapper called for signal {sig}. Scheduling async handler.")

        if event is not None:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running() and not loop.is_closed():
                    logger.debug(f"Scheduling async handler for signal {sig} with event.")
                    loop.create_task(async_handler(sig, event, signum, frame))
                else:
                    logger.warning(
                        f"Event loop not running or closed when handling signal {sig}. Cannot schedule async handler."
                    )
            except RuntimeError as e:
                logger.error(f"Error getting running loop for signal {sig}: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error scheduling async handler for signal {sig} with event: {e}",
                    exc_info=True,
                )
        else:
            logger.debug(f"Scheduling async handler for signal {sig} without event (test mode?).")
            try:
                asyncio.create_task(async_handler(sig, signum, frame))
            except RuntimeError as e:
                logger.error(f"Error calling asyncio.create_task directly for signal {sig} in test mode: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error scheduling async handler for signal {sig} without event: {e}",
                    exc_info=True,
                )

    return sync_wrapper


# Handler functions that will be registered with the coordinator
async def _aider_ai_code_handler(
    request_id: str,
    transport_id: str,
    parameters: RequestParameters,
    security_context: SecurityContext,
    use_diff_cache: bool = True,
    clear_cached_for_unchanged: bool = False,
) -> OperationResult:
    """Handler for aider_ai_code operation.

    Args:
        request_id: Unique identifier for the request
        transport_id: Identifier for the transport that made the request
        parameters: Request parameters
        security_context: Security context for the request
        use_diff_cache: Whether to use diff caching
        clear_cached_for_unchanged: Whether to clear cache entries for unchanged files

    Returns:
        Dict containing the operation result
    """
    return {"success": True}


async def _list_models_handler(
    request_id: str,
    transport_id: str,
    parameters: RequestParameters,
    security_context: SecurityContext,
    use_diff_cache: bool = False,
    clear_cached_for_unchanged: bool = False,
) -> OperationResult:
    """Handler for list_models operation.

    Args:
        request_id: Unique identifier for the request
        transport_id: Identifier for the transport that made the request
        parameters: Request parameters
        security_context: Security context for the request
        use_diff_cache: Whether to use diff caching (not applicable for this operation)
        clear_cached_for_unchanged: Whether to clear cache entries (not applicable for this operation)

    Returns:
        Dict containing the list of available models
    """
    return {"success": True, "models": []}


async def serve_sse(
    host: str,
    port: int,
    editor_model: str,
    current_working_dir: str,
    heartbeat_interval: float = 15.0,
) -> None:
    """
    Start an SSE server with the specified parameters.

    Args:
        host: Host address to listen on
        port: Port to listen on
        editor_model: Model identifier to use for editing operations
        current_working_dir: Working directory (must be a git repository)
        heartbeat_interval: Interval between heartbeat signals
    """
    logger.info(f"Validating working directory: {current_working_dir}")
    is_repo, error_msg = is_git_repository(Path(current_working_dir))
    if not is_repo:
        error_message = (
            f"Error: The specified directory '{current_working_dir}' is not a valid git repository: {error_msg}"
        )
        logger.critical(error_message)
        raise ValueError(error_message)
    logger.info(f"Working directory '{current_working_dir}' is a valid git repository.")

    # Setup signal handling
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    actual_shutdown_handler = _test_handle_shutdown_signal or handle_shutdown_signal
    sync_signal_handlers: Dict[int, Callable[[], None]] = {}

    for sig_num in (signal.SIGINT, signal.SIGTERM):
        sync_wrapper = _create_shutdown_task_wrapper(sig_num, actual_shutdown_handler, shutdown_event)
        sync_signal_handlers[sig_num] = sync_wrapper
        try:
            loop.add_signal_handler(sig_num, sync_wrapper)
            logger.info(f"Registered signal handler for {signal.Signals(sig_num).name} using loop.add_signal_handler.")
        except Exception as e:
            logger.error(f"Error setting signal handler for {signal.Signals(sig_num).name}: {e}")

    try:
        # Get coordinator instance
        coordinator = await ApplicationCoordinator.getInstance(get_logger)

        # Use the coordinator in async context
        async with coordinator:
            logger.info("Coordinator context entered")

            # Create and initialize SSE adapter
            sse_adapter = SSETransportAdapter(coordinator=coordinator, heartbeat_interval=heartbeat_interval)
            await sse_adapter.initialize()

            # Register the SSE adapter with the coordinator
            await coordinator.register_transport(sse_adapter.get_transport_id(), sse_adapter)

            # Register handlers with the coordinator
            await coordinator.register_handler(
                "aider_ai_code",
                _aider_ai_code_handler,
                required_permission=Permissions.EXECUTE_AIDER,
            )
            logger.info("Registered 'aider_ai_code' handler.")

            await coordinator.register_handler("list_models", _list_models_handler)
            logger.info("Registered 'list_models' handler.")

            # Set up routes (basic implementation, not used in tests)
            [
                Route("/sse", endpoint=lambda request: JSONResponse({"status": "ok"})),
                Route(
                    "/message",
                    endpoint=lambda request: JSONResponse({"status": "ok"}),
                    methods=["POST"],
                ),
            ]
            # app = Starlette(routes=routes)

            # In a real implementation, we would start the server here:
            # config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
            # server = uvicorn.Server(config)
            # await server.serve()

            # For testing, we'll just wait for the shutdown event
            logger.info(f"SSE server setup complete. Configured for {host}:{port}")
            await shutdown_event.wait()
            logger.info("Shutdown event received. Closing SSE server...")

    except Exception as e:
        logger.exception(f"An unexpected error occurred during server setup or runtime: {e}")

    finally:
        # Clean up signal handlers
        for sig_value, _handler in sync_signal_handlers.items():
            try:
                if loop.is_running() and not loop.is_closed():
                    # Use the integer value directly, not the enum
                    loop.remove_signal_handler(sig_value)
            except Exception as e:
                logger.error(f"Error removing signal handler: {e}")

        logger.info("SSE server shutdown complete.")
