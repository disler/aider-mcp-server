import asyncio
import signal
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Coroutine, Optional

from aider_mcp_server.atoms.logging import Logger, get_logger
from aider_mcp_server.security import SecurityContext
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


async def run_sse_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    security_context: Optional[SecurityContext] = None,
    log_level: str = "INFO",
    editor_model: str = "",
    current_working_dir: str = "",
) -> None:
    """
    Run an SSE server using the official MCP SDK approach.

    Args:
        host: Host address to listen on
        port: Port to listen on
        security_context: Security context for the server
        log_level: Logging level
        editor_model: Model identifier to use for editing operations
        current_working_dir: Working directory (must be a git repository)
    """
    # Validate working directory if provided
    if current_working_dir:
        logger.info(f"Validating working directory: {current_working_dir}")
        is_repo, error_msg = is_git_repository(Path(current_working_dir))
        if not is_repo:
            error_message = (
                f"Error: The specified directory '{current_working_dir}' is not a valid git repository: {error_msg}"
            )
            logger.critical(error_message)
            raise ValueError(error_message)
        logger.info(f"Working directory '{current_working_dir}' is a valid git repository.")

    # Get coordinator instance
    coordinator = await ApplicationCoordinator.getInstance(get_logger)

    # Use the coordinator in async context
    async with coordinator:
        logger.info("Coordinator context entered")

        # Create the SSE adapter with coordinator
        sse_adapter = SSETransportAdapter(coordinator=coordinator, host=host, port=port, get_logger=get_logger)

        # Initialize the adapter (this will create the FastMCP server)
        await sse_adapter.initialize()

        # Register the adapter with the coordinator
        await coordinator.register_transport(sse_adapter.get_transport_id(), sse_adapter)
        logger.info("Registered SSE transport with coordinator")

        # Start the SSE server
        await sse_adapter.start_listening()
        logger.info(f"SSE server listening on {host}:{port}")

        # Setup shutdown event
        shutdown_event = asyncio.Event()

        async def handle_shutdown() -> None:
            """Handle graceful shutdown"""
            logger.info("Initiating graceful shutdown...")
            shutdown_event.set()

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):

            def create_handler(s: int = sig) -> None:
                asyncio.create_task(handle_shutdown())

            loop.add_signal_handler(sig, create_handler)

        # Wait for shutdown signal and server task
        try:
            # Get the server task from the adapter if available
            server_task = getattr(sse_adapter, "_server_task", None)
            if server_task:
                await asyncio.gather(shutdown_event.wait(), server_task, return_exceptions=True)
            else:
                await shutdown_event.wait()
            logger.info("Shutdown event received. Closing SSE server...")
        except asyncio.CancelledError:
            logger.info("Server tasks cancelled")
        except Exception as e:
            logger.error(f"Error during server operation: {e}")
        finally:
            # Shutdown the adapter
            await sse_adapter.shutdown()
            logger.info("SSE server shutdown complete")


# Keep the old function signature for compatibility
async def serve_sse(
    host: str,
    port: int,
    editor_model: str,
    current_working_dir: str,
    heartbeat_interval: float = 15.0,
) -> None:
    """
    Compatibility wrapper for the old serve_sse function.
    """
    await run_sse_server(host=host, port=port, editor_model=editor_model, current_working_dir=current_working_dir)
