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
        logger.debug("Signaling main loop to shut down via event...")
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
        logger.debug(f"Validating working directory: {current_working_dir}")
        is_repo, error_msg = is_git_repository(Path(current_working_dir))
        if not is_repo:
            error_message = (
                f"Error: The specified directory '{current_working_dir}' is not a valid git repository: {error_msg}"
            )
            logger.critical(error_message)
            raise ValueError(error_message)
        logger.debug(f"Working directory '{current_working_dir}' is a valid git repository.")

    # Get coordinator instance
    coordinator = await ApplicationCoordinator.getInstance(get_logger)

    logger.debug("Coordinator instance obtained")

    # Create the SSE adapter with coordinator
    sse_adapter = SSETransportAdapter(
        coordinator=coordinator,
        host=host,
        port=port,
        get_logger=get_logger,
        editor_model=editor_model,
        current_working_dir=current_working_dir,
    )

    initialized_successfully = False
    try:
        # Initialize the adapter (this will create the FastMCP server and register with coordinator)
        await sse_adapter.initialize()
        initialized_successfully = True  # Set flag only after successful initialization

        # Start the SSE server
        await sse_adapter.start_listening()
        logger.debug(f"SSE server listening on {host}:{port}")

        # Setup shutdown event
        shutdown_event = asyncio.Event()

        async def handle_shutdown() -> None:
            """Handle graceful shutdown by setting the event."""
            logger.info("Graceful shutdown initiated by signal.")
            if not shutdown_event.is_set():
                shutdown_event.set()

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig_val in (signal.SIGTERM, signal.SIGINT):
            # Capture sig_val correctly in the closure
            def create_handler(s: int = sig_val) -> None:
                logger.debug(f"Signal handler for {signal.Signals(s).name} creating task for handle_shutdown.")
                asyncio.create_task(handle_shutdown())

            loop.add_signal_handler(sig_val, create_handler)

        # Wait for shutdown signal and server task (nested try for server operation phase)
        try:
            server_task = getattr(sse_adapter, "_server_task", None)
            if server_task:
                logger.debug("Waiting on shutdown_event and server_task.")
                # Gather tasks, return_exceptions=True allows us to inspect individual task outcomes
                results = await asyncio.gather(shutdown_event.wait(), server_task, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        task_name = "shutdown_event.wait()" if i == 0 else "server_task"
                        if not isinstance(result, asyncio.CancelledError):
                            logger.error(f"Exception in gathered task '{task_name}': {result}", exc_info=result)
                            raise result # Propagate critical exceptions
                        else:
                            logger.info(f"Task '{task_name}' was cancelled.")
            else:
                logger.debug("Waiting on shutdown_event (no server_task).")
                await shutdown_event.wait()
            
            logger.info("Shutdown event processed or server task completed. Proceeding to close server.")

        except asyncio.CancelledError:
            logger.info("Server operation tasks were cancelled.")
            # Do not re-raise CancelledError from here; allow graceful shutdown path via finally.
        except Exception as e:
            logger.error(f"Error during server operation: {e}", exc_info=True)
            raise # Re-raise to be caught by the outer exception handler, ensuring cleanup

    except Exception as e:
        # This outer except catches:
        # - Errors from initialize() (initialized_successfully will be False)
        # - Errors from start_listening() (initialized_successfully will be True)
        # - Errors re-raised from the inner try's exception block (initialized_successfully will be True)
        # - Any other unexpected errors in the main try block before the inner try
        logger.error(f"Unhandled exception in run_sse_server: {e}", exc_info=True)
        raise # Re-raise the exception to signal failure to the caller
    finally:
        # This is the new outer finally block.
        # It will execute regardless of where an exception occurred in the try block,
        # or if no exception occurred.
        if initialized_successfully:
            logger.info("Attempting to shut down SSE adapter (as initialization was successful)...")
            try:
                await sse_adapter.shutdown()
                logger.info("SSE adapter shutdown process completed.")
            except Exception as e_shutdown:
                logger.error(f"Error during SSE adapter shutdown: {e_shutdown}", exc_info=True)
        else:
            logger.info("SSE adapter initialization did not complete successfully; skipping shutdown.")


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
