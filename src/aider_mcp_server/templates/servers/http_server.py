import asyncio
import signal
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Coroutine, Optional

from ...atoms.logging.logger import Logger, get_logger
from ...atoms.security.context import SecurityContext
from ...atoms.utils.config_constants import DEFAULT_HTTP_HOST, DEFAULT_HTTP_PORT
from ...organisms.transports.http.http_streamable_transport_adapter import (
    HttpStreamableTransportAdapter,
)
from ...pages.application.coordinator import ApplicationCoordinator

# Assuming is_git_repository is in a 'server.py' file in the same directory
# For example, if your project structure is:
# your_package/
#   servers/
#     http_server.py
#     sse_server.py
#     server.py  <- contains is_git_repository
# then the import below is correct.
# If is_git_repository is located elsewhere, this import will need adjustment.
from .server import is_git_repository

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
    log_message = f"Received signal {received_signal_name}. Initiating graceful shutdown for HTTP server..."
    logger.warning(log_message)

    if not event.is_set():
        logger.debug("Signaling HTTP server main loop to shut down via event...")
        event.set()
    else:
        logger.warning("HTTP server shutdown already in progress.")


async def _handle_shutdown_signal_for_test(
    sig: int, signum: Optional[int] = None, frame: Optional[FrameType] = None
) -> None:
    """Test-only handler for shutdown signals for HTTP server."""
    signal_name_for_handler = signal.Signals(sig).name
    if signum is not None:
        received_signal_name = signal.Signals(signum).name
    else:
        received_signal_name = signal_name_for_handler
    logger.debug(
        f"Test HTTP server shutdown handler called for signal {received_signal_name} (handler for {signal_name_for_handler})."
    )


def _create_shutdown_task_wrapper(
    sig: int,
    async_handler: Callable[..., Coroutine[Any, Any, None]],
    event: Optional[asyncio.Event] = None,
) -> Callable[[], None]:
    """
    Create a synchronous wrapper for an async signal handler for HTTP server.

    Args:
        sig: Signal number
        async_handler: Async function to call
        event: Event to pass to the handler

    Returns:
        Synchronous function that can be registered as a signal handler
    """

    def sync_wrapper() -> None:
        signum = sig
        frame = None
        logger.debug(f"Sync wrapper called for signal {sig} for HTTP server. Scheduling async handler.")

        if event is not None:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running() and not loop.is_closed():
                    logger.debug(f"Scheduling async handler for signal {sig} with event for HTTP server.")
                    loop.create_task(async_handler(sig, event, signum, frame))
                else:
                    logger.warning(
                        f"Event loop not running or closed when handling signal {sig} for HTTP server. Cannot schedule async handler."
                    )
            except RuntimeError as e:
                logger.error(f"Error getting running loop for signal {sig} for HTTP server: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error scheduling async handler for signal {sig} with event for HTTP server: {e}",
                    exc_info=True,
                )
        else:
            logger.debug(f"Scheduling async handler for signal {sig} without event for HTTP server (test mode?).")
            try:
                asyncio.create_task(async_handler(sig, signum, frame))
            except RuntimeError as e:
                logger.error(f"Error calling asyncio.create_task directly for signal {sig} in HTTP server test mode: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error scheduling async handler for signal {sig} without event for HTTP server: {e}",
                    exc_info=True,
                )

    return sync_wrapper


async def _validate_working_directory(current_working_dir: str) -> None:
    """Validate that the current working directory is a git repository."""
    if current_working_dir:
        logger.debug(f"Validating working directory for HTTP server: {current_working_dir}")
        is_repo, error_msg = is_git_repository(Path(current_working_dir))
        if not is_repo:
            error_message = (
                f"Error: The specified directory '{current_working_dir}' for HTTP server is not a valid git repository: {error_msg}"
            )
            logger.critical(error_message)
            raise ValueError(error_message)
        logger.debug(f"Working directory '{current_working_dir}' for HTTP server is a valid git repository.")


async def _setup_http_adapter(
    host: str,
    port: int,
    editor_model: str,
    current_working_dir: str,
    # heartbeat_interval is managed by HttpStreamableTransportAdapter's default
) -> HttpStreamableTransportAdapter:
    """Initialize and prepare the HTTP adapter."""
    coordinator = ApplicationCoordinator()
    logger.debug("HTTP Server: ApplicationCoordinator instance obtained.")

    # Initialize the coordinator's internal state.
    # This is crucial as it sets up locks and other necessary components.
    # Registering in discovery allows other services to find this server.
    await coordinator._initialize_coordinator(host=host, port=port, register_in_discovery=True)
    logger.debug("HTTP Server: ApplicationCoordinator internal state initialized.")

    http_adapter = HttpStreamableTransportAdapter(
        coordinator=coordinator,
        host=host,
        port=port,
        get_logger=get_logger, # Pass the get_logger function from this module's scope
        editor_model=editor_model,
        current_working_dir=current_working_dir,
        # stream_queue_size uses its default in HttpStreamableTransportAdapter
        # heartbeat_interval uses its default in HttpStreamableTransportAdapter
    )
    await http_adapter.initialize()
    # start_listening will be called in run_http_server after this function returns
    return http_adapter


def _setup_signal_handlers(loop: asyncio.AbstractEventLoop, shutdown_event: asyncio.Event) -> None:
    """Setup signal handlers for graceful shutdown of the HTTP server."""

    async def handle_shutdown() -> None:
        """Handle graceful shutdown by setting the event."""
        logger.info("HTTP server: Graceful shutdown initiated by signal.")
        if not shutdown_event.is_set():
            shutdown_event.set()

    for sig_val in (signal.SIGTERM, signal.SIGINT):

        def create_handler(s: int = sig_val) -> None:
            logger.debug(f"HTTP server: Signal handler for {signal.Signals(s).name} creating task for handle_shutdown.")
            asyncio.create_task(handle_shutdown())

        loop.add_signal_handler(sig_val, create_handler)


async def _wait_for_shutdown(http_adapter: HttpStreamableTransportAdapter, shutdown_event: asyncio.Event) -> None:
    """Wait for the shutdown signal and HTTP server task to complete."""
    SHUTDOWN_TIMEOUT = 10.0  # 10 seconds timeout

    server_task = getattr(http_adapter, "_server_task", None)
    if server_task:
        logger.debug("HTTP server: Waiting on shutdown_event and server_task.")
        try:
            results = await asyncio.wait_for(
                asyncio.gather(shutdown_event.wait(), server_task, return_exceptions=True), timeout=SHUTDOWN_TIMEOUT
            )
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    task_name = "shutdown_event.wait()" if i == 0 else "server_task"
                    if not isinstance(result, asyncio.CancelledError):
                        logger.error(f"HTTP server: Exception in gathered task '{task_name}': {result}", exc_info=result)
                        raise result # Propagate significant errors
                    else:
                        logger.info(f"HTTP server: Task '{task_name}' was cancelled.")
        except asyncio.TimeoutError:
            logger.warning(f"HTTP server: Shutdown wait timed out after {SHUTDOWN_TIMEOUT}s, forcing shutdown.")
            if server_task and not server_task.done():
                server_task.cancel()
                await asyncio.sleep(0.1) # Give cancellation a moment
    else:
        logger.debug("HTTP server: Waiting on shutdown_event (no server_task found).")
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=SHUTDOWN_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"HTTP server: Shutdown event wait timed out after {SHUTDOWN_TIMEOUT}s.")
    logger.info("HTTP server: Shutdown event processed or server task completed. Proceeding to close server.")


async def run_http_server(
    host: str = DEFAULT_HTTP_HOST,
    port: int = DEFAULT_HTTP_PORT,
    security_context: Optional[SecurityContext] = None, # Kept for structural consistency
    log_level: str = "INFO", # Kept for structural consistency, actual logging level managed by logger config
    editor_model: str = "",
    current_working_dir: str = "",
    # heartbeat_interval is not taken here, adapter uses its default.
) -> None:
    """
    Run an HTTP Streamable server.

    Args:
        host: Host address to listen on.
        port: Port to listen on.
        security_context: Security context for the server (placeholder).
        log_level: Logging level (placeholder, managed by logger config).
        editor_model: Model identifier for editing operations.
        current_working_dir: Working directory (must be a git repository).
    """
    await _validate_working_directory(current_working_dir)

    http_adapter: Optional[HttpStreamableTransportAdapter] = None
    adapter_initialize_succeeded = False
    try:
        http_adapter = await _setup_http_adapter(
            host=host,
            port=port,
            editor_model=editor_model,
            current_working_dir=current_working_dir,
        )
        adapter_initialize_succeeded = True

        await http_adapter.start_listening()
        actual_port = http_adapter.get_actual_port() or port # Use actual port if available
        logger.info(f"HTTP Streamable server listening on http://{host}:{actual_port}")

        shutdown_event = asyncio.Event()
        loop = asyncio.get_event_loop()
        _setup_signal_handlers(loop, shutdown_event)

        try:
            await _wait_for_shutdown(http_adapter, shutdown_event)
        except asyncio.CancelledError:
            logger.info("HTTP server: Main operation tasks were cancelled.")
        except Exception as e:
            logger.error(f"HTTP server: Error during server operation: {e}", exc_info=True)
            # Decide if to re-raise or handle; re-raising for now
            raise

    except Exception as e:
        logger.error(f"HTTP server: Unhandled exception in run_http_server: {e}", exc_info=True)
        raise # Propagate critical startup errors
    finally:
        if adapter_initialize_succeeded and http_adapter:
            logger.info("HTTP server: Adapter's initialize() was successful. Attempting shutdown...")
            try:
                await http_adapter.shutdown()
                logger.info("HTTP server: Adapter shutdown process completed.")
            except Exception as e_shutdown:
                logger.error(f"HTTP server: Error during adapter shutdown: {e_shutdown}", exc_info=True)
        elif http_adapter and not adapter_initialize_succeeded:
             logger.info("HTTP server: Adapter's initialize() failed. Attempting shutdown of partially initialized adapter...")
             try:
                await http_adapter.shutdown() # Attempt shutdown even if initialize failed
                logger.info("HTTP server: Partial adapter shutdown process completed.")
             except Exception as e_shutdown:
                logger.error(f"HTTP server: Error during partial adapter shutdown: {e_shutdown}", exc_info=True)
        else:
            logger.info(
                "HTTP server: Adapter's initialize() did not complete or adapter was not created; skipping shutdown."
            )


async def serve_http(
    host: str,
    port: int,
    editor_model: str,
    current_working_dir: str,
    heartbeat_interval: float = 30.0, # Kept for structural consistency, but not used by run_http_server
) -> None:
    """
    Compatibility wrapper for running the HTTP server.

    Args:
        host: Host address.
        port: Port number.
        editor_model: Editor model identifier.
        current_working_dir: Current working directory.
        heartbeat_interval: Heartbeat interval (parameter kept for consistency, adapter uses its own default).
    """
    logger.info(
        f"serve_http called with host={host}, port={port}, editor_model='{editor_model}', cwd='{current_working_dir}'. Heartbeat interval param value {heartbeat_interval} is noted but HttpStreamableTransportAdapter will use its internal default."
    )
    await run_http_server(
        host=host,
        port=port,
        editor_model=editor_model,
        current_working_dir=current_working_dir,
    )
