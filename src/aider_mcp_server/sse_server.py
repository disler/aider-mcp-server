import asyncio
import json
import logging
import signal
import sys
import typing
import uuid
import os
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import partial
from pathlib import Path # Import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import uvicorn # type: ignore
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse # Import JSONResponse
# sse_starlette is used by the adapter, not directly here anymore
# from sse_starlette.sse import EventSourceResponse, ServerSentEvent

# Use absolute imports from the package root
from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.atoms.utils.fallback_config import get_retry_delay
# Correct the import path for is_git_repository
from aider_mcp_server.server import is_git_repository # Import repo check
from aider_mcp_server.security import SecurityContext, Permissions # Import Permissions
# Import the correct adapter
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter
# Import ApplicationCoordinator directly at the module level for patching
from aider_mcp_server.transport_coordinator import ApplicationCoordinator

# Keep TYPE_CHECKING block for potential static analysis benefits
if TYPE_CHECKING:
    # Use absolute import path (already imported above, but keep for clarity/mypy)
    from aider_mcp_server.transport_coordinator import ApplicationCoordinator


# Define the LoggerProtocol and get_logger_func setup locally
class LoggerProtocol(Protocol):
    def debug(self, message: str, **kwargs: Any) -> None: ...
    def info(self, message: str, **kwargs: Any) -> None: ...
    def warning(self, message: str, **kwargs: Any) -> None: ...
    def error(self, message: str, **kwargs: Any) -> None: ...
    def critical(self, message: str, **kwargs: Any) -> None: ...
    def exception(self, message: str, **kwargs: Any) -> None: ...

get_logger_func: Callable[..., LoggerProtocol]

try:
    # Use absolute import path
    from aider_mcp_server.atoms.logging import get_logger as custom_get_logger
    get_logger_func = typing.cast(Callable[..., LoggerProtocol], custom_get_logger)
except ImportError:
    def fallback_get_logger(name: str, *args: Any, **kwargs: Any) -> LoggerProtocol:
        logger = logging.getLogger(name)
        # Ensure basic configuration if fallback is used
        if not logging.root.handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Set level for this specific logger if needed
        # logger.setLevel(logging.INFO)
        return typing.cast(LoggerProtocol, logger)
    get_logger_func = fallback_get_logger
    temp_logger = logging.getLogger(__name__)
    temp_logger.warning(
        "Could not import custom logger from aider_mcp_server.atoms.logging. Using standard logging fallback."
    )

logger: LoggerProtocol = get_logger_func(__name__)


# --- FastAPI App and Server State ---
# Keep FastAPI app instance global for endpoint definition
_app = FastAPI()
# Store the single adapter instance globally within this module
_adapter: Optional[SSETransportAdapter] = None
# Coordinator instance will be retrieved via getInstance()
# _coordinator: Optional["ApplicationCoordinator"] = None # No longer needed as global


# --- FastAPI Endpoints ---
# Define endpoints at the module level, they will access the global _adapter

@_app.post("/message", status_code=202) # Match route used in tests
async def handle_message(request: Request) -> Response:
    """
    Endpoint to handle incoming message requests (like tool calls).
    Delegates processing and security validation to the SSETransportAdapter.
    """
    if not _adapter:
        logger.error("Message request received but SSE Transport Adapter is not initialized.")
        raise HTTPException(status_code=503, detail="Server not ready")
    if _adapter._coordinator and _adapter._coordinator.is_shutting_down():
        raise HTTPException(status_code=503, detail="Server is shutting down")

    # Delegate directly to the adapter's handler method
    # The adapter handles JSON parsing, validation, security, and coordinator interaction
    try:
        response = await _adapter.handle_message_request(request)
        return response
    except HTTPException as http_exc:
        # Re-raise FastAPI/Starlette HTTP exceptions directly
        raise http_exc
    except Exception as e:
        # Catch unexpected errors during adapter processing
        request_id = "unknown" # Try to get request_id if possible, otherwise default
        try:
            # Attempt to parse body again to get request_id for logging, ignore errors
            body = await request.json()
            if isinstance(body, dict):
                request_id = body.get("request_id", "unknown")
        except Exception:
            pass
        logger.exception(f"Unexpected error handling message request {request_id}: {e}")
        # Return a generic 500 error
        return JSONResponse(
            {"success": False, "error": "Internal server error processing message request"},
            status_code=500
        )


@_app.get("/sse") # Match route used in tests
async def event_stream(request: Request) -> Response:
    """
    Endpoint for clients to establish an SSE connection and receive events.
    Delegates connection handling to the SSETransportAdapter.
    """
    if not _adapter:
        logger.error("SSE connection attempt but SSE Transport Adapter is not initialized.")
        # Return an immediate error response, not SSE
        return Response(content="Server not ready", status_code=503, media_type="text/plain")
    if _adapter._coordinator and _adapter._coordinator.is_shutting_down():
         return Response(content="Server is shutting down", status_code=503, media_type="text/plain")

    # Delegate directly to the adapter's handler method
    # The adapter handles creating the connection, queue, generator, and EventSourceResponse
    try:
        response = await _adapter.handle_sse_request(request)
        return response
    except HTTPException as http_exc:
        # Re-raise FastAPI/Starlette HTTP exceptions directly
        raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error handling SSE connection request: {e}")
        # Return a generic 500 error
        return Response(content="Internal server error handling SSE request", status_code=500, media_type="text/plain")


# --- Shutdown Signal Handling ---

async def handle_shutdown_signal(
    sig: Union[signal.Signals, int], # Signal enum or number
    shutdown_event: asyncio.Event, # Event to signal shutdown completion/request
    *args: Any # Accept extra args from signal handler (like frame)
) -> None:
    """
    Asynchronous signal handler for graceful shutdown.

    Args:
        sig: The signal received (e.g., signal.SIGINT).
        shutdown_event: The asyncio.Event to set to signal shutdown.
        *args: Additional arguments passed by the signal handler (ignored).
    """
    signal_name = signal.Signals(sig).name if isinstance(sig, int) else sig.name
    logger.warning(f"Received signal {signal_name}. Initiating graceful shutdown...")

    # Only set the event. The main loop/context manager handles coordinator shutdown.
    if not shutdown_event.is_set():
        logger.info("Signaling main loop to shut down via event...")
        shutdown_event.set()
    else:
        logger.info("Shutdown event already set.")

def _create_shutdown_task_wrapper(
    sig: Union[signal.Signals, int], # Signal enum or number
    async_handler: Callable[..., typing.Awaitable[None]],
) -> Callable[..., None]:
    """
    Creates a synchronous wrapper for an async signal handler.

    This wrapper is suitable for use with `loop.add_signal_handler` or `signal.signal`.
    It schedules the async handler to run using `asyncio.create_task`.

    Args:
        sig: The signal the handler is for (passed to the async handler).
        async_handler: The awaitable function to call (e.g., handle_shutdown_signal).

    Returns:
        A synchronous function that takes signal args (signum, frame) and
        schedules the async_handler.
    """
    def sync_wrapper(*args: Any):
        """Synchronous function called by the signal mechanism."""
        # args typically contains (signal_number, stack_frame)
        logger.debug(f"Sync wrapper called for signal {sig} with args: {args}")
        # Create task to run the async handler, passing the original signal
        # object/enum and any args received by the sync wrapper.
        asyncio.create_task(async_handler(sig, *args))

    return sync_wrapper


# --- Core Server Logic (Exported Function) ---

async def serve_sse(
    host: str,
    port: int,
    editor_model: str, # Parameter required by __main__
    current_working_dir: str, # Parameter required by __main__
    heartbeat_interval: float = 15.0, # Add heartbeat interval parameter
) -> None:
    """
    Sets up and runs the SSE server using FastAPI and Uvicorn.

    Manages the Uvicorn server lifecycle, integrates with the ApplicationCoordinator,
    and handles graceful shutdown via signals.
    """
    global _adapter # Allow modification of the global adapter instance

    # ApplicationCoordinator is now imported at the module level
    # Remove local import:
    # from aider_mcp_server.transport_coordinator import ApplicationCoordinator

    # Import handlers (adjust path if necessary)
    from aider_mcp_server.handlers import handle_aider_request, handle_list_models

    # --- Configuration and Initialization ---
    logger.info(f"Starting SSE server configuration for http://{host}:{port}")
    logger.info(f"Config - Editor Model: {editor_model}, CWD: {current_working_dir}, Heartbeat: {heartbeat_interval}s")

    # Validate CWD is a git repository (as done in tests)
    is_repo, repo_path = is_git_repository(Path(current_working_dir))
    if not is_repo:
        logger.warning(f"Warning: Provided CWD '{current_working_dir}' is not a git repository.")
        # Decide if this is a fatal error or just a warning
        # For now, just log a warning.

    # Get the singleton instance of the coordinator
    coordinator = ApplicationCoordinator.getInstance()
    # Potentially configure coordinator if needed (example)
    # coordinator.set_config(editor_model=editor_model, cwd=current_working_dir)
    logger.info(f"Using ApplicationCoordinator instance: {id(coordinator)}")

    # Create and initialize the single SSE Transport Adapter instance
    _adapter = SSETransportAdapter(
        coordinator=coordinator,
        heartbeat_interval=heartbeat_interval,
        # sse_queue_size can be default or configured here
    )
    # Initialize the adapter (registers with coordinator, starts heartbeat)
    # This is now handled within the coordinator's context manager

    # Event to signal shutdown requested by signal handler
    shutdown_event = asyncio.Event()

    # --- Signal Handling Setup ---
    loop = asyncio.get_running_loop()
    # Create a partial function for the async handler, pre-filling the shutdown_event
    async_signal_handler_partial = partial(
        handle_shutdown_signal,
        shutdown_event=shutdown_event
    )
    # Store original handlers to restore them later
    original_handlers: Dict[int, Any] = {}

    for sig in (signal.SIGINT, signal.SIGTERM):
        # Store original handler
        try:
            original_handlers[sig] = signal.getsignal(sig)
        except (ValueError, OSError) as e: # Handle potential errors getting signal on some platforms
             logger.warning(f"Could not get original signal handler for {sig.name}: {e}")
             original_handlers[sig] = signal.SIG_DFL # Default handler

        # Create the synchronous wrapper for this specific signal and partial handler
        sync_signal_wrapper = _create_shutdown_task_wrapper(sig, async_signal_handler_partial)
        try:
            # Try loop handler first (preferred)
            loop.add_signal_handler(sig, sync_signal_wrapper)
            logger.debug(f"Registered signal handler for {sig.name} using loop.add_signal_handler")
        except NotImplementedError:
            # Fallback for Windows etc.
            # Use signal.signal carefully, might not be safe in threaded/asyncio context
            try:
                signal.signal(sig, sync_signal_wrapper) # type: ignore
                logger.debug(f"Registered signal handler for {sig.name} using signal.signal (fallback)")
            except (ValueError, OSError) as e:
                 logger.error(f"Could not set signal handler for {sig.name} using fallback: {e}")
        except (ValueError, OSError) as e:
             logger.error(f"Could not set signal handler for {sig.name}: {e}")


    # --- Uvicorn Server Configuration ---
    config = uvicorn.Config(
        app=_app, # Use the global FastAPI app instance
        host=host,
        port=port,
        log_config=None, # Disable Uvicorn's logging setup if using custom logger extensively
        # log_level="info", # Can be set if log_config is None
        lifespan="off", # Manage lifespan via coordinator context manager
        handle_signals=False, # Crucial: Disable Uvicorn's signal handling
        # loop="asyncio", # Let uvicorn/asyncio manage the loop
    )
    server = uvicorn.Server(config)

    # --- Main Server Execution Block ---
    server_task = None
    try:
        # Use the coordinator as an async context manager
        # This ensures adapter initialization and shutdown are handled correctly
        async with coordinator:
            logger.info("ApplicationCoordinator context entered.")

            # Register handlers *after* coordinator is initialized
            # Use Permissions enum for required permissions
            coordinator.register_handler("aider_ai_code", handle_aider_request, required_permission=Permissions.EXECUTE_AIDER)
            coordinator.register_handler("list_models", handle_list_models) # No permission needed by default
            logger.info("Request handlers registered with coordinator.")

            # Start the Uvicorn server task
            logger.info(f"Starting Uvicorn server task on http://{host}:{port}...")
            server_task = asyncio.create_task(server.serve(), name="uvicorn_server_task")

            # Wait for either the server task to complete or shutdown signal
            await asyncio.wait(
                [server_task, asyncio.create_task(shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # --- Shutdown Sequence ---
            if shutdown_event.is_set():
                logger.info("Shutdown event received, initiating server stop.")
            else:
                # Server task finished unexpectedly
                logger.warning("Uvicorn server task finished unexpectedly. Initiating shutdown.")
                # Ensure shutdown event is set so coordinator context manager exits cleanly
                shutdown_event.set()

            # Signal Uvicorn server to stop gracefully
            # Uvicorn 0.17+ recommends server.shutdown()
            # For older versions or if server.shutdown() isn't working as expected,
            # cancelling the task might be necessary.
            if hasattr(server, "shutdown"):
                 logger.info("Attempting graceful Uvicorn shutdown via server.shutdown()...")
                 # Add a timeout to prevent hanging indefinitely
                 shutdown_timeout = 10.0 # seconds
                 try:
                     await asyncio.wait_for(server.shutdown(), timeout=shutdown_timeout)
                     logger.info("Uvicorn server.shutdown() completed.")
                 except asyncio.TimeoutError:
                      logger.warning(f"Uvicorn server.shutdown() timed out after {shutdown_timeout}s. Cancelling task.")
                      if server_task and not server_task.done():
                          server_task.cancel()
                 except Exception as shutdown_err:
                      logger.error(f"Error during server.shutdown(): {shutdown_err}. Cancelling task.")
                      if server_task and not server_task.done():
                          server_task.cancel()
            else:
                 logger.info("server.shutdown() not available. Cancelling Uvicorn server task...")
                 if server_task and not server_task.done():
                     server_task.cancel()

            # Wait for the server task to fully complete after shutdown/cancellation
            if server_task:
                try:
                    await server_task
                except asyncio.CancelledError:
                    logger.info("Uvicorn server task successfully cancelled.")
                except Exception as e:
                    # Log errors during task completion, but avoid logging expected shutdown errors again
                    if not isinstance(e, (SystemExit, KeyboardInterrupt, OSError)):
                         logger.error(f"Error waiting for Uvicorn server task completion: {e}", exc_info=True)
                    else:
                         logger.debug(f"Uvicorn server task finished with expected exception type: {type(e).__name__}")

        # Coordinator context exit handles adapter shutdown and coordinator cleanup
        logger.info("ApplicationCoordinator context exited.")

    except asyncio.CancelledError:
        logger.info("serve_sse task cancelled externally.")
        # Ensure shutdown event is set if cancelled externally
        if not shutdown_event.is_set():
            shutdown_event.set()
        # Ensure server task is cancelled if main task is cancelled
        if server_task and not server_task.done():
            logger.info("Cancelling Uvicorn server task due to external cancellation...")
            server_task.cancel()
            try:
                await server_task # Wait briefly for cancellation
            except asyncio.CancelledError:
                pass # Expected
            except Exception as e:
                 logger.error(f"Error waiting for server task cancellation after external cancel: {e}")

    except Exception as e:
        logger.critical(f"Critical error in serve_sse execution: {e}", exc_info=True)
        # Ensure shutdown event is set on unexpected error
        if not shutdown_event.is_set():
            shutdown_event.set()
        # Ensure server task is cancelled
        if server_task and not server_task.done():
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass # Expected
            except Exception as e_inner:
                 logger.error(f"Error waiting for server task cancellation after critical error: {e_inner}")

    finally:
        logger.info("serve_sse: Final cleanup...")
        # Restore original signal handlers
        for sig, handler in original_handlers.items():
            try:
                # Check if loop handler was used and remove it
                # Note: loop.remove_signal_handler might need the *exact* wrapper function
                # which is harder to retrieve here. Relying on signal.signal might be safer
                # for restoration if loop handler was used.
                # For simplicity, just restore using signal.signal
                signal.signal(sig, handler)
                logger.debug(f"Restored original signal handler for {sig.name}")
            except (ValueError, OSError, NotImplementedError) as e:
                 logger.warning(f"Could not restore original signal handler for {sig.name}: {e}")

        _adapter = None # Clear global adapter reference
        logger.info("serve_sse: Cleanup complete.")


# --- Standalone Execution Removed ---
# The main entry point should be via aider_mcp_server.__main__.py,
# which calls serve_sse. Removing the standalone main() and main_async()
# functions from this file simplifies it and aligns with the intended usage.
# if __name__ == "__main__":
#     # main() - Removed
