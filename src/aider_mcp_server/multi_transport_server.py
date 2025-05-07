import argparse
import asyncio
import functools
import signal
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import uvicorn

# Use absolute imports from the package root
from fastapi import FastAPI, Request, Response

from aider_mcp_server.atoms.atoms_utils import (
    DEFAULT_EDITOR_MODEL,
    DEFAULT_WS_HOST,
    DEFAULT_WS_PORT,
)
from aider_mcp_server.atoms.logging import get_logger
from aider_mcp_server.handlers import (
    process_aider_ai_code_request,
    process_list_models_request,
)
from aider_mcp_server.security import SecurityContext

# Import is_git_repository for validation if needed here, or rely on __main__ validation
from aider_mcp_server.server import is_git_repository
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.stdio_transport_adapter import StdioTransportAdapter
from aider_mcp_server.transport_coordinator import ApplicationCoordinator

# Define a generic type variable for Task
T = TypeVar("T")

# Default values
DEFAULT_LOG_DIR = Path("./.aider_mcp_logs")


# --- Helper Context Managers for Shutdown ---
# Protocol defining the full expected interface for a transport adapter
class Shutdownable(Protocol):
    transport_id: str
    async def shutdown(self) -> None: ...
    async def initialize(self) -> None: ...
    async def start_listening(self) -> None: ...

# Protocol defining only the members needed by the adapter_shutdown_context
class ShutdownContextProtocol(Protocol):
    transport_id: str
    async def shutdown(self) -> None: ...


@asynccontextmanager
async def adapter_shutdown_context(adapter: ShutdownContextProtocol) -> AsyncIterator[None]:
    """Ensures adapter.shutdown() is called on context exit."""
    adapter_name = f"{adapter.transport_id} adapter"
    try:
        yield
    finally:
        logger.info(f"Shutting down {adapter_name}...")
        try:
            await asyncio.wait_for(adapter.shutdown(), timeout=10.0)
            logger.info(f"{adapter_name} shut down successfully.")
        except asyncio.TimeoutError:
            logger.warning(f"{adapter_name} shutdown timed out.")
        except Exception as e:
            logger.error(f"Error shutting down {adapter_name}: {e}", exc_info=True)

@asynccontextmanager
async def coordinator_shutdown_context(coordinator: ApplicationCoordinator) -> AsyncIterator[None]:
    """Ensures coordinator.shutdown() is called on context exit."""
    try:
        yield
    finally:
        logger.info("Shutting down Application Coordinator...")
        try:
            await coordinator.shutdown()
            logger.info("Application Coordinator shut down successfully.")
        except Exception as e:
            logger.error(f"Error shutting down Application Coordinator: {e}", exc_info=True)

# Global logger instance needed for context managers and setup
# Initialize logger early for setup messages
logger = get_logger("multi_transport_server_setup")


def _setup_logger(log_dir: Optional[Union[str, Path]]) -> None:
    """Configure the logger with the specified log directory."""
    global logger

    if log_dir:
        log_dir_path = Path(log_dir).resolve()
        try:
            log_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create log directory {log_dir_path}: {e}. Logging may be incomplete.")
            log_dir_path = None
    else:
        log_dir_path = None

    logger = get_logger("multi_transport_server", log_dir=log_dir_path)
    logger.info(f"Logging to: {log_dir_path or 'console'}")


def _validate_working_directory(current_working_dir: str) -> Path:
    """Validate that the working directory is a git repository."""
    cwd_path = Path(current_working_dir)
    is_repo, git_error = is_git_repository(cwd_path)
    if not is_repo:
        logger.critical(f"Working directory '{cwd_path}' is not a valid git repository: {git_error}. Aborting.")
        raise ValueError(f"Working directory '{cwd_path}' is not a valid git repository: {git_error}")
    return cwd_path


def _setup_signal_handlers(shutdown_event: asyncio.Event) -> None:
    """Set up signal handlers for graceful shutdown."""
    def _signal_handler(sig: int, *_: Any) -> None:
        signame = signal.Signals(sig).name
        logger.warning(f"Received signal {signame} ({sig}). Initiating shutdown...")
        if not shutdown_event.is_set():
            shutdown_event.set()
        else:
            logger.warning("Shutdown already in progress.")

    loop = asyncio.get_running_loop()
    for sig_enum in (signal.SIGINT, signal.SIGTERM):
        sig_num = int(sig_enum)
        try:
            loop.add_signal_handler(sig_num, functools.partial(_signal_handler, sig_num))
        except (NotImplementedError, ValueError, OSError) as e:
            logger.warning(f"Could not register signal handler for {sig_enum.name} using loop.add_signal_handler: {e}. Trying signal.signal.")
            try:
                signal.signal(sig_num, lambda s, f: functools.partial(_signal_handler, s)())
            except (ValueError, OSError) as sig_e:
                logger.error(f"Failed to register signal handler for {sig_enum.name} using signal.signal: {sig_e}")

    logger.debug("Signal handlers registered.")


async def _register_handlers(coordinator: ApplicationCoordinator, editor_model: str, cwd_path: Path) -> None:
    """Register operation handlers with the coordinator."""
    # Aider AI Code handler
    async def aider_handler_wrapper(
        request_id: str,
        transport_id: str,
        params: Dict[str, Any],
        security_context: SecurityContext
    ) -> Dict[str, Any]:
        """Wrapper for aider_ai_code handler that includes context parameters"""
        return await process_aider_ai_code_request(
            request_id=request_id,
            transport_id=transport_id,
            params=params,
            security_context=security_context,
            editor_model=editor_model,
            current_working_dir=str(cwd_path),
        )

    coordinator.register_handler("aider_ai_code", aider_handler_wrapper, required_permission="execute_aider")
    logger.info("Registered handler for 'aider_ai_code' with permission 'execute_aider'.")

    # List Models handler
    async def list_models_wrapper(
        request_id: str,
        transport_id: str,
        params: Dict[str, Any],
        security_context: SecurityContext
    ) -> Dict[str, Any]:
        """Wrapper for list_models handler"""
        return await process_list_models_request(
            request_id=request_id,
            transport_id=transport_id,
            params=params,
            security_context=security_context,
        )

    coordinator.register_handler("list_models", list_models_wrapper)
    logger.info("Registered handler for 'list_models'.")


async def _setup_transports(stack: AsyncExitStack, coordinator: ApplicationCoordinator, heartbeat_interval: float) -> Tuple[SSETransportAdapter, StdioTransportAdapter]:
    """Set up and initialize transport adapters."""
    # SSE Adapter
    sse_adapter = SSETransportAdapter(coordinator=coordinator, heartbeat_interval=heartbeat_interval)
    await stack.enter_async_context(adapter_shutdown_context(sse_adapter))
    await sse_adapter.initialize()
    logger.info(f"SSETransportAdapter '{sse_adapter.transport_id}' initialized.")

    # Stdio Adapter
    stdio_adapter = StdioTransportAdapter(coordinator=coordinator, heartbeat_interval=None)
    await stack.enter_async_context(adapter_shutdown_context(stdio_adapter))
    await stdio_adapter.initialize()
    await stdio_adapter.start_listening()  # Stdio needs explicit start
    logger.info(f"StdioTransportAdapter '{stdio_adapter.transport_id}' initialized and listening.")

    # Log transport capabilities
    logger.info(f"SSE adapter '{sse_adapter.transport_id}' subscribed based on capabilities: {sse_adapter.get_capabilities()}")
    logger.info(f"Stdio adapter '{stdio_adapter.transport_id}' subscribed based on capabilities: {stdio_adapter.get_capabilities()}")

    return sse_adapter, stdio_adapter


def _create_fastapi_app(sse_adapter: SSETransportAdapter) -> FastAPI:
    """Create and configure the FastAPI application with SSE endpoints."""
    app = FastAPI(title="Aider MCP Server (Multi-Transport)", description="Handles SSE and Stdio connections.")

    @app.get("/sse", summary="Establish SSE Connection", tags=["SSE"])
    async def sse_endpoint(request: Request) -> Response:
        response: Response = await sse_adapter.handle_sse_request(request)
        return response

    @app.post("/message", summary="Submit Operation Request", tags=["SSE"])
    async def message_endpoint(request: Request) -> Response:
        response: Response = await sse_adapter.handle_message_request(request)
        return response

    logger.info("FastAPI app created and SSE routes added (/sse GET, /message POST).")
    return app


def _configure_uvicorn_server(app: FastAPI, host: str, port: int) -> uvicorn.Server:
    """Configure the Uvicorn server with the FastAPI app."""
    # Disable Uvicorn's default logging
    uvicorn_log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"fmt": "%(message)s"}},
        "handlers": {"default": {"class": "logging.NullHandler"}},
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }

    config = uvicorn.Config(
        app, host=host, port=port,
        log_config=uvicorn_log_config,
        lifespan="off",
    )
    server = uvicorn.Server(config)
    logger.info(f"Uvicorn server configured for http://{host}:{port}")
    return server


async def _monitor_server_tasks(server_task: asyncio.Task, shutdown_event: asyncio.Event) -> None:
    """Monitor server tasks and handle shutdown when needed."""
    monitor_task: asyncio.Task[bool] = asyncio.create_task(shutdown_event.wait(), name="shutdown_monitor")
    tasks_to_wait = {server_task, monitor_task}
    done: Set[asyncio.Task[Any]] = set()
    pending: Set[asyncio.Task[Any]] = tasks_to_wait

    try:
        while not shutdown_event.is_set() and server_task in pending:
            done_part, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            done.update(done_part)
            if server_task in done_part and not server_task.cancelled():
                exc = server_task.exception()
                if exc:
                    logger.error(f"Task {server_task.get_name()} exited with exception:", exc_info=exc)
                else:
                    logger.warning(f"Task {server_task.get_name()} exited unexpectedly without error.")
                if not shutdown_event.is_set():
                    logger.warning(f"Triggering shutdown due to unexpected exit of task {server_task.get_name()}.")
                    shutdown_event.set()
    except asyncio.CancelledError:
        logger.info("Main server loop cancelled, initiating shutdown.")
        if not shutdown_event.is_set():
            shutdown_event.set()

    # Cancel the monitor task if it's still pending
    if monitor_task in pending:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


async def _signal_server_shutdown(server: uvicorn.Server, server_task: asyncio.Task) -> None:
    """Signal the Uvicorn server to shut down."""
    if server.should_exit:
        return

    logger.info("Signaling Uvicorn server task to exit.")
    if hasattr(server, "shutdown"):
        uv_shutdown_task = asyncio.create_task(server.shutdown())
        try:
            await asyncio.wait_for(uv_shutdown_task, timeout=10.0)
            logger.info("Uvicorn server shutdown initiated.")
        except asyncio.TimeoutError:
            logger.warning("Uvicorn server shutdown() call timed out.")
            if not server_task.done():
                server_task.cancel()
        except Exception as e:
            logger.error(f"Error calling Uvicorn server shutdown(): {e}")
            if not server_task.done():
                server_task.cancel()
    else:
        server.should_exit = True
        logger.info("Uvicorn server signaled via should_exit=True.")


async def _wait_for_server_task(server_task: asyncio.Task, done: Set[asyncio.Task[Any]]) -> None:
    """Wait for the Uvicorn server task to finish."""
    if server_task in done:
        return

    logger.info("Waiting for Uvicorn server task to complete shutdown...")
    try:
        await asyncio.wait_for(server_task, timeout=5.0)
        logger.info("Uvicorn server task finished.")
    except asyncio.TimeoutError:
        logger.warning("Uvicorn server task did not finish within timeout. Attempting cancellation.")
        server_task.cancel()
    except asyncio.CancelledError:
        logger.info("Uvicorn server task was cancelled.")
    except Exception as e:
        logger.error(f"Error waiting for Uvicorn server task: {e}", exc_info=True)


async def _shutdown_uvicorn_server(server: uvicorn.Server, server_task: asyncio.Task, done: Set[asyncio.Task[Any]]) -> None:
    """Gracefully shut down the Uvicorn server."""
    await _signal_server_shutdown(server, server_task)
    await _wait_for_server_task(server_task, done)


async def serve_multi_transport(
    host: str,
    port: int,
    editor_model: str,
    current_working_dir: str,
    log_dir: Optional[Union[str, Path]] = DEFAULT_LOG_DIR,
    heartbeat_interval: float = 15.0,
) -> None:
    """
    Sets up and runs the multi-transport server (SSE and Stdio).
    """
    # Setup and initialization
    _setup_logger(log_dir)
    cwd_path = _validate_working_directory(current_working_dir)
    logger.info(f"Starting multi-transport server with config: host={host}, port={port}, editor_model={editor_model}, cwd={cwd_path}, heartbeat={heartbeat_interval}s")

    # Create shutdown event and set up signal handlers
    shutdown_event = asyncio.Event()
    _setup_signal_handlers(shutdown_event)

    # Main server setup and execution
    async with AsyncExitStack() as stack:
        # Initialize coordinator
        coordinator = ApplicationCoordinator.getInstance()
        logger.info("ApplicationCoordinator instance obtained.")
        await stack.enter_async_context(coordinator_shutdown_context(coordinator))

        # Register operation handlers
        await _register_handlers(coordinator, editor_model, cwd_path)

        # Set up transport adapters
        sse_adapter, stdio_adapter = await _setup_transports(stack, coordinator, heartbeat_interval)

        # Create and configure FastAPI app
        app = _create_fastapi_app(sse_adapter)

        # Configure and start Uvicorn server
        server = _configure_uvicorn_server(app, host, port)
        server_task = asyncio.create_task(server.serve(), name="uvicorn_server")
        logger.info("Starting Uvicorn server task. Stdio adapter is listening in background.")

        # Monitor server tasks and handle shutdown
        await _monitor_server_tasks(server_task, shutdown_event)

        # Shutdown sequence
        logger.info("Shutdown signal detected or server task finished. Proceeding with cleanup.")
        await _shutdown_uvicorn_server(server, server_task, set())

    # Exit stack context manager handles adapter and coordinator shutdown
    logger.info("Server shutdown sequence complete.")


# --- Main Execution (for direct script run, usually called via __main__) ---
def main() -> None:
    """Entry point for the multi-transport server script."""
    parser = argparse.ArgumentParser(description="Aider MCP Multi-Transport Server")
    parser.add_argument("--host", type=str, default=DEFAULT_WS_HOST, help=f"Host for SSE (default: {DEFAULT_WS_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_WS_PORT, help=f"Port for SSE (default: {DEFAULT_WS_PORT})")
    parser.add_argument("--editor-model", type=str, default=DEFAULT_EDITOR_MODEL, help=f"Primary AI model (default: {DEFAULT_EDITOR_MODEL})")
    parser.add_argument("--cwd", type=str, default=".", help="Working directory (git repo) (default: .)")
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR), help=f"Log directory (default: {DEFAULT_LOG_DIR})")
    parser.add_argument("--heartbeat", type=float, default=15.0, help="SSE heartbeat interval (sec) (default: 15.0)")
    args = parser.parse_args()

    try:
        cwd_path = Path(args.cwd).resolve(strict=True)
    except FileNotFoundError:
         print(f"Error: Working directory '{args.cwd}' not found.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         print(f"Error resolving working directory '{args.cwd}': {e}", file=sys.stderr)
         sys.exit(1)

    log_dir_path = Path(args.log_dir) if args.log_dir else None

    try:
        asyncio.run(
            serve_multi_transport(
                host=args.host,
                port=args.port,
                editor_model=args.editor_model,
                current_working_dir=str(cwd_path),
                log_dir=log_dir_path,
                heartbeat_interval=args.heartbeat,
            )
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nServer interrupted. Exiting.")
    except ValueError as e:
         print(f"\nConfiguration Error: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"\nFatal error during server execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up signal handlers (best effort)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, signal.SIG_DFL)
            except (ValueError, OSError, AttributeError):
                pass

if __name__ == "__main__":
    main()
