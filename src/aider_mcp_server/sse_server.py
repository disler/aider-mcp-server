import asyncio
import json
import signal
import sys
from functools import partial
from types import FrameType
from typing import Any, Callable, Coroutine, Optional, Dict, List
from pathlib import Path

# Starlette imports
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.routing import Route

# MCP Server imports
from aider_mcp_server.transport_coordinator import ApplicationCoordinator
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.security import Permissions, SecurityContext
from aider_mcp_server.server import is_git_repository

# Import the actual tool functions
try:
    from aider_mcp_server.atoms.tools.aider_ai_code import code_with_aider
    aider_ai_code_available = True
except ImportError:
    def code_with_aider(*args: Any, **kwargs: Any) -> str:
        print("Warning: code_with_aider function not found. Using placeholder.", file=sys.stderr)
        return json.dumps({"success": False, "error": "code_with_aider function not implemented"})
    aider_ai_code_available = False
    print("Warning: Could not import code_with_aider function. Handler will use placeholder.", file=sys.stderr)

try:
    from aider_mcp_server.atoms.tools.aider_list_models import list_models
    list_models_available = True
except ImportError:
     def list_models(*args: Any, **kwargs: Any) -> List[str]:
        print("Warning: list_models function not found. Using placeholder.", file=sys.stderr)
        return []
     list_models_available = False
     print("Warning: Could not import list_models function. Handler will use placeholder.", file=sys.stderr)


try:
    from .atoms.logging import get_logger_func, LoggerProtocol
except ImportError:
    import logging
    from typing import Protocol
    class LoggerProtocol(Protocol):
        def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
        def debug(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
        def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
        def error(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
        def exception(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
        def critical(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def get_logger_func(name: str) -> logging.Logger:
        _logger = logging.getLogger(name)
        if not _logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
        return _logger # type: ignore

logger: LoggerProtocol = get_logger_func(__name__)

_test_handle_shutdown_signal: Optional[Callable[..., Coroutine[Any, Any, None]]] = None

async def handle_shutdown_signal(
    sig: int,
    event: asyncio.Event,
    signum: Optional[int] = None,
    frame: Optional[FrameType] = None
) -> None:
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
    sig: int,
    signum: Optional[int] = None,
    frame: Optional[FrameType] = None
) -> None:
    signal_name_for_handler = signal.Signals(sig).name
    if signum is not None:
        received_signal_name = signal.Signals(signum).name
    else:
        received_signal_name = signal_name_for_handler
    logger.debug(f"Test shutdown handler called for signal {received_signal_name} (handler for {signal_name_for_handler}).")

def _create_shutdown_task_wrapper(
    sig: int,
    async_handler: Callable[..., Coroutine[Any, Any, None]],
    event: Optional[asyncio.Event] = None
) -> Callable[[int, Optional[FrameType]], None]:
    def sync_wrapper(signum: int, frame: Optional[FrameType]) -> None:
        logger.debug(f"Sync wrapper called for signal {sig} (received {signum}). Scheduling async handler.")

        if event is not None:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running() and not loop.is_closed():
                    logger.debug(f"Scheduling async handler for signal {sig} with event.")
                    loop.create_task(async_handler(sig, event, signum, frame))
                else:
                    logger.warning(f"Event loop not running or closed when handling signal {sig}. Cannot schedule async handler.")
            except RuntimeError as e:
                logger.error(f"Error getting running loop for signal {sig}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error scheduling async handler for signal {sig} with event: {e}", exc_info=True)
        else:
            logger.debug(f"Scheduling async handler for signal {sig} without event (test mode?).")
            try:
                asyncio.create_task(async_handler(sig, signum, frame))
            except RuntimeError as e:
                logger.error(f"Error calling asyncio.create_task directly for signal {sig} in test mode: {e}")
            except Exception as e:
                 logger.error(f"Unexpected error scheduling async handler for signal {sig} without event: {e}", exc_info=True)

    return sync_wrapper

async def serve_sse(
    host: str,
    port: int,
    editor_model: str,
    current_working_dir: str,
    heartbeat_interval: float = 15.0
) -> None:
    logger.info(f"Validating working directory: {current_working_dir}")
    is_repo, error_msg = is_git_repository(Path(current_working_dir))
    if not is_repo:
        error_message = f"Error: The specified directory '{current_working_dir}' is not a valid git repository: {error_msg}"
        logger.critical(error_message)
        raise ValueError(error_message)
    logger.info(f"Working directory '{current_working_dir}' is a valid git repository.")

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    actual_shutdown_handler = _test_handle_shutdown_signal or handle_shutdown_signal
    sync_signal_handlers: Dict[int, Callable[[int, Optional[FrameType]], None]] = {}
    original_signal_handlers: Dict[int, Any] = {}

    for sig_num in (signal.SIGINT, signal.SIGTERM):
        sync_wrapper = _create_shutdown_task_wrapper(sig_num, actual_shutdown_handler, shutdown_event)
        sync_signal_handlers[sig_num] = sync_wrapper
        try:
            loop.add_signal_handler(sig_num, sync_wrapper)
            logger.info(f"Registered signal handler for {signal.Signals(sig_num).name} using loop.add_signal_handler.")
        except NotImplementedError:
            logger.warning(f"loop.add_signal_handler not supported for {signal.Signals(sig_num).name}. Falling back to signal.signal().")
            try:
                 original = signal.signal(sig_num, sync_wrapper) # type: ignore[arg-type]
                 original_signal_handlers[sig_num] = original
                 logger.info(f"Registered signal handler for {signal.Signals(sig_num).name} using signal.signal().")
            except (ValueError, OSError, TypeError) as e:
                 logger.error(f"Failed to set signal handler using signal.signal for {signal.Signals(sig_num).name}: {e}")
        except Exception as e:
             logger.error(f"Unexpected error setting signal handler for {signal.Signals(sig_num).name}: {e}", exc_info=True)

    coordinator = None
    sse_adapter = None
    server: Optional[uvicorn.Server] = None

    try:
        coordinator = await ApplicationCoordinator.getInstance()

        async with coordinator:
            logger.info("Coordinator context entered.")
            sse_adapter = SSETransportAdapter(
                coordinator=coordinator,
                heartbeat_interval=heartbeat_interval
            )

            async def aider_ai_code_handler(
                request_id: str,
                transport_id: str,
                parameters: Dict[str, Any],
                security_context: SecurityContext
            ) -> Dict[str, Any]:
                logger.info(f"Handler 'aider_ai_code_handler' invoked for request {request_id}")
                try:
                    ai_coding_prompt = parameters.get("ai_coding_prompt")
                    relative_editable_files = parameters.get("relative_editable_files", [])
                    relative_readonly_files = parameters.get("relative_readonly_files", [])
                    model_to_use = parameters.get("model", editor_model)
                    use_diff_cache = parameters.get("use_diff_cache", True)
                    clear_cached_for_unchanged = parameters.get("clear_cached_for_unchanged", True)

                    if not ai_coding_prompt:
                        logger.error(f"Missing 'ai_coding_prompt' parameter for request {request_id}")
                        return {"success": False, "error": "Missing 'ai_coding_prompt' parameter"}
                    if not relative_editable_files:
                         logger.error(f"Missing or empty 'relative_editable_files' parameter for request {request_id}")
                         return {"success": False, "error": "Missing or empty 'relative_editable_files' parameter"}

                    logger.debug(f"Running code_with_aider for request {request_id}")
                    result_dict = await code_with_aider(
                        ai_coding_prompt, relative_editable_files,
                        relative_readonly_files, model_to_use, current_working_dir,
                        use_diff_cache, clear_cached_for_unchanged
                    )
                    logger.debug(f"code_with_aider execution finished for request {request_id}")
                    logger.info(f"Handler 'aider_ai_code_handler' completed successfully for request {request_id}")
                    return result_dict

                except Exception as e:
                    logger.exception(f"Error in 'aider_ai_code_handler' for request {request_id}: {e}")
                    return {"success": False, "error": f"Internal handler error: {str(e)}"}

            async def list_models_handler(
                request_id: str,
                transport_id: str,
                parameters: Dict[str, Any],
                security_context: SecurityContext
            ) -> Dict[str, Any]:
                logger.info(f"Handler 'list_models_handler' invoked for request {request_id}")
                try:
                    substring = parameters.get("substring", "")
                    logger.debug(f"Running list_models for request {request_id}")
                    models_list = await list_models(substring)
                    logger.debug(f"list_models execution finished for request {request_id}")
                    logger.info(f"Handler 'list_models_handler' completed successfully for request {request_id}")
                    return {"success": True, "models": models_list}
                except Exception as e:
                    logger.exception(f"Error in 'list_models_handler' for request {request_id}: {e}")
                    return {"success": False, "error": f"Internal handler error: {str(e)}"}

            async def sse_endpoint(request: Request) -> Response:
                if not sse_adapter:
                    logger.error("SSE adapter not initialized when handling /sse request.")
                    return JSONResponse({"success": False, "error": "Server setup error"}, status_code=500)
                return await sse_adapter.handle_sse_request(request)

            async def message_endpoint(request: Request) -> Response:
                 if not sse_adapter:
                    logger.error("SSE adapter not initialized when handling /message request.")
                    return JSONResponse({"success": False, "error": "Server setup error"}, status_code=500)
                return await sse_adapter.handle_message_request(request)

            logger.info("Registering SSE transport adapter with coordinator...")
            await sse_adapter.initialize()

            logger.info("Registering operation handlers with coordinator...")
            if aider_ai_code_available:
                await coordinator.register_handler("aider_ai_code", aider_ai_code_handler, required_permission=Permissions.EXECUTE_AIDER)
                logger.info("Registered 'aider_ai_code' handler.")
            else:
                logger.warning("Skipping registration of 'aider_ai_code' handler due to import failure.")

            if list_models_available:
                await coordinator.register_handler("list_models", list_models_handler)
                logger.info("Registered 'list_models' handler.")
            else:
                 logger.warning("Skipping registration of 'list_models' handler due to import failure.")

            routes = [
                Route("/sse", endpoint=sse_endpoint),
                Route("/message", endpoint=message_endpoint, methods=["POST"]),
            ]
            app = Starlette(routes=routes)
            config = uvicorn.Config(app=app, host=host, port=port, log_config=None, handle_signals=False)
            server = uvicorn.Server(config)

            logger.info(f"Starting Uvicorn server on {host}:{port}...")
            await server.serve()
            logger.info("Uvicorn server has stopped.")

            if server:
                server.should_exit = True
                logger.info("Set server.should_exit = True.")
            else:
                logger.warning("Server instance not available to signal exit.")

        logger.info("Exiting coordinator context (will trigger coordinator cleanup)...")

    except Exception as e:
        logger.exception(f"An unexpected error occurred during server setup or runtime: {e}")

    finally:
        logger.info("Starting final cleanup (removing signal handlers)...")
        for sig_num, sync_wrapper_func in sync_signal_handlers.items():
            signal_name = signal.Signals(sig_num).name
            try:
                if loop.is_running() and not loop.is_closed():
                    loop.remove_signal_handler(sig_num)
                    logger.debug(f"Removed signal handler for {signal_name} using loop.remove_signal_handler.")
                logger.debug(f"Restoring default handler for {signal_name} using signal.signal().")
                try:
                    signal.signal(sig_num, signal.SIG_DFL) # type: ignore[arg-type]
                    logger.debug(f"Restored default handler for {signal_name} using signal.signal.")
                except (ValueError, OSError, TypeError) as e:
                    logger.error(f"Failed to restore default signal handler for {signal_name} using signal.signal: {e}")
            except Exception as e:
                 logger.error(f"Unexpected error removing/restoring signal handler for {signal_name}: {e}", exc_info=True)

        logger.info("SSE Server shutdown process complete.")
