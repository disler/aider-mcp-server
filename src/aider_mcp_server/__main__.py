import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path # Import Path
from typing import Optional

# Use absolute imports from the package root
from aider_mcp_server.atoms.atoms_utils import (
    DEFAULT_EDITOR_MODEL,
    DEFAULT_WS_HOST,
    DEFAULT_WS_PORT,
)
from aider_mcp_server.atoms.logging import Logger, get_logger
from aider_mcp_server.server import serve, is_git_repository # stdio mode and validation
from aider_mcp_server.sse_server import serve_sse # sse mode
from aider_mcp_server.multi_transport_server import serve_multi_transport # multi mode
from aider_mcp_server.transport_coordinator import ApplicationCoordinator


# Configure logging early
log_dir_path: Optional[Path] = None
try:
    log_dir = os.path.expanduser("~/.aider-mcp-server/logs")
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    logger: Logger = get_logger(__name__, log_dir=log_dir_path)
except OSError as e:
    print(f"Warning: Could not create log directory {log_dir}: {e}. Logging to console only.", file=sys.stderr)
    logger = get_logger(__name__) # Basic console logger


async def handle_sigterm(loop: asyncio.AbstractEventLoop, logger_instance: Logger) -> None:
    """Gracefully shuts down the coordinator when SIGTERM is received (for stdio/sse modes)."""
    logger_instance.info("SIGTERM received. Initiating graceful shutdown via Coordinator...")
    try:
        coordinator = ApplicationCoordinator.getInstance()
        await coordinator.shutdown()
        logger_instance.info("ApplicationCoordinator shutdown initiated via SIGTERM.")
    except Exception as e:
        logger_instance.error(f"Error during coordinator shutdown via SIGTERM: {e}", exc_info=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aider MCP Server - Offload AI coding tasks to Aider.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server-mode", type=str, choices=["stdio", "sse", "multi"], default="stdio",
        help="Server communication mode (default: stdio)."
    )
    parser.add_argument(
        "--editor-model", type=str, default=DEFAULT_EDITOR_MODEL,
        help=f"Primary AI model for Aider (default: {DEFAULT_EDITOR_MODEL})"
    )
    parser.add_argument(
        "--current-working-dir", type=str, required=True,
        help="Path to the current working directory (must be a valid git repository)"
    )
    web_group = parser.add_argument_group("SSE/Multi Server Options")
    web_group.add_argument(
        "--host", type=str, default=DEFAULT_WS_HOST,
        help=f"Host address for SSE/Multi server (default: {DEFAULT_WS_HOST})."
    )
    web_group.add_argument(
        "--port", type=int, default=DEFAULT_WS_PORT,
        help=f"Port number for SSE/Multi server (default: {DEFAULT_WS_PORT})."
    )
    args: argparse.Namespace = parser.parse_args()

    # Validate CWD early
    try:
        # Resolve to absolute path and check existence/type
        abs_cwd_path: Path = Path(args.current_working_dir).resolve(strict=True)
        if not abs_cwd_path.is_dir():
             # This case should be caught by strict=True, but check explicitly
             logger.critical(f"Error: Specified working directory is not a directory: {abs_cwd_path}")
             sys.exit(1)
    except FileNotFoundError:
         logger.critical(f"Error: Specified working directory does not exist: {args.current_working_dir}")
         sys.exit(1)
    except Exception as e:
         logger.critical(f"Error resolving working directory '{args.current_working_dir}': {e}")
         sys.exit(1)

    # Validate CWD is a git repository
    is_repo, git_error = is_git_repository(abs_cwd_path) # Pass Path object
    if not is_repo:
        logger.critical(f"Error: Specified working directory is not a valid git repository: {abs_cwd_path} ({git_error})")
        sys.exit(1)
    logger.info(f"Validated working directory (git repository): {abs_cwd_path}")
    # Convert Path to string for passing to server functions
    abs_cwd_str = str(abs_cwd_path)

    # Validate host/port arguments based on server mode
    if args.server_mode == "stdio":
        if args.host != DEFAULT_WS_HOST or args.port != DEFAULT_WS_PORT:
            logger.warning("Warning: --host and --port arguments are ignored in 'stdio' mode.")

    # Signal handling setup (conditional)
    if args.server_mode != "multi":
        loop = asyncio.get_event_loop()
        try:
            loop.add_signal_handler(
                signal.SIGTERM,
                lambda: asyncio.create_task(handle_sigterm(loop, logger)),
            )
            logger.info(f"SIGTERM handler registered for {args.server_mode} mode.")
        except NotImplementedError:
            logger.warning("Signal handlers (SIGTERM) not supported on this platform.")
        except Exception as e:
            logger.error(f"Error setting up SIGTERM handler: {e}")
    else:
        logger.info("Signal handling delegated to multi_transport_server for 'multi' mode.")

    try:
        if args.server_mode == "multi":
            logger.info(f"Starting in Multi-Transport server mode (SSE on http://{args.host}:{args.port}, plus Stdio)")
            asyncio.run(
                serve_multi_transport(
                    host=args.host, port=args.port, editor_model=args.editor_model,
                    current_working_dir=abs_cwd_str, # Pass validated string path
                )
            )
        elif args.server_mode == "sse":
            logger.info(f"Starting in SSE server mode (http://{args.host}:{args.port})")
            asyncio.run(
                serve_sse(
                    host=args.host, port=args.port, editor_model=args.editor_model,
                    current_working_dir=abs_cwd_str, # Pass validated string path
                )
            )
        elif args.server_mode == "stdio":
            logger.info("Starting in stdio server mode")
            asyncio.run(
                serve(
                    editor_model=args.editor_model,
                    current_working_dir=abs_cwd_str, # Pass validated string path
                )
            )
        else:
            logger.critical(f"Error: Unknown server mode '{args.server_mode}'")
            sys.exit(1)

    except ValueError as e: # Catch validation errors from server functions
        logger.critical(f"Server configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info(f"Server stopped by user (KeyboardInterrupt) in {args.server_mode} mode.")
    except asyncio.CancelledError:
         logger.info(f"Main server task cancelled, likely during shutdown in {args.server_mode} mode.")
    except Exception as e:
        logger.exception(f"Critical server error in {args.server_mode} mode: {e}")
        sys.exit(1)
    finally:
        logger.info(f"Aider MCP Server ({args.server_mode} mode) shutting down.")

if __name__ == "__main__":
    main()
