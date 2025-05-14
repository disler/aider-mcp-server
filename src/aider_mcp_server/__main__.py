import argparse
import asyncio
import os
import sys
from pathlib import Path  # Import Path
from typing import Callable, Optional, Tuple

# Use absolute imports from the package root
from aider_mcp_server.atoms.atoms_utils import (
    DEFAULT_EDITOR_MODEL,
    DEFAULT_WS_HOST,
    DEFAULT_WS_PORT,
)
from aider_mcp_server.atoms.logging import Logger, get_logger
from aider_mcp_server.mcp_types import LoggerProtocol
from aider_mcp_server.multi_transport_server import serve_multi_transport  # multi mode
from aider_mcp_server.server import (  # stdio mode and validation
    is_git_repository,
    serve,
)
from aider_mcp_server.sse_server import serve_sse  # sse mode
from aider_mcp_server.transport_coordinator import ApplicationCoordinator

# Configure logging early
log_dir_path: Optional[Path] = None
try:
    log_dir = os.path.expanduser("~/.aider-mcp-server/logs")
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    logger: Logger = get_logger(__name__, log_dir=log_dir_path)
except OSError as e:
    print(
        f"Warning: Could not create log directory {log_dir}: {e}. Logging to console only.",
        file=sys.stderr,
    )
    logger = get_logger(__name__)  # Basic console logger


async def handle_sigterm(
    loop: asyncio.AbstractEventLoop, logger_instance: Logger
) -> None:
    """Gracefully shuts down the coordinator when SIGTERM is received (for stdio/sse modes)."""
    logger_instance.info(
        "SIGTERM received. Initiating graceful shutdown via Coordinator..."
    )
    try:
        coordinator = await ApplicationCoordinator.getInstance(get_logger)
        await coordinator.shutdown()
        logger_instance.info("ApplicationCoordinator shutdown initiated via SIGTERM.")
    except Exception as e:
        logger_instance.error(
            f"Error during coordinator shutdown via SIGTERM: {e}", exc_info=True
        )


def _setup_argument_parser() -> argparse.ArgumentParser:
    """Set up and return the argument parser for command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aider MCP Server - Offload AI coding tasks to Aider.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server-mode",
        type=str,
        choices=["stdio", "sse", "multi"],
        default="stdio",
        help="Server communication mode (default: stdio).",
    )
    parser.add_argument(
        "--editor-model",
        type=str,
        default=DEFAULT_EDITOR_MODEL,
        help=f"Primary AI model for Aider (default: {DEFAULT_EDITOR_MODEL})",
    )
    parser.add_argument(
        "--current-working-dir",
        type=str,
        required=True,
        help="Path to the current working directory (must be a valid git repository)",
    )
    web_group = parser.add_argument_group("SSE/Multi Server Options")
    web_group.add_argument(
        "--host",
        type=str,
        default=DEFAULT_WS_HOST,
        help=f"Host address for SSE/Multi server (default: {DEFAULT_WS_HOST}).",
    )
    web_group.add_argument(
        "--port",
        type=int,
        default=DEFAULT_WS_PORT,
        help=f"Port number for SSE/Multi server (default: {DEFAULT_WS_PORT}).",
    )
    return parser


def _validate_working_directory(log: LoggerProtocol, cwd_path: str) -> Tuple[Path, str]:
    """Validate that the working directory exists and is a git repository.

    Returns:
        Tuple containing the absolute Path object and string representation.
    """
    # Validate directory exists and is a directory
    try:
        abs_cwd_path: Path = Path(cwd_path).resolve(strict=True)
        if not abs_cwd_path.is_dir():
            log.critical(
                f"Error: Specified working directory is not a directory: {abs_cwd_path}"
            )
            sys.exit(1)
    except FileNotFoundError:
        log.critical(f"Error: Specified working directory does not exist: {cwd_path}")
        sys.exit(1)
    except Exception as e:
        log.critical(f"Error resolving working directory '{cwd_path}': {e}")
        sys.exit(1)

    # Validate it's a git repository
    is_repo, git_error = is_git_repository(abs_cwd_path)
    if not is_repo:
        log.critical(
            f"Error: Specified working directory is not a valid git repository: {abs_cwd_path} ({git_error})"
        )
        sys.exit(1)

    log.info(f"Validated working directory (git repository): {abs_cwd_path}")
    return abs_cwd_path, str(abs_cwd_path)


def _validate_server_mode_args(
    log: LoggerProtocol, server_mode: str, host: str, port: int
) -> None:
    """Validate arguments based on server mode."""
    if server_mode == "stdio":
        if host != DEFAULT_WS_HOST or port != DEFAULT_WS_PORT:
            log.warning(
                "Warning: --host and --port arguments are ignored in 'stdio' mode."
            )


def _setup_signal_handling(log: LoggerProtocol, server_mode: str) -> None:
    """Set up signal handling based on server mode."""
    if server_mode != "multi":
        try:
            # Signal handlers will be set up in the server functions
            log.info(
                f"SIGTERM handler registration will be done in server function for {server_mode} mode."
            )
        except NotImplementedError:
            log.warning("Signal handlers (SIGTERM) not supported on this platform.")
        except Exception as e:
            log.error(f"Error preparing SIGTERM handler: {e}")
    else:
        log.info(
            "Signal handling delegated to multi_transport_server for 'multi' mode."
        )


async def _run_server(
    server_mode: str, host: str, port: int, editor_model: str, cwd_str: str
) -> None:
    """Run the appropriate server based on the server mode."""
    if server_mode == "multi":
        await serve_multi_transport(
            host=host,
            port=port,
            editor_model=editor_model,
            current_working_dir=cwd_str,
        )
    elif server_mode == "sse":
        await serve_sse(
            host=host,
            port=port,
            editor_model=editor_model,
            current_working_dir=cwd_str,
        )
    elif server_mode == "stdio":
        await serve(
            editor_model=editor_model,
            current_working_dir=cwd_str,
        )
    else:
        raise ValueError(f"Unknown server mode: {server_mode}")


def main(logger_factory: Optional[Callable[..., LoggerProtocol]] = None) -> None:
    """Main entry point for the Aider MCP Server."""
    # Set up logging
    get_logger_func = logger_factory or get_logger
    log = get_logger_func(__name__, log_dir=log_dir_path)

    # Parse command-line arguments
    parser = _setup_argument_parser()
    args: argparse.Namespace = parser.parse_args()

    # Validate working directory
    abs_cwd_path, abs_cwd_str = _validate_working_directory(
        log, args.current_working_dir
    )

    # Validate server mode arguments
    _validate_server_mode_args(log, args.server_mode, args.host, args.port)

    # Set up signal handling
    _setup_signal_handling(log, args.server_mode)

    # Run the appropriate server
    try:
        if args.server_mode == "multi":
            log.info(
                f"Starting in Multi-Transport server mode (SSE on http://{args.host}:{args.port}, plus Stdio)"
            )
            asyncio.run(
                serve_multi_transport(
                    host=args.host,
                    port=args.port,
                    editor_model=args.editor_model,
                    current_working_dir=abs_cwd_str,
                )
            )
        elif args.server_mode == "sse":
            log.info(f"Starting in SSE server mode (http://{args.host}:{args.port})")
            asyncio.run(
                serve_sse(
                    host=args.host,
                    port=args.port,
                    editor_model=args.editor_model,
                    current_working_dir=abs_cwd_str,
                )
            )
        elif args.server_mode == "stdio":
            log.info("Starting in stdio server mode")
            asyncio.run(
                serve(
                    editor_model=args.editor_model,
                    current_working_dir=abs_cwd_str,
                )
            )
        else:
            log.critical(f"Error: Unknown server mode '{args.server_mode}'")
            sys.exit(1)
    except ValueError as e:  # Catch validation errors from server functions
        log.critical(f"Server configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        log.info(
            f"Server stopped by user (KeyboardInterrupt) in {args.server_mode} mode."
        )
    except asyncio.CancelledError:
        log.info(
            f"Main server task cancelled, likely during shutdown in {args.server_mode} mode."
        )
    except Exception as e:
        log.exception(f"Critical server error in {args.server_mode} mode: {e}")
        sys.exit(1)
    finally:
        log.info(f"Aider MCP Server ({args.server_mode} mode) shutting down.")


if __name__ == "__main__":
    main()
# Search for any other calls to logging.getLogger or logging.basicConfig
# If found, we may need to patch those as well to ensure consistent mocking
