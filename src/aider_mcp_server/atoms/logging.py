import logging
import os
from pathlib import Path
from typing import Any, Optional, Union


class Logger:
    """Custom logger that writes to both console and file."""

    def __init__(
        self,
        name: str,
        log_dir: Optional[Union[str, Path]] = None,
        level: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            log_dir: Directory to store log files (defaults to ./logs)
            level: Logging level (defaults to INFO, or DEBUG if MCP_LOG_LEVEL=DEBUG/VERBOSE)
            verbose: Enable verbose logging mode (applies if level is DEBUG)
        """
        self.name = name
        # Prioritize the verbose flag passed to constructor
        self._verbose = verbose

        # Check environment variable for log level
        env_log_level = os.environ.get("MCP_LOG_LEVEL", "").upper()
        if level is None:  # Only apply env var if level is not explicitly passed
            if env_log_level == "VERBOSE":
                level = logging.DEBUG
                if not verbose: # if verbose is not already set by the arg
                    self._verbose = True  # VERBOSE implies debug level and verbose mode
            elif env_log_level == "DEBUG":
                level = logging.DEBUG
                # self._verbose is determined by the 'verbose' argument
            elif env_log_level == "WARNING":
                level = logging.WARNING
            elif env_log_level == "ERROR":
                level = logging.ERROR
            else:
                level = logging.INFO
        elif level == logging.DEBUG:
            if env_log_level == "VERBOSE" and not verbose:
                # If level is explicitly DEBUG, env var is VERBOSE, and verbose arg is False, enable verbose mode
                self._verbose = True
            # If verbose arg is True, self._verbose is already True

        self.level = level

        # Set up the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Define a standard formatter
        if level == logging.DEBUG:
            if self._verbose:
                log_formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s (%(pathname)s:%(lineno)d): %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            else:
                log_formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
        else:
            # Compact formatter for non-debug levels
            log_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )

        # Add console handler with standard formatting
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(level)
        self.logger.addHandler(console_handler)

        # Add file handler if log_dir is provided
        if log_dir is not None:
            # Create log directory if it doesn't exist
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Use a fixed log file name
            log_file_name = "aider_mcp_server.log"
            log_file_path = log_dir / log_file_name

            # Set up file handler to append
            file_handler = logging.FileHandler(log_file_path, mode="a")
            # Use the same formatter as the console handler
            file_handler.setFormatter(log_formatter)
            file_handler.setLevel(level)
            self.logger.addHandler(file_handler)

            self.log_file_path = log_file_path
            # Commented out to reduce verbosity
            # self.logger.info(f"Logging to: {log_file_path}")

    def is_verbose(self) -> bool:
        """Check if verbose mode is enabled."""
        return self._verbose

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        self.logger.debug(message, **kwargs)

    def verbose(self, message: str, **kwargs: Any) -> None:
        """Log a message at DEBUG level only if verbose mode is enabled."""
        if self._verbose:
            self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log a critical message."""
        self.logger.critical(message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log an exception message with traceback."""
        self.logger.exception(message, **kwargs)


def get_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    level: Optional[int] = None,
    verbose: bool = False,
) -> Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        log_dir: Directory to store log files (defaults to ./logs)
        level: Logging level (optional, will check MCP_LOG_LEVEL env var)
        verbose: Enable verbose logging mode (applies if level is DEBUG)

    Returns:
        Configured Logger instance
    """
    if log_dir is None:
        # Default log directory is ./logs
        log_dir = Path("./logs")

    return Logger(
        name=name,
        log_dir=log_dir,
        level=level,
        verbose=verbose,
    )
