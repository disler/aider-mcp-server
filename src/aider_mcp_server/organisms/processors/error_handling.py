import asyncio
import traceback
from typing import Any, Dict, Optional

# Assuming 'atoms' is a sub-package of 'aider_mcp_server' based on other project files.
# If 'atoms' is a top-level package, this import might need adjustment
# (e.g., from atoms.logging import get_logger, Logger as ProjectLogger)
from aider_mcp_server.atoms.logging.logger import Logger as ProjectLogger
from aider_mcp_server.atoms.logging.logger import get_logger


# Custom Exception Hierarchy
class AiderMCPError(Exception):
    """Base exception for all Aider MCP Server errors."""

    pass


class TransportError(AiderMCPError):
    """Error related to transport adapters."""

    pass


class HandlerError(AiderMCPError):
    """Error related to request handlers."""

    pass


class EventError(AiderMCPError):
    """Error related to event system."""

    pass


class InitializationError(AiderMCPError):
    """Error during component or application initialization."""

    pass


class ConfigurationError(AiderMCPError):
    """Error related to configuration loading or validation."""

    pass


# Error Handler Utilities
class ErrorHandler:
    """Provides utility methods for handling exceptions consistently."""

    # Initialize a default logger for the class, using the project's logging system.
    # The name of the logger will typically be 'aider_mcp_server.error_handling'.
    _logger: ProjectLogger = get_logger(__name__)

    @classmethod
    def format_exception(cls, exception: Exception) -> Dict[str, Any]:
        """
        Format an exception as a structured error response.
        This format is based on the Task 11 specification.

        Args:
            exception: The exception instance to format.

        Returns:
            A dictionary representing the structured error.
        """
        error_type_name = type(exception).__name__
        error_message = str(exception)

        # Capture traceback. Note: Sending full tracebacks to clients can be a security risk.
        # This implementation follows the Task 11 specification.
        error_traceback_str = traceback.format_exc()

        return {
            "type": "error",  # Main type indicating an error response
            "error": {
                "type": error_type_name,  # Specific type of the error (exception class name)
                "message": error_message,  # Human-readable error message
                "traceback": error_traceback_str,  # Full traceback string
            },
        }

    @classmethod
    def log_exception(
        cls,
        exception: Exception,
        context: Optional[str] = None,
        logger_instance: Optional[ProjectLogger] = None,
    ) -> None:
        """
        Log an exception with optional context using the project's logging system.

        Args:
            exception: The exception instance to log.
            context: Optional string providing context for where the error occurred.
            logger_instance: Optional specific logger instance to use.
                             Defaults to ErrorHandler's own logger if None.
        """
        logger_to_use = logger_instance if logger_instance is not None else cls._logger

        log_message = f"Error: {str(exception)}"
        if context:
            log_message = f"Error in {context}: {str(exception)}"

        # Use exc_info=True to ensure traceback is included in the log entry
        # The actual formatting of the traceback depends on the logger's configuration.
        logger_to_use.error(log_message, exc_info=True)

    @classmethod
    def handle_exception(
        cls,
        exception: Exception,
        context: Optional[str] = None,
        logger_instance: Optional[ProjectLogger] = None,
    ) -> Dict[str, Any]:
        """
        Handle an exception by logging it and then formatting it as a structured response.

        Args:
            exception: The exception instance to handle.
            context: Optional string providing context for the error.
            logger_instance: Optional specific logger instance to use for logging.

        Returns:
            A dictionary representing the structured error, suitable for an error response.
        """
        cls.log_exception(exception, context, logger_instance)
        return cls.format_exception(exception)

    @classmethod
    def install_global_exception_handler(
        cls,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        logger_instance: Optional[ProjectLogger] = None,
    ) -> None:
        """
        Install a global exception handler for unhandled exceptions in an asyncio event loop.

        Args:
            loop: Optional asyncio event loop. If None, asyncio.get_event_loop() is used.
            logger_instance: Optional specific logger instance for the global handler.
                             Defaults to ErrorHandler's own logger if None.
        """
        effective_loop = loop if loop is not None else asyncio.get_event_loop()
        logger_to_use = logger_instance if logger_instance is not None else cls._logger

        def global_async_exception_handler(async_loop: asyncio.AbstractEventLoop, context_dict: Dict[str, Any]) -> None:
            """Handles uncaught exceptions in the asyncio event loop."""
            exception = context_dict.get("exception")
            message = context_dict.get("message", "Unhandled error in asyncio event loop")

            log_context_message = "Unhandled exception in asyncio event loop"
            future = context_dict.get("future")
            if future:
                log_context_message += f" (Future: {future})"

            if exception:
                # Log the exception with its type, message, and traceback
                logger_to_use.error(
                    f"{log_context_message}: {type(exception).__name__}: {str(exception)}",
                    exc_info=exception,  # Pass exception object for full traceback logging
                )
            else:
                # Log the message if no exception object is present
                logger_to_use.error(f"{log_context_message}: {message}")

            # Note: The default asyncio exception handler prints to stderr and stops the loop
            # if the exception is not handled. Depending on desired behavior, one might
            # call loop.default_exception_handler(context_dict) or implement specific recovery.

        effective_loop.set_exception_handler(global_async_exception_handler)
        logger_to_use.info(f"Global asyncio exception handler installed on loop {id(effective_loop)}.")
