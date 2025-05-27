"""
Module for formatting error responses in a standardized way.
"""

from typing import Any, Callable, Dict, Optional

from aider_mcp_server.atoms.errors.application_errors import BaseApplicationError
from aider_mcp_server.atoms.types.mcp_types import LoggerFactory, OperationResult


class ErrorResponseFormatter:
    """
    Formats exceptions into standardized error responses.

    Responsibilities:
    - Convert exceptions to error responses
    - Sanitize sensitive error details
    - Format errors for specific transport types
    - Integrate with existing response formatter
    """

    def __init__(self, logger_factory: LoggerFactory) -> None:
        """
        Initialize the ErrorResponseFormatter.

        Args:
            logger_factory: Factory function to create loggers
        """
        self._logger = logger_factory(__name__)

    def format_exception_to_response(self, exc: Exception) -> OperationResult:
        """
        Convert an exception to a standardized error response.

        Args:
            exc: The exception to convert

        Returns:
            Formatted error response
        """
        if isinstance(exc, BaseApplicationError):
            error_code = exc.error_code
            user_message = exc.user_friendly_message
            details = self.sanitize_error_details(exc.details)
        else:
            error_code = "internal_server_error"
            user_message = "An internal error occurred"
            details = {}

        error_data: Dict[str, Any] = {
            "message": user_message,
            "code": error_code,
            "details": details,
        }

        error_response: OperationResult = {
            "success": False,
            "error": error_data,
        }

        return error_response

    def sanitize_error_details(self, details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Remove sensitive information from error details.

        Args:
            details: Raw error details dictionary

        Returns:
            Sanitized error details
        """
        sanitized = {}
        if details:
            for key, value in details.items():
                if key.lower() not in ["password", "token", "secret"]:
                    sanitized[key] = value
        return sanitized

    def format_for_transport(self, error_response: OperationResult, transport_type: str) -> OperationResult:
        """
        Format an error response for a specific transport type.

        Args:
            error_response: The error response to format
            transport_type: The transport type (e.g. 'sse', 'stdio')

        Returns:
            Transport-specific formatted error response
        """
        formatter = self.get_transport_specific_formatter(transport_type)
        if formatter:
            formatted_response = formatter(error_response)
        else:
            formatted_response = error_response

        return formatted_response

    def get_transport_specific_formatter(
        self, transport_type: str
    ) -> Optional[Callable[[OperationResult], Dict[str, Any]]]:
        """
        Get the transport-specific error formatting function.

        Args:
            transport_type: Type of transport (e.g., 'sse', 'stdio')

        Returns:
            Transport-specific error formatting function or None if not available
        """
        transport_formatters: Dict[str, Callable[[OperationResult], Dict[str, Any]]] = {
            # Example: "sse": lambda resp: {"event": "error", "data": resp}
        }
        return transport_formatters.get(transport_type)
