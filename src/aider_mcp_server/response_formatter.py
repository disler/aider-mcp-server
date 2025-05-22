"""
Response formatter for standardizing response formatting for different transport types.
"""

from typing import Any, Dict, Optional

from aider_mcp_server.application_errors import BaseApplicationError
from aider_mcp_server.error_formatter import ErrorResponseFormatter
from aider_mcp_server.mcp_types import LoggerFactory, OperationResult


class ResponseFormatter:
    """
    Handles formatting of responses for different transport types.

    Responsibilities:
    - Standardizes error response formatting
    - Customizes response format based on transport type
    - Ensures consistent response structures
    """

    def __init__(self, logger_factory: LoggerFactory, error_formatter: ErrorResponseFormatter) -> None:
        """
        Initialize the ResponseFormatter.

        Args:
            logger_factory: Factory function to create loggers
            error_formatter: Formatter for standardizing error responses
        """
        self._logger = logger_factory(__name__)
        self._error_formatter = error_formatter

    def format_success_response(self, request_id: str, transport_id: str, result: Dict[str, Any]) -> OperationResult:
        """
        Format a successful response.

        Args:
            request_id: Unique identifier for the request
            transport_id: Identifier for the transport that made the request
            result: The operation result data

        Returns:
            Formatted operation result
        """
        self._logger.debug(f"Formatting success response for request {request_id}")
        response: OperationResult = {
            "success": True,
            "request_id": request_id,
            "transport_id": transport_id,
            "result": result,
        }
        return response

    def format_error_response(
        self,
        request_id: str,
        transport_id: str,
        error_message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> OperationResult:
        """
        Format an error response.

        Args:
            request_id: Unique identifier for the request
            transport_id: Identifier for the transport that made the request
            error_message: Human-readable error message
            error_code: Optional error code for categorization
            details: Optional additional error details

        Returns:
            Formatted operation result with error information
        """
        self._logger.debug(f"Formatting error response for request {request_id}: {error_message}")

        if isinstance(error_message, BaseApplicationError):
            error_response = self._error_formatter.format_exception_to_response(error_message)
        else:
            error_data: Dict[str, Any] = {
                "message": error_message,
            }

            if error_code:
                error_data["code"] = error_code

            if details:
                error_data["details"] = details

            error_response = {
                "success": False,
                "error": error_data,
            }

        error_response["request_id"] = request_id
        error_response["transport_id"] = transport_id

        return error_response

    def format_exception_response(self, request_id: str, transport_id: str, exc: Exception) -> OperationResult:
        """
        Format an error response from an exception.

        Args:
            request_id: Unique identifier for the request
            transport_id: Identifier for the transport that made the request
            exc: The exception to format into a response

        Returns:
            Formatted operation result with error information
        """
        self._logger.exception(f"Formatting exception response for request {request_id}")

        error_response = self._error_formatter.format_exception_to_response(exc)
        error_response["request_id"] = request_id
        error_response["transport_id"] = transport_id

        return error_response

    def get_transport_specific_formatter(self, transport_type: str) -> Optional[Dict[str, Any]]:
        """
        Get transport-specific formatting configurations.

        Args:
            transport_type: Type of transport (e.g., 'sse', 'stdio')

        Returns:
            Transport-specific formatting configuration or None if not available
        """
        # Currently no custom formatters, but this provides extension point
        transport_formatters: Dict[str, Dict[str, Any]] = {
            # Example: "sse": {"wrap_events": True, "include_timestamp": True}
        }

        return transport_formatters.get(transport_type)
