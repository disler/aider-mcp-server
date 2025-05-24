"""
Tests for the updated ResponseFormatter class.
"""

from unittest.mock import MagicMock

import pytest

from aider_mcp_server.atoms.errors.application_errors import BaseApplicationError
from aider_mcp_server.error_formatter import ErrorResponseFormatter
from aider_mcp_server.molecules.handlers.response_formatter import ResponseFormatter


@pytest.fixture
def logger_factory():
    """Create a logger factory mock."""
    mock_logger = MagicMock()
    logger_factory = MagicMock(return_value=mock_logger)
    return logger_factory


@pytest.fixture
def error_formatter():
    """Create an error formatter mock."""
    return MagicMock(spec=ErrorResponseFormatter)


@pytest.fixture
def response_formatter(logger_factory, error_formatter):
    """Create a ResponseFormatter instance for testing."""
    return ResponseFormatter(logger_factory, error_formatter)


def test_init(logger_factory, error_formatter):
    """Test ResponseFormatter initialization."""
    formatter = ResponseFormatter(logger_factory, error_formatter)
    assert formatter._logger == logger_factory.return_value
    assert formatter._error_formatter == error_formatter


def test_format_success_response(response_formatter):
    """Test formatting successful responses."""
    result = {"data": "test_data"}
    response = response_formatter.format_success_response("req-1", "transport-1", result)

    assert response["success"] is True
    assert response["request_id"] == "req-1"
    assert response["transport_id"] == "transport-1"
    assert response["result"] == result


def test_format_error_response(response_formatter):
    """Test formatting standard error responses."""
    response = response_formatter.format_error_response("req-1", "transport-1", "An error occurred")

    assert response["success"] is False
    assert response["request_id"] == "req-1"
    assert response["transport_id"] == "transport-1"
    assert response["error"]["message"] == "An error occurred"
    assert "code" not in response["error"]
    assert "details" not in response["error"]


def test_format_error_response_application_error(response_formatter, error_formatter):
    """Test formatting error responses with BaseApplicationError."""
    error = BaseApplicationError("Something went wrong", "APP_ERROR", {"id": 123})
    error_formatter.format_exception_to_response.return_value = {
        "success": False,
        "error": {
            "message": "Something went wrong",
            "code": "APP_ERROR",
            "details": {"id": 123},
        },
    }

    response = response_formatter.format_error_response("req-1", "transport-1", error)

    assert response == {
        "success": False,
        "request_id": "req-1",
        "transport_id": "transport-1",
        "error": {
            "message": "Something went wrong",
            "code": "APP_ERROR",
            "details": {"id": 123},
        },
    }
    error_formatter.format_exception_to_response.assert_called_once_with(error)


def test_format_exception_response(response_formatter, error_formatter):
    """Test formatting exception responses."""
    exc = ValueError("Invalid value")
    error_formatter.format_exception_to_response.return_value = {
        "success": False,
        "error": {
            "message": "An internal error occurred",
            "code": "internal_server_error",
        },
    }

    response = response_formatter.format_exception_response("req-1", "transport-1", exc)

    assert response == {
        "success": False,
        "request_id": "req-1",
        "transport_id": "transport-1",
        "error": {
            "message": "An internal error occurred",
            "code": "internal_server_error",
        },
    }
    error_formatter.format_exception_to_response.assert_called_once_with(exc)


def test_get_transport_specific_formatter(response_formatter):
    """Test getting transport-specific formatters."""
    assert response_formatter.get_transport_specific_formatter("sse") is None
    assert response_formatter.get_transport_specific_formatter("stdio") is None
