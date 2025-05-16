"""
Tests for the ResponseFormatter class.
"""

from unittest.mock import MagicMock

import pytest

from aider_mcp_server.response_formatter import ResponseFormatter


@pytest.fixture
def logger_factory():
    """Create a logger factory mock."""
    mock_logger = MagicMock()
    logger_factory = MagicMock(return_value=mock_logger)
    return logger_factory


@pytest.fixture
def response_formatter(logger_factory):
    """Create a ResponseFormatter instance for testing."""
    return ResponseFormatter(logger_factory)


def test_format_success_response(response_formatter):
    """Test formatting successful responses."""
    # Test basic success response
    result = {"data": "test_data"}
    response = response_formatter.format_success_response("req-1", "transport-1", result)

    assert response["success"] is True
    assert response["request_id"] == "req-1"
    assert response["transport_id"] == "transport-1"
    assert response["result"] == result


def test_format_error_response(response_formatter):
    """Test formatting error responses with different error details."""
    # Test basic error response
    response = response_formatter.format_error_response("req-1", "transport-1", "An error occurred")

    assert response["success"] is False
    assert response["request_id"] == "req-1"
    assert response["transport_id"] == "transport-1"
    assert response["error"]["message"] == "An error occurred"
    assert "code" not in response["error"]
    assert "details" not in response["error"]

    # Test error response with error code
    response = response_formatter.format_error_response(
        "req-2", "transport-1", "Permission denied", error_code="AUTH_ERROR"
    )

    assert response["success"] is False
    assert response["error"]["message"] == "Permission denied"
    assert response["error"]["code"] == "AUTH_ERROR"
    assert "details" not in response["error"]

    # Test error response with details
    details = {"field": "username", "reason": "Already exists"}
    response = response_formatter.format_error_response("req-3", "transport-1", "Validation error", details=details)

    assert response["success"] is False
    assert response["error"]["message"] == "Validation error"
    assert response["error"]["details"] == details
    assert "code" not in response["error"]

    # Test error response with both code and details
    response = response_formatter.format_error_response(
        "req-4",
        "transport-1",
        "Invalid input",
        error_code="VALIDATION_ERROR",
        details={"errors": ["Field required"]},
    )

    assert response["success"] is False
    assert response["error"]["message"] == "Invalid input"
    assert response["error"]["code"] == "VALIDATION_ERROR"
    assert response["error"]["details"] == {"errors": ["Field required"]}


def test_get_transport_specific_formatter(response_formatter):
    """Test getting transport-specific formatters."""
    # Currently returns None for any transport type as no formatters are defined
    assert response_formatter.get_transport_specific_formatter("sse") is None
    assert response_formatter.get_transport_specific_formatter("stdio") is None
