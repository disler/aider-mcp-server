import pytest
from typing import Dict, Any

from aider_mcp_server.error_formatter import ErrorResponseFormatter
from aider_mcp_server.application_errors import (
    BaseApplicationError,
    TransportError,
    SecurityError,
    ValidationError,
    ProcessingError,
)


@pytest.fixture
def logger_factory_mock(mocker):
    return mocker.Mock()


@pytest.fixture
def error_formatter(logger_factory_mock):
    return ErrorResponseFormatter(logger_factory_mock)


def test_formats_base_application_error(error_formatter):
    error = BaseApplicationError("test_code", "Test message", {"foo": "bar"})
    result = error_formatter.format_exception_to_response(error)
    
    assert result == {
        "success": False,
        "error": {
            "message": "Test message",
            "code": "test_code",
            "details": {"foo": "bar"},
        },
    }


@pytest.mark.parametrize(
    "error_class",
    [TransportError, SecurityError, ValidationError, ProcessingError],
)
def test_formats_specific_error_types(error_formatter, error_class):
    error = error_class("test_code", "Test message", {"foo": "bar"})
    result = error_formatter.format_exception_to_response(error)

    assert result["error"]["code"] == "test_code"
    assert result["error"]["message"] == "Test message"
    assert result["error"]["details"] == {"foo": "bar"}


@pytest.mark.parametrize("error_class", [ValueError, RuntimeError])
def test_formats_standard_exceptions(error_formatter, error_class):
    error = error_class("Test message")
    result = error_formatter.format_exception_to_response(error)

    assert result["error"]["code"] == "internal_server_error"
    assert result["error"]["message"] == "An internal error occurred"
    assert result["error"]["details"] == {}


def test_sanitizes_sensitive_error_details(error_formatter):
    error = BaseApplicationError(
        "test_code", "Test message", {"password": "secret", "foo": "bar"}
    )
    result = error_formatter.format_exception_to_response(error)

    assert "password" not in result["error"]["details"]
    assert result["error"]["details"] == {"foo": "bar"}


def test_format_for_transport(error_formatter, mocker):
    transport_formatter = mocker.Mock(return_value={"formatted": True})
    mocker.patch.object(
        error_formatter,
        "get_transport_specific_formatter",
        return_value=transport_formatter,
    )

    response: Dict[str, Any] = {"success": False, "error": {}}
    result = error_formatter.format_for_transport(response, "test_transport")

    assert result == {"formatted": True}
    transport_formatter.assert_called_once_with(response)
