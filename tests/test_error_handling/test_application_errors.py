import pytest

from aider_mcp_server.atoms.errors.application_errors import (
    AuthorizationError,
    BaseApplicationError,
    ConnectionFailedError,
    InputValidationError,
    ProcessingError,
    ResourceNotFoundError,
    SecurityError,
    TransportError,
    ValidationError,
)
from aider_mcp_server.atoms.types.data_types import MCPErrorResponse


def test_base_application_error():
    """Test BaseApplicationError initialization and to_dict method."""
    error = BaseApplicationError("test_code", "Test message", {"detail": "value"})
    assert error.error_code == "test_code"
    assert error.user_friendly_message == "Test message"
    assert error.details == {"detail": "value"}
    assert error.to_dict() == {
        "error_code": "test_code",
        "user_friendly_message": "Test message",
        "details": {"detail": "value"},
    }


@pytest.mark.parametrize("error_class", [TransportError, SecurityError, ValidationError, ProcessingError])
def test_main_error_categories(error_class):
    """Test main error category classes."""
    error = error_class("test_code", "Test message")
    assert isinstance(error, BaseApplicationError)
    assert error.error_code == "test_code"
    assert error.user_friendly_message == "Test message"


@pytest.mark.parametrize(
    "error_class,parent_class",
    [
        (ConnectionFailedError, TransportError),
        (AuthorizationError, SecurityError),
        (InputValidationError, ValidationError),
        (ResourceNotFoundError, ProcessingError),
    ],
)
def test_error_subclasses(error_class, parent_class):
    """Test specific error subclasses."""
    error = error_class("test_code", "Test message")
    assert isinstance(error, parent_class)
    assert error.error_code == "test_code"
    assert error.user_friendly_message == "Test message"


def test_mcp_error_response_from_base_application_error():
    """Test MCPErrorResponse.from_exception with BaseApplicationError."""
    error = BaseApplicationError("test_code", "Test message", {"detail": "value"})
    response = MCPErrorResponse.from_exception(error)
    assert response.error_code == "test_code"
    assert response.message == "Test message"
    assert response.details == {"detail": "value"}


def test_mcp_error_response_from_standard_exception():
    """Test MCPErrorResponse.from_exception with a standard Exception."""
    error = Exception("Test standard exception")
    response = MCPErrorResponse.from_exception(error)
    assert response.error_code == "internal_server_error"
    assert response.message == "Test standard exception"
    assert response.details == {}
