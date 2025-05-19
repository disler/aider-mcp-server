"""
Custom exception hierarchy for the application.
"""

from typing import Any, Dict, Optional



class BaseApplicationError(Exception):
    """Base class for all application-specific exceptions."""

    def __init__(self, error_code: str, user_friendly_message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize a new BaseApplicationError.

        Args:
            error_code: A unique code identifying the error.
            user_friendly_message: A message suitable for displaying to end users.
            details: Additional details about the error for debugging purposes.
        """
        super().__init__(user_friendly_message)
        self.error_code = error_code
        self.user_friendly_message = user_friendly_message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception to a dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "user_friendly_message": self.user_friendly_message,
            "details": self.details,
        }


class TransportError(BaseApplicationError):
    """Raised when there is an error related to communication or connectivity."""

    pass


class SecurityError(BaseApplicationError):
    """Raised when there is an authentication or authorization failure."""

    pass


class ValidationError(BaseApplicationError):
    """Raised when input validation fails."""

    pass


class ProcessingError(BaseApplicationError):
    """Raised when there is an error during business logic processing."""

    pass


class ConnectionFailedError(TransportError):
    """Raised when a connection cannot be established or is lost."""

    pass


class OperationTimeoutError(TransportError):
    """Raised when an operation times out."""

    pass


class MessageFormatError(TransportError):
    """Raised when a message is malformed or cannot be parsed."""

    pass


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    
    pass


class AuthorizationError(SecurityError):
    """Raised when the user lacks sufficient permissions."""

    pass


class TokenError(SecurityError):
    """Raised when there is an issue with an authentication token."""

    pass


class InputValidationError(ValidationError):
    """Raised when input data fails validation checks."""

    pass


class SchemaValidationError(ValidationError):
    """Raised when data does not conform to the expected schema."""

    pass


class ConstraintViolationError(ValidationError):
    """Raised when a data constraint is violated."""

    pass


class ResourceNotFoundError(ProcessingError):
    """Raised when a requested resource cannot be found."""

    pass


class OperationFailedError(ProcessingError):
    """Raised when an operation fails to complete successfully."""

    pass


class ConflictError(ProcessingError):
    """Raised when there is a conflict with the current state of a resource."""

    pass


class RateLimitError(ProcessingError):
    """Raised when a rate limit has been exceeded."""

    pass
