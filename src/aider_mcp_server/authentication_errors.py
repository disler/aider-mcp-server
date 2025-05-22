"""
Authentication-related exception classes.
"""


class AuthenticationError(Exception):
    """Base exception for authentication errors."""

    pass


class InvalidCredentialsError(AuthenticationError):
    """Raised when provided credentials are invalid."""

    pass


class TokenExpiredError(AuthenticationError):
    """Raised when a token has expired."""

    pass


class TokenInvalidError(AuthenticationError):
    """Raised when a token is invalid or tampered with."""

    pass


class PermissionDeniedError(AuthenticationError):
    """Raised when user lacks required permissions."""

    pass


class AuthenticationProviderError(AuthenticationError):
    """Raised when authentication provider encounters an error."""

    pass
