"""
Default authentication provider implementation.

This module implements the authentication provider interface using
the existing authentication method for backward compatibility.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Set, Union

from aider_mcp_server.authentication_errors import InvalidCredentialsError
from aider_mcp_server.interfaces.authentication_provider import (
    AuthToken,
    IAuthenticationProvider,
    UserInfo,
)
from aider_mcp_server.mcp_types import LoggerFactory
from aider_mcp_server.security import Permissions

# Type alias for token data
TokenData = Dict[str, Union[str, Set[Permissions]]]

# Hardcoded tokens for testing - replace with secure storage in production
VALID_TOKENS: Dict[str, TokenData] = {
    "test-api-key": {
        "user_id": "test-user",
        "permissions": {Permissions.EXECUTE_AIDER, Permissions.LIST_MODELS, Permissions.VIEW_CONFIG},
    },
    "admin-api-key": {
        "user_id": "admin-user",
        "permissions": {
            Permissions.EXECUTE_AIDER,
            Permissions.LIST_MODELS,
            Permissions.VIEW_CONFIG,
        },
    },
}


class DefaultAuthenticationProvider(IAuthenticationProvider):
    """Default authentication provider using API keys."""

    def __init__(self, logger_factory: LoggerFactory) -> None:
        """Initialize the authentication provider."""
        self._logger = logger_factory(__name__)
        self._active_tokens: Dict[str, AuthToken] = {}

    async def authenticate(self, credentials: Dict[str, Any]) -> AuthToken:
        """
        Authenticate using API key credentials.

        Args:
            credentials: Must contain 'api_key' field

        Returns:
            AuthToken for the authenticated user

        Raises:
            InvalidCredentialsError: If credentials are invalid
        """
        api_key = credentials.get("api_key")
        if not api_key:
            raise InvalidCredentialsError("API key is required")

        # Check if API key is valid
        if api_key not in VALID_TOKENS:
            self._logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
            raise InvalidCredentialsError("Invalid API key")

        # Get user info from valid tokens
        user_data = VALID_TOKENS[api_key]

        # Create and store auth token
        auth_token = AuthToken(
            token=str(uuid.uuid4()),
            user_id=str(user_data["user_id"]),
            issued_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
            permissions=user_data["permissions"],  # type: ignore[arg-type]
        )

        self._active_tokens[auth_token.token] = auth_token
        self._logger.info(f"User '{auth_token.user_id}' authenticated successfully")

        return auth_token

    async def validate_token(self, token: str) -> bool:
        """
        Validate an authentication token.

        Args:
            token: The token to validate

        Returns:
            True if valid and not expired, False otherwise
        """
        if token not in self._active_tokens:
            return False

        auth_token = self._active_tokens[token]
        if auth_token.is_expired():
            # Remove expired token
            del self._active_tokens[token]
            self._logger.info(f"Token for user '{auth_token.user_id}' expired")
            return False

        return True

    async def get_user_info(self, token: str) -> Optional[UserInfo]:
        """
        Get user information from a valid token.

        Args:
            token: The authentication token

        Returns:
            UserInfo if token is valid, None otherwise
        """
        if not await self.validate_token(token):
            return None

        auth_token = self._active_tokens[token]
        return UserInfo(
            user_id=auth_token.user_id,
            username=auth_token.user_id,  # Using user_id as username for simplicity
            permissions=auth_token.permissions,
            metadata={"authenticated_at": auth_token.issued_at.isoformat()},
        )

    async def revoke_token(self, token: str) -> bool:
        """
        Revoke an authentication token.

        Args:
            token: The token to revoke

        Returns:
            True if token was revoked, False if not found
        """
        if token in self._active_tokens:
            auth_token = self._active_tokens[token]
            del self._active_tokens[token]
            self._logger.info(f"Token for user '{auth_token.user_id}' revoked")
            return True
        return False

    async def refresh_token(self, token: str) -> Optional[AuthToken]:
        """
        Refresh an existing token.

        Args:
            token: The current token to refresh

        Returns:
            New AuthToken if refresh successful, None otherwise
        """
        if not await self.validate_token(token):
            return None

        old_token = self._active_tokens[token]

        # Create new token with same permissions
        new_token = AuthToken(
            token=str(uuid.uuid4()),
            user_id=old_token.user_id,
            issued_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
            permissions=old_token.permissions,
        )

        # Remove old token and add new one
        del self._active_tokens[token]
        self._active_tokens[new_token.token] = new_token

        self._logger.info(f"Token refreshed for user '{new_token.user_id}'")
        return new_token
