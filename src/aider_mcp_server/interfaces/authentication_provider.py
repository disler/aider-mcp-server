"""
Authentication provider interface for modular authentication support.

This module defines the abstract authentication provider interface that allows
for different authentication methods to be plugged into the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Set

from aider_mcp_server.atoms.security.context import Permissions


@dataclass
class AuthToken:
    """Authentication token containing user identity and permissions."""

    token: str
    user_id: str
    issued_at: datetime
    expires_at: Optional[datetime] = None
    permissions: Optional[Set[Permissions]] = None

    def __post_init__(self) -> None:
        """Initialize permissions to empty set if None."""
        if self.permissions is None:
            self.permissions = set()

    def is_expired(self) -> bool:
        """Check if the token has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def has_permission(self, permission: Permissions) -> bool:
        """Check if the token has a specific permission."""
        if self.permissions is None:
            return False
        return permission in self.permissions


@dataclass
class UserInfo:
    """User information retrieved from authentication."""

    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None
    permissions: Optional[Set[Permissions]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Initialize fields to defaults if None."""
        if self.permissions is None:
            self.permissions = set()
        if self.metadata is None:
            self.metadata = {}


class IAuthenticationProvider(ABC):
    """Abstract interface for authentication providers."""

    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> AuthToken:
        """
        Authenticate user with provided credentials.

        Args:
            credentials: Dictionary containing authentication credentials
                        (e.g., username/password, API key, OAuth token)

        Returns:
            AuthToken with user identity and permissions

        Raises:
            AuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def validate_token(self, token: str) -> bool:
        """
        Validate an authentication token.

        Args:
            token: The token string to validate

        Returns:
            True if the token is valid, False otherwise
        """
        pass

    @abstractmethod
    async def get_user_info(self, token: str) -> Optional[UserInfo]:
        """
        Get user information from a valid token.

        Args:
            token: The authentication token

        Returns:
            UserInfo object if token is valid, None otherwise
        """
        pass

    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """
        Revoke an authentication token.

        Args:
            token: The token to revoke

        Returns:
            True if the token was successfully revoked, False otherwise
        """
        pass

    @abstractmethod
    async def refresh_token(self, token: str) -> Optional[AuthToken]:
        """
        Refresh an existing token.

        Args:
            token: The current token to refresh

        Returns:
            New AuthToken if refresh successful, None otherwise
        """
        pass
