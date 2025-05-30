"""
Security service interface for centralized security operations.
"""

from typing import Any, Dict, Optional, Protocol

from typing_extensions import runtime_checkable

from aider_mcp_server.atoms.security import Permissions
from aider_mcp_server.atoms.security.context import SecurityContext


@runtime_checkable
class ISecurityService(Protocol):
    """Interface for security service operations."""

    async def validate_token(self, token: Optional[str]) -> SecurityContext:
        """
        Validate an authentication token and return the appropriate security context.

        Args:
            token: The authentication token to validate, or None for anonymous access.

        Returns:
            A SecurityContext with the appropriate user identity and permissions.
        """
        ...

    async def check_permission(self, context: SecurityContext, permission: Permissions) -> bool:
        """
        Check if the given security context has the specified permission.

        Args:
            context: The security context to check.
            permission: The permission to check for.

        Returns:
            True if the context has the permission, False otherwise.
        """
        ...

    async def generate_session_token(self, client_id: str) -> str:
        """
        Generate a JWT session token for a given client ID.

        Args:
            client_id: The unique identifier for the client.

        Returns:
            A JWT string.
        """
        ...

    async def validate_session_token(self, token: str) -> Optional[str]:
        """
        Validate a JWT session token and extract the client ID.

        Args:
            token: The JWT string to validate.

        Returns:
            The client_id if the token is valid, None otherwise.
        """
        ...

    async def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Log a security-related event.

        Args:
            event_type: The type of security event (e.g., "auth_success", "auth_failure", "permission_denied").
            details: Additional details about the event.
        """
        ...
