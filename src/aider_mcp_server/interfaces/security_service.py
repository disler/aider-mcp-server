"""
Security service interface for centralized security operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from aider_mcp_server.atoms.security.context import Permissions, SecurityContext


class ISecurityService(ABC):
    """Interface for security service operations."""

    @abstractmethod
    async def validate_token(self, token: Optional[str]) -> SecurityContext:
        """
        Validate an authentication token and return the appropriate security context.

        Args:
            token: The authentication token to validate, or None for anonymous access.

        Returns:
            A SecurityContext with the appropriate user identity and permissions.
        """
        pass

    @abstractmethod
    async def check_permission(self, context: SecurityContext, permission: Permissions) -> bool:
        """
        Check if the given security context has the specified permission.

        Args:
            context: The security context to check.
            permission: The permission to check for.

        Returns:
            True if the context has the permission, False otherwise.
        """
        pass

    @abstractmethod
    async def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Log a security-related event.

        Args:
            event_type: The type of security event (e.g., "auth_success", "auth_failure", "permission_denied").
            details: Additional details about the event.
        """
        pass
