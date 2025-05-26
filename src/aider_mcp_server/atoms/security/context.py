"""
Security context management for the Aider MCP Server.

Provides a simple, atomic security context object for managing user identity
and permissions in a request/response cycle.
"""

import logging
from typing import Optional, Set, Union

from aider_mcp_server.atoms.security.permissions import Permissions

# Use standard logging
logger = logging.getLogger(__name__)


class SecurityContext:
    """
    Represents the security context of a request, including user identity
    and granted permissions.

    This is an atomic component focused solely on context representation.
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        permissions: Optional[Set[Union[Permissions, str]]] = None,
        is_anonymous: bool = False,
        transport_id: Optional[str] = None,
    ):
        """
        Initialize the SecurityContext.

        Args:
            user_id: An identifier for the user, if authenticated.
            permissions: A set of Permissions granted to the user.
            is_anonymous: True if the context represents an anonymous user.
            transport_id: Optional ID of the transport associated with this context.
        """
        self.user_id = user_id
        self.transport_id = transport_id
        self.is_anonymous = is_anonymous

        # Process permissions with simple logic
        processed_permissions: Set[Permissions] = set()
        if permissions and not is_anonymous:
            for perm in permissions:
                if isinstance(perm, Permissions):
                    processed_permissions.add(perm)
                elif isinstance(perm, str):
                    enum_perm = Permissions.from_string(perm)
                    if enum_perm:
                        processed_permissions.add(enum_perm)
                    elif perm == "*":
                        # Grant all permissions for wildcard
                        processed_permissions.update(Permissions)
                        break

        self.permissions = processed_permissions

        # Ensure anonymous users have no identity or permissions
        if self.is_anonymous:
            self.user_id = None
            self.permissions = set()

    def has_permission(self, required_permission: Union[Permissions, str]) -> bool:
        """
        Check if the context has the required permission.

        Args:
            required_permission: The permission to check for.

        Returns:
            True if the permission is granted, False otherwise.
        """
        if self.is_anonymous:
            return False

        if isinstance(required_permission, str):
            enum_perm = Permissions.from_string(required_permission)
            if not enum_perm:
                return False
            required_permission = enum_perm

        return required_permission in self.permissions

    def has_any_permission(self, required_permissions: Set[Union[Permissions, str]]) -> bool:
        """Check if the context has any of the required permissions."""
        return any(self.has_permission(perm) for perm in required_permissions)

    def has_all_permissions(self, required_permissions: Set[Union[Permissions, str]]) -> bool:
        """Check if the context has all of the required permissions."""
        return all(self.has_permission(perm) for perm in required_permissions)

    def __str__(self) -> str:
        """String representation of the security context."""
        if self.is_anonymous:
            return f"SecurityContext(anonymous, transport={self.transport_id})"
        return (
            f"SecurityContext(user={self.user_id}, permissions={len(self.permissions)}, transport={self.transport_id})"
        )


# Anonymous security context singleton for reuse
ANONYMOUS_SECURITY_CONTEXT = SecurityContext(is_anonymous=True)
