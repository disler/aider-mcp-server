"""
Security context management for the Aider MCP Server.

Defines permissions, the security context object, and functions for
creating security contexts based on credentials.
"""

import enum
import logging
from typing import Any, Dict, Optional, Set, Union

# Use standard logging
logger = logging.getLogger(__name__)


# Define Permissions enum
class Permissions(enum.Enum):
    """Defines the set of possible permissions within the system."""

    EXECUTE_AIDER = "execute_aider"  # Renamed for clarity
    VIEW_CONFIG = "view_config"
    LIST_MODELS = "list_models"  # Added permission
    # Add other relevant permissions as needed

    @classmethod
    def from_string(cls, value: str) -> Optional["Permissions"]:
        """Convert a string value to a Permissions enum member."""
        for member in cls:
            if member.value == value:
                return member
        return None


# Define the SecurityContext class
class SecurityContext:
    """
    Represents the security context of a request, including user identity
    and granted permissions.
    """

    # Add transport_id for context logging
    def __init__(
        self,
        user_id: Optional[str] = None,
        permissions: Optional[Set[Union[Permissions, str]]] = None,  # Allow string permissions initially
        is_anonymous: bool = False,
        transport_id: Optional[str] = None,  # Added transport ID
    ):
        """
        Initializes the SecurityContext.

        Args:
            user_id: An identifier for the user, if authenticated.
            permissions: A set of Permissions granted to the user (can include strings).
            is_anonymous: True if the context represents an anonymous user.
            transport_id: Optional ID of the transport associated with this context.
        """
        self.user_id = user_id
        self.transport_id = transport_id  # Store transport ID
        self.is_anonymous = is_anonymous
        processed_permissions: Set[Permissions] = set()

        if permissions:
            for perm in permissions:
                if isinstance(perm, Permissions):
                    processed_permissions.add(perm)
                elif isinstance(perm, str):
                    enum_perm = Permissions.from_string(perm)
                    if enum_perm:
                        processed_permissions.add(enum_perm)
                    elif perm == "*":  # Handle wildcard permission
                        # Grant all known permissions if '*' is present
                        processed_permissions.update(Permissions)
                        logger.warning(
                            f"Granting all permissions due to wildcard '*' in context for user '{user_id or 'anonymous'}'."
                        )
                        break  # No need to process further permissions if wildcard found
                    else:
                        logger.warning(f"Ignoring unknown permission string '{perm}' during SecurityContext creation.")
        self.permissions: Set[Permissions] = processed_permissions

        if self.is_anonymous:
            self.user_id = None  # Ensure anonymous users have no user_id
            self.permissions = set()  # Ensure anonymous users have no permissions

    def has_permission(self, required_permission: Union[Permissions, str]) -> bool:
        """
        Check if the context has the required permission.

        Args:
            required_permission: The permission to check for, either as a
                                 Permissions enum member or its string value.

        Returns:
            True if the permission is granted, False otherwise.
        """
        if self.is_anonymous:
            # Anonymous users typically have no permissions unless explicitly granted
            # (which is handled during init - anonymous context has empty permissions set)
            return False

        permission_to_check: Optional[Permissions] = None
        if isinstance(required_permission, Permissions):
            permission_to_check = required_permission
        elif isinstance(required_permission, str):
            permission_to_check = Permissions.from_string(required_permission)
            if permission_to_check is None:
                logger.warning(f"Attempted to check for unknown permission string: '{required_permission}'")
                return False
        else:
            # This is needed only for ensuring exhaustive type checking
            raise TypeError(f"Invalid type for required_permission: {type(required_permission)}")

        # Check if the specific permission exists in the user's granted permissions
        return permission_to_check in self.permissions

    def __repr__(self) -> str:
        perm_str = ", ".join(p.value for p in sorted(self.permissions, key=lambda x: x.value))
        context_type = "Anonymous" if self.is_anonymous else f"User='{self.user_id}'"
        transport_info = f", Transport='{self.transport_id}'" if self.transport_id else ""
        return f"SecurityContext({context_type}, Permissions={{{perm_str}}}{transport_info})"


# Define a default anonymous security context
# Provide a default transport_id or leave as None
ANONYMOUS_SECURITY_CONTEXT = SecurityContext(is_anonymous=True, transport_id="anonymous")


# Implement the `create_context_from_credentials` function
def create_context_from_credentials(credentials: Dict[str, Any], transport_id: Optional[str] = None) -> SecurityContext:
    """
    Creates a SecurityContext based on the provided credentials dictionary.

    Args:
        credentials: A dictionary containing authentication information.
        transport_id: Optional ID of the transport making the request.

    Returns:
        A SecurityContext instance.
    """
    # Avoid logging secrets by logging only keys or presence of keys
    cred_keys = list(credentials.keys())
    logger.debug(
        f"Attempting to create security context from credentials (keys: {cred_keys}) for transport '{transport_id}'"
    )

    auth_token = credentials.get("auth_token")

    if auth_token and isinstance(auth_token, str):
        # --- Simple Hardcoded Token Check (Replace with real validation) ---
        if auth_token == "VALID_TEST_TOKEN":  # noqa: S105
            user_id = "test_user"
            # Grant specific permissions - explicitly type to satisfy mypy with Union
            granted_permissions: Set[Union[Permissions, str]] = {
                Permissions.EXECUTE_AIDER,
                Permissions.LIST_MODELS,
                Permissions.VIEW_CONFIG,
            }
            logger.info(
                f"Successfully created authenticated context for user '{user_id}' from transport '{transport_id}'"
            )
            return SecurityContext(
                user_id=user_id,
                permissions=granted_permissions,
                transport_id=transport_id,
            )
        elif auth_token == "ADMIN_TOKEN":  # noqa: S105
            user_id = "admin_user"
            # Grant all permissions using wildcard - explicitly type to satisfy mypy with Union
            granted_permissions = {"*"}  # Use wildcard string
            logger.info(f"Successfully created admin context for user '{user_id}' from transport '{transport_id}'")
            return SecurityContext(
                user_id=user_id,
                permissions=granted_permissions,
                transport_id=transport_id,
            )
        else:
            logger.warning(f"Invalid auth_token provided from transport '{transport_id}'.")
            # Fall through to return anonymous context

    # If no valid credentials found, return the anonymous context
    logger.debug(
        f"No valid credentials found or processed for transport '{transport_id}', returning anonymous context."
    )
    # Pass transport_id to the anonymous context for logging clarity
    return SecurityContext(is_anonymous=True, transport_id=transport_id)
