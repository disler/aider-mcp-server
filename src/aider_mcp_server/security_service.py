"""
Centralized security service implementation.
"""

import asyncio
from typing import Any, Dict, Optional

from aider_mcp_server.authentication_errors import AuthenticationError
from aider_mcp_server.interfaces.authentication_provider import IAuthenticationProvider
from aider_mcp_server.interfaces.security_service import ISecurityService
from aider_mcp_server.mcp_types import LoggerFactory
from aider_mcp_server.security import ANONYMOUS_SECURITY_CONTEXT, Permissions, SecurityContext


class SecurityService(ISecurityService):
    """
    Centralized security service that handles authentication, authorization, and security logging.
    This service is stateless and thread-safe.
    """

    def __init__(self, logger_factory: LoggerFactory, auth_provider: IAuthenticationProvider):
        """
        Initialize the security service.

        Args:
            logger_factory: Factory for creating loggers.
            auth_provider: Authentication provider for handling authentication.
        """
        self._logger = logger_factory(__name__)
        self._auth_provider = auth_provider
        self._lock = asyncio.Lock()  # For thread-safe operations if needed

    async def validate_token(self, token: Optional[str]) -> SecurityContext:
        """
        Validate an authentication token and return the appropriate security context.

        Args:
            token: The authentication token to validate, or None for anonymous access.

        Returns:
            A SecurityContext with the appropriate user identity and permissions.
        """
        # Log the authentication attempt
        await self.log_security_event(
            "auth_attempt", {"token_present": token is not None, "token_prefix": token[:10] if token else None}
        )

        if token is None:
            # Return anonymous context
            await self.log_security_event("auth_anonymous", {"user_id": "anonymous"})
            return ANONYMOUS_SECURITY_CONTEXT

        try:
            # Validate the token using the authentication provider
            if await self._auth_provider.validate_token(token):
                # Get user info from valid token
                user_info = await self._auth_provider.get_user_info(token)
                if user_info:
                    context = SecurityContext(
                        user_id=user_info.user_id,
                        permissions=user_info.permissions,  # type: ignore[arg-type]
                        is_anonymous=False,
                    )
                    await self.log_security_event(
                        "auth_success",
                        {"user_id": context.user_id, "permissions": [p.value for p in context.permissions]},
                    )
                    return context

            # Invalid token
            await self.log_security_event("auth_failure", {"reason": "invalid_token", "token_prefix": token[:10]})
            return ANONYMOUS_SECURITY_CONTEXT

        except AuthenticationError as e:
            await self.log_security_event(
                "auth_failure", {"reason": str(e), "token_prefix": token[:10] if token else None}
            )
            return ANONYMOUS_SECURITY_CONTEXT
        except Exception as e:
            self._logger.error(f"Unexpected error during token validation: {e}")
            await self.log_security_event("auth_error", {"error": str(e)})
            return ANONYMOUS_SECURITY_CONTEXT

    async def check_permission(self, context: SecurityContext, permission: Permissions) -> bool:
        """
        Check if the given security context has the specified permission.

        Args:
            context: The security context to check.
            permission: The permission to check for.

        Returns:
            True if the context has the permission, False otherwise.
        """
        has_permission = permission in context.permissions

        # Log the permission check
        await self.log_security_event(
            "permission_check", {"user_id": context.user_id, "permission": permission.value, "granted": has_permission}
        )

        return has_permission

    async def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Log a security-related event.

        Args:
            event_type: The type of security event.
            details: Additional details about the event.
        """
        # Use the logger in a thread-safe manner
        async with self._lock:
            self._logger.info(f"Security event: {event_type}", extra={"details": details})

            # Log specific event types with appropriate severity
            if event_type == "auth_failure":
                self._logger.warning(f"Authentication failure: {details}")
            elif event_type == "permission_denied":
                self._logger.warning(f"Permission denied: {details}")
            elif event_type in ["auth_success", "permission_check"]:
                self._logger.debug(f"Security event {event_type}: {details}")
            else:
                self._logger.info(f"Security event {event_type}: {details}")
