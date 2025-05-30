"""
Centralized security service implementation.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt

from aider_mcp_server.atoms.security import Permissions
from aider_mcp_server.atoms.security.context import ANONYMOUS_SECURITY_CONTEXT, SecurityContext
from aider_mcp_server.atoms.security.errors import AuthenticationError
from aider_mcp_server.atoms.types.mcp_types import LoggerFactory
from aider_mcp_server.interfaces.authentication_provider import IAuthenticationProvider
from aider_mcp_server.interfaces.security_service import ISecurityService


class SecurityService(ISecurityService):
    """
    Centralized security service that handles authentication, authorization, and security logging.
    This service is stateless and thread-safe.
    """

    # It is CRUCIAL that JWT_SECRET_KEY is managed securely and not hardcoded in production.
    # It should be loaded from a secure configuration source.
    DEFAULT_JWT_SECRET_KEY = "your-super-secret-and-long-jwt-secret-key"  # Placeholder # noqa: S105
    DEFAULT_JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Default expiration time

    def __init__(
        self,
        logger_factory: LoggerFactory,
        auth_provider: IAuthenticationProvider,
        jwt_secret_key: str = DEFAULT_JWT_SECRET_KEY,
        jwt_expire_minutes: int = DEFAULT_JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    ):
        """
        Initialize the security service.

        Args:
            logger_factory: Factory for creating loggers.
            auth_provider: Authentication provider for handling authentication.
            jwt_secret_key: Secret key for encoding/decoding JWTs.
            jwt_expire_minutes: Expiration time for JWT access tokens in minutes.
        """
        self._logger = logger_factory(__name__)
        self._auth_provider = auth_provider
        self._jwt_secret_key = jwt_secret_key
        self._jwt_expire_minutes = jwt_expire_minutes
        self._lock = asyncio.Lock()  # For thread-safe operations if needed

        if jwt_secret_key == self.DEFAULT_JWT_SECRET_KEY:
            self._logger.warning("Using default JWT secret key. This is insecure and should be changed for production.")

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

    async def generate_session_token(self, client_id: str) -> str:
        """
        Generate a JWT session token for a given client ID.
        """
        issue_time = datetime.now(timezone.utc)
        expire_time = issue_time + timedelta(minutes=self._jwt_expire_minutes)

        payload = {
            "sub": client_id,  # Standard claim for subject (client_id in this case)
            "iat": issue_time,  # Standard claim for issued at
            "exp": expire_time,  # Standard claim for expiration time
            "type": "session",  # Custom claim to identify token type
        }
        try:
            encoded_jwt: str = jwt.encode(payload, self._jwt_secret_key, algorithm="HS256")
            await self.log_security_event(
                "session_token_generated", {"client_id": client_id, "expires_at": expire_time.isoformat()}
            )
            return encoded_jwt
        except Exception as e:
            self._logger.error(f"Error generating session token for client {client_id}: {e}")
            await self.log_security_event("session_token_generation_failed", {"client_id": client_id, "error": str(e)})
            # Depending on policy, could raise an error or return a specific indicator
            raise  # Re-raise the exception as token generation is critical

    async def validate_session_token(self, token: str) -> Optional[str]:
        """
        Validate a JWT session token and extract the client ID.
        Returns client_id if valid, None otherwise.
        """
        token_prefix = token[:10] if token else None
        try:
            payload: Dict[str, Any] = jwt.decode(token, self._jwt_secret_key, algorithms=["HS256"])

            client_id_val = payload.get("sub")
            token_type_val = payload.get("type")

            # Validate client_id: must be a non-empty string
            if not isinstance(client_id_val, str) or not client_id_val:
                await self.log_security_event(
                    "session_token_validation_failed",
                    {"reason": "invalid_or_missing_client_id_in_payload", "token_prefix": token_prefix},
                )
                return None

            # Validate token_type: must be "session"
            if token_type_val != "session":  # noqa: S105
                await self.log_security_event(
                    "session_token_validation_failed",
                    {
                        "reason": "invalid_token_type_in_payload",
                        "token_prefix": token_prefix,
                        "expected_type": "session",
                        "actual_type": str(token_type_val),
                    },
                )
                return None

            # If all checks pass, client_id_val is a valid non-empty string
            client_id: str = client_id_val

            await self.log_security_event(
                "session_token_validated", {"client_id": client_id, "token_prefix": token_prefix}
            )
            return client_id
        except jwt.ExpiredSignatureError:
            await self.log_security_event(
                "session_token_validation_failed", {"reason": "expired_token", "token_prefix": token_prefix}
            )
            self._logger.warning(f"Expired session token received: {token_prefix}...")
            return None
        except jwt.InvalidTokenError as e:
            await self.log_security_event(
                "session_token_validation_failed", {"reason": f"invalid_token: {str(e)}", "token_prefix": token_prefix}
            )
            self._logger.warning(f"Invalid session token received: {token_prefix}... Error: {e}")
            return None
        except Exception as e:
            # Catch any other unexpected errors during decoding
            self._logger.error(f"Unexpected error validating session token {token_prefix}...: {e}")
            await self.log_security_event(
                "session_token_validation_error", {"error": str(e), "token_prefix": token_prefix}
            )
            return None
