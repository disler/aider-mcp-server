"""Tests for the security service."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest

from aider_mcp_server.atoms.security.context import Permissions, SecurityContext
from aider_mcp_server.atoms.security.errors import AuthenticationError
from aider_mcp_server.interfaces.authentication_provider import UserInfo
from aider_mcp_server.molecules.security.security_service import SecurityService


@pytest.fixture
def mock_logger_factory():
    """Fixture for a mock logger factory."""
    mock_factory = MagicMock()
    mock_logger = MagicMock()
    mock_logger.info = MagicMock()
    mock_logger.warning = MagicMock()
    mock_logger.error = MagicMock()
    mock_logger.debug = MagicMock()
    mock_logger.verbose = MagicMock()  # Keep if used, or remove if not. Assuming it might be used.
    mock_factory.return_value = mock_logger
    return mock_factory


@pytest.fixture
def mock_auth_provider():
    """Fixture for a mock authentication provider."""
    mock_provider = AsyncMock()
    return mock_provider


@pytest.fixture
def security_service(mock_logger_factory, mock_auth_provider):
    """Fixture for a security service instance."""
    return SecurityService(mock_logger_factory, mock_auth_provider)


@pytest.mark.asyncio
async def test_validate_token_anonymous(security_service, mock_logger_factory):
    """Test validating a null token returns an anonymous context."""
    context = await security_service.validate_token(None)

    # Anonymous user should have None as user_id due to is_anonymous flag
    assert context.user_id is None
    assert context.is_anonymous is True
    assert not context.permissions

    # Check logging calls
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: auth_attempt", extra={"details": {"token_present": False, "token_prefix": None}}
    )

    # Check that anonymous auth event was logged - check all logging levels
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: auth_anonymous", extra={"details": {"user_id": "anonymous"}}
    )


@pytest.mark.asyncio
async def test_validate_token_valid_user(security_service, mock_logger_factory, mock_auth_provider):
    """Test validating a valid token returns a user context with permissions."""
    # Set up the mock auth provider
    mock_auth_provider.validate_token.return_value = True
    mock_auth_provider.get_user_info.return_value = UserInfo(
        user_id="test-user",
        permissions={Permissions.EXECUTE_AIDER},
    )

    # Use a valid token
    context = await security_service.validate_token("VALID_TEST_TOKEN")

    assert context.user_id == "test-user"
    assert Permissions.EXECUTE_AIDER in context.permissions

    # Verify auth provider was called
    mock_auth_provider.validate_token.assert_called_once_with("VALID_TEST_TOKEN")
    mock_auth_provider.get_user_info.assert_called_once_with("VALID_TEST_TOKEN")

    # Check logging calls - token_prefix is only 10 chars
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: auth_attempt", extra={"details": {"token_present": True, "token_prefix": "VALID_TEST"}}
    )


@pytest.mark.asyncio
async def test_validate_token_admin(security_service, mock_logger_factory, mock_auth_provider):
    """Test validating an admin token returns a context with admin permissions."""
    # Set up the mock auth provider for admin
    mock_auth_provider.validate_token.return_value = True
    mock_auth_provider.get_user_info.return_value = UserInfo(
        user_id="admin",
        permissions={Permissions.EXECUTE_AIDER, Permissions.LIST_MODELS, Permissions.VIEW_CONFIG},
    )

    context = await security_service.validate_token("ADMIN_TOKEN")

    assert context.user_id == "admin"
    assert len(context.permissions) > 0
    assert Permissions.EXECUTE_AIDER in context.permissions
    assert Permissions.LIST_MODELS in context.permissions

    # Verify auth provider was called
    mock_auth_provider.validate_token.assert_called_once_with("ADMIN_TOKEN")
    mock_auth_provider.get_user_info.assert_called_once_with("ADMIN_TOKEN")

    # Check logging call for auth success
    info_calls = mock_logger_factory.return_value.info.call_args_list
    found_auth_success = False
    for call in info_calls:
        if len(call.args) > 0:
            arg = call.args[0]
            if "auth_success" in arg:
                found_auth_success = True
                break
    assert found_auth_success


@pytest.mark.asyncio
async def test_check_permission_granted(security_service, mock_logger_factory):
    """Test permission check when permission is granted."""
    # Create a security context with the permission
    context = SecurityContext(user_id="test-user", permissions={Permissions.EXECUTE_AIDER})

    # Check the permission
    result = await security_service.check_permission(context, Permissions.EXECUTE_AIDER)

    # Verify the result
    assert result is True

    # Check logging calls
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: permission_check",
        extra={"details": {"user_id": "test-user", "permission": "execute_aider", "granted": True}},
    )


@pytest.mark.asyncio
async def test_check_permission_denied(security_service, mock_logger_factory):
    """Test permission check when permission is denied."""
    # Create a security context without the permission
    context = SecurityContext(user_id="test-user", permissions=set())

    # Check the permission
    result = await security_service.check_permission(context, Permissions.EXECUTE_AIDER)

    # Verify the result
    assert result is False

    # Check logging calls
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: permission_check",
        extra={"details": {"user_id": "test-user", "permission": "execute_aider", "granted": False}},
    )


@pytest.mark.asyncio
async def test_log_security_event(security_service, mock_logger_factory):
    """Test logging a security event."""
    # Log a security event
    event_type = "custom_event"
    details = {"key": "value", "another_key": 123}

    await security_service.log_security_event(event_type, details)

    # Check logging calls
    mock_logger_factory.return_value.info.assert_any_call(f"Security event: {event_type}", extra={"details": details})

    mock_logger_factory.return_value.info.assert_any_call(f"Security event {event_type}: {details}")


@pytest.mark.asyncio
async def test_log_security_event_auth_failure(security_service, mock_logger_factory):
    """Test logging an auth failure event with warning level."""
    # Log an auth failure event
    event_type = "auth_failure"
    details = {"token_prefix": "INVALID_", "reason": "token_not_found"}

    await security_service.log_security_event(event_type, details)

    # Check warning log call
    mock_logger_factory.return_value.warning.assert_any_call(f"Authentication failure: {details}")


@pytest.mark.asyncio
async def test_log_security_event_permission_denied(security_service, mock_logger_factory):
    """Test logging a permission denied event with warning level."""
    # Log a permission denied event
    event_type = "permission_denied"
    details = {"user_id": "test-user", "permission": "execute_aider"}

    await security_service.log_security_event(event_type, details)

    # Check warning log call
    mock_logger_factory.return_value.warning.assert_any_call(f"Permission denied: {details}")


@pytest.mark.asyncio
async def test_validate_token_invalid(security_service, mock_logger_factory, mock_auth_provider):
    """Test validating an invalid token returns anonymous context."""
    # Set up the mock auth provider to return False for invalid token
    mock_auth_provider.validate_token.return_value = False

    context = await security_service.validate_token("INVALID_TOKEN")

    # Should get anonymous context
    assert context.user_id is None
    assert context.is_anonymous is True
    assert not context.permissions

    # Verify auth provider was called
    mock_auth_provider.validate_token.assert_called_once_with("INVALID_TOKEN")
    # get_user_info should not be called for invalid token
    mock_auth_provider.get_user_info.assert_not_called()

    # Check logging for auth failure
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: auth_failure", extra={"details": {"reason": "invalid_token", "token_prefix": "INVALID_TO"}}
    )


@pytest.mark.asyncio
async def test_validate_token_auth_error(security_service, mock_logger_factory, mock_auth_provider):
    """Test handling authentication error during token validation."""
    # Set up the mock auth provider to raise an exception
    mock_auth_provider.validate_token.side_effect = AuthenticationError("Mock authentication error")

    context = await security_service.validate_token("ERROR_TOKEN")

    # Should get anonymous context
    assert context.user_id is None
    assert context.is_anonymous is True
    assert not context.permissions

    # Check logging for auth failure
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: auth_failure",
        extra={"details": {"reason": "Mock authentication error", "token_prefix": "ERROR_TOKE"}},
    )


@pytest.mark.asyncio
async def test_validate_token_unexpected_error(security_service, mock_logger_factory, mock_auth_provider):
    """Test handling unexpected error during token validation."""
    # Set up the mock auth provider to raise an unexpected exception
    mock_auth_provider.validate_token.side_effect = RuntimeError("Unexpected error")

    context = await security_service.validate_token("UNEXPECTED_TOKEN")

    # Should get anonymous context
    assert context.user_id is None
    assert context.is_anonymous is True
    assert not context.permissions

    # Check error logging
    mock_logger_factory.return_value.error.assert_called_once_with(
        "Unexpected error during token validation: Unexpected error"
    )

    # Check auth error event
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: auth_error", extra={"details": {"error": "Unexpected error"}}
    )


@pytest.mark.asyncio
async def test_generate_session_token_success(security_service, mock_logger_factory):
    """Test successful generation of a session token."""
    client_id = "test_client_123"
    token = await security_service.generate_session_token(client_id)

    assert token is not None
    assert isinstance(token, str)

    # Decode to check basic content - without verifying signature here as that's jwt lib's job
    # We trust jwt.encode works; we're testing our usage.
    payload = jwt.decode(
        token, security_service._jwt_secret_key, algorithms=["HS256"], options={"verify_signature": True}
    )
    assert payload["sub"] == client_id
    assert payload["type"] == "session"
    assert "iat" in payload
    assert "exp" in payload

    # Check logging
    generated_event_logged = False
    expected_dt_from_payload = datetime.fromtimestamp(payload["exp"], timezone.utc)
    for call in mock_logger_factory.return_value.info.call_args_list:
        if call.args and call.args[0] == "Security event: session_token_generated":
            details = call.kwargs.get("extra", {}).get("details", {})
            if details.get("client_id") == client_id:
                logged_expires_at_str = details.get("expires_at")
                assert logged_expires_at_str is not None, "expires_at not found in log details"

                logged_dt = datetime.fromisoformat(logged_expires_at_str)

                # Ensure both datetimes are comparable (e.g., both offset-aware or both naive)
                # Here, expected_dt_from_payload is UTC-aware, and fromisoformat should parse tz info.
                assert logged_dt.tzinfo is not None and expected_dt_from_payload.tzinfo is not None, (
                    "Timezone information missing from one of the datetimes"
                )

                # Compare ensuring they are within a small delta (e.g., 1 second)
                time_difference_seconds = abs((logged_dt - expected_dt_from_payload).total_seconds())
                assert time_difference_seconds < 1, (
                    f"Logged expiration time {logged_dt.isoformat()} is not close to expected {expected_dt_from_payload.isoformat()}"
                )

                generated_event_logged = True
                break
    assert generated_event_logged, "Session token generation event not logged correctly or with correct details."


@pytest.mark.asyncio
async def test_generate_session_token_error(security_service, mock_logger_factory):
    """Test error handling during session token generation."""
    client_id = "test_client_error"
    original_jwt_encode = jwt.encode
    jwt.encode = MagicMock(side_effect=Exception("JWT encoding error"))

    with pytest.raises(Exception, match="JWT encoding error"):
        await security_service.generate_session_token(client_id)

    # Check logging for failure
    mock_logger_factory.return_value.error.assert_any_call(
        f"Error generating session token for client {client_id}: JWT encoding error"
    )
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: session_token_generation_failed",
        extra={"details": {"client_id": client_id, "error": "JWT encoding error"}},
    )
    jwt.encode = original_jwt_encode  # Restore original function


@pytest.mark.asyncio
async def test_validate_session_token_valid(security_service, mock_logger_factory):
    """Test validation of a valid session token."""
    client_id = "test_client_valid"
    token = await security_service.generate_session_token(client_id)

    validated_client_id = await security_service.validate_session_token(token)
    assert validated_client_id == client_id

    # Check logging
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: session_token_validated",
        extra={"details": {"client_id": client_id, "token_prefix": token[:10]}},
    )


@pytest.mark.asyncio
async def test_validate_session_token_expired(security_service, mock_logger_factory):
    """Test validation of an expired session token."""
    client_id = "test_client_expired"
    # Generate a token with very short expiry for testing
    short_lived_service = SecurityService(
        mock_logger_factory,
        security_service._auth_provider,
        jwt_secret_key=security_service._jwt_secret_key,
        jwt_expire_minutes=-1,  # Expired in the past
    )
    expired_token = await short_lived_service.generate_session_token(client_id)

    validated_client_id = await security_service.validate_session_token(expired_token)
    assert validated_client_id is None

    # Check logging
    mock_logger_factory.return_value.warning.assert_any_call(f"Expired session token received: {expired_token[:10]}...")
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: session_token_validation_failed",
        extra={"details": {"reason": "expired_token", "token_prefix": expired_token[:10]}},
    )


@pytest.mark.asyncio
async def test_validate_session_token_invalid_signature(security_service, mock_logger_factory):
    """Test validation of a session token with an invalid signature."""
    client_id = "test_client_tampered"
    token_payload = {
        "sub": client_id,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
        "type": "session",
    }
    # Encode with a different key to make signature invalid
    invalid_token = jwt.encode(token_payload, "wrong-secret-key", algorithm="HS256")

    validated_client_id = await security_service.validate_session_token(invalid_token)
    assert validated_client_id is None

    # Check logging
    mock_logger_factory.return_value.warning.assert_any_call(
        f"Invalid session token received: {invalid_token[:10]}... Error: Signature verification failed"
    )
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: session_token_validation_failed",
        extra={
            "details": {"reason": "invalid_token: Signature verification failed", "token_prefix": invalid_token[:10]}
        },
    )


@pytest.mark.asyncio
async def test_validate_session_token_malformed_no_sub(security_service, mock_logger_factory):
    """Test validation of a malformed session token (missing 'sub' claim)."""
    token_payload = {
        # "sub": "test_client_malformed", # Missing sub
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
        "type": "session",
    }
    malformed_token = jwt.encode(token_payload, security_service._jwt_secret_key, algorithm="HS256")

    validated_client_id = await security_service.validate_session_token(malformed_token)
    assert validated_client_id is None

    # Check logging
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: session_token_validation_failed",
        extra={"details": {"reason": "invalid_or_missing_client_id_in_payload", "token_prefix": malformed_token[:10]}},
    )


@pytest.mark.asyncio
async def test_validate_session_token_malformed_wrong_type(security_service, mock_logger_factory):
    """Test validation of a malformed session token (wrong 'type' claim)."""
    token_payload = {
        "sub": "test_client_malformed_type",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
        "type": "not_session_type",  # Wrong type
    }
    malformed_token = jwt.encode(token_payload, security_service._jwt_secret_key, algorithm="HS256")

    validated_client_id = await security_service.validate_session_token(malformed_token)
    assert validated_client_id is None

    # Check logging - match the new more descriptive error message format
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: session_token_validation_failed",
        extra={
            "details": {
                "reason": "invalid_token_type_in_payload",
                "token_prefix": malformed_token[:10],
                "expected_type": "session",
                "actual_type": "not_session_type",
            }
        },
    )


@pytest.mark.asyncio
async def test_validate_session_token_completely_invalid_token_string(security_service, mock_logger_factory):
    """Test validation with a completely invalid token string."""
    invalid_token_string = "this.is.not.a.jwt"
    validated_client_id = await security_service.validate_session_token(invalid_token_string)
    assert validated_client_id is None

    # Check logging - use flexible error message checking since PyJWT error messages can vary
    warning_calls = [call.args[0] for call in mock_logger_factory.return_value.warning.call_args_list if call.args]
    token_validation_warning_found = any(
        call.startswith(f"Invalid session token received: {invalid_token_string[:10]}... Error:")
        for call in warning_calls
    )
    assert token_validation_warning_found, (
        f"Expected warning about invalid token not found. Actual warnings: {warning_calls}"
    )

    # Check info event - also make this flexible
    info_calls = mock_logger_factory.return_value.info.call_args_list
    token_validation_event_found = any(
        call.args
        and call.args[0] == "Security event: session_token_validation_failed"
        and call.kwargs.get("extra", {}).get("details", {}).get("token_prefix") == invalid_token_string[:10]
        and "invalid_token:" in call.kwargs.get("extra", {}).get("details", {}).get("reason", "")
        for call in info_calls
    )
    assert token_validation_event_found, (
        f"Expected session token validation failed event not found. Actual info calls: {info_calls}"
    )


@pytest.mark.asyncio
async def test_validate_session_token_unexpected_error_during_decode(security_service, mock_logger_factory):
    """Test handling of unexpected errors during token decoding."""
    token = "valid_looking_token_that_will_cause_decode_error"

    original_jwt_decode = jwt.decode
    jwt.decode = MagicMock(side_effect=Exception("Unexpected decode error"))

    validated_client_id = await security_service.validate_session_token(token)
    assert validated_client_id is None

    # Check logging
    mock_logger_factory.return_value.error.assert_any_call(
        f"Unexpected error validating session token {token[:10]}...: Unexpected decode error"
    )
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: session_token_validation_error",
        extra={"details": {"error": "Unexpected decode error", "token_prefix": token[:10]}},
    )
    jwt.decode = original_jwt_decode  # Restore


@pytest.mark.asyncio
async def test_generate_and_validate_session_token_flow(security_service, mock_logger_factory):
    """Test the full flow of generating and then validating a session token."""
    client_id = "client_flow_test"

    # Generate token
    token = await security_service.generate_session_token(client_id)
    assert token is not None

    # Validate token
    validated_client_id = await security_service.validate_session_token(token)
    assert validated_client_id == client_id

    # Check logging calls (simplified check, more detailed checks are in individual tests)
    # Check generation log
    generation_logged = any(
        call.args[0] == "Security event: session_token_generated"
        and call.kwargs.get("extra", {}).get("details", {}).get("client_id") == client_id
        for call in mock_logger_factory.return_value.info.call_args_list
    )
    assert generation_logged

    # Check validation log
    validation_logged = any(
        call.args[0] == "Security event: session_token_validated"
        and call.kwargs.get("extra", {}).get("details", {}).get("client_id") == client_id
        for call in mock_logger_factory.return_value.info.call_args_list
    )
    assert validation_logged
