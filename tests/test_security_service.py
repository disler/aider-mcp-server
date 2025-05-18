"""Tests for the security service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aider_mcp_server.interfaces.security_service import ISecurityService
from aider_mcp_server.security import Permissions, SecurityContext
from aider_mcp_server.security_service import SecurityService


@pytest.fixture
def mock_logger_factory():
    """Fixture for a mock logger factory."""
    mock_factory = MagicMock()
    mock_logger = MagicMock()
    mock_logger.info = MagicMock()
    mock_logger.warning = MagicMock()
    mock_logger.error = MagicMock()
    mock_logger.debug = MagicMock()
    mock_logger.verbose = MagicMock()
    mock_factory.return_value = mock_logger
    return mock_factory


@pytest.fixture
def security_service(mock_logger_factory):
    """Fixture for a security service instance."""
    return SecurityService(mock_logger_factory)


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
        "Security event: auth_attempt", 
        extra={"details": {"token_present": False, "token_prefix": None}}
    )
    
    # Check that anonymous auth event was logged - check all logging levels
    all_calls = (
        mock_logger_factory.return_value.debug.call_args_list +
        mock_logger_factory.return_value.info.call_args_list
    )
    found_auth_anonymous = False
    for call in all_calls:
        if len(call.args) > 0 and call.args[0].startswith("Security event auth_anonymous:"):
            found_auth_anonymous = True
            break
    assert found_auth_anonymous


@pytest.mark.asyncio
async def test_validate_token_valid_user(security_service, mock_logger_factory):
    """Test validating a valid token returns a user context with permissions."""
    # Use a token that's recognized as valid by the existing code
    context = await security_service.validate_token("VALID_TEST_TOKEN")
    
    assert context.user_id == "test-user"
    assert Permissions.EXECUTE_AIDER in context.permissions
    
    # Check logging calls - token_prefix is only 10 chars
    mock_logger_factory.return_value.info.assert_any_call(
        "Security event: auth_attempt", 
        extra={"details": {"token_present": True, "token_prefix": "VALID_TEST"}}
    )


@pytest.mark.asyncio
async def test_validate_token_admin(security_service, mock_logger_factory):
    """Test validating an admin token returns a context with admin permissions."""
    context = await security_service.validate_token("ADMIN_TOKEN")
    
    assert context.user_id == "admin"
    assert len(context.permissions) > 0
    assert Permissions.EXECUTE_AIDER in context.permissions
    assert Permissions.LIST_MODELS in context.permissions
    
    # Check logging call for auth success - the order of set items can vary
    # So let's check the actual call instead
    debug_calls = mock_logger_factory.return_value.debug.call_args_list
    found_auth_success = False
    for call in debug_calls:
        if call.args[0].startswith("Security event auth_success:"):
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
        extra={"details": {
            "user_id": "test-user", 
            "permission": "execute_aider", 
            "granted": True
        }}
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
        extra={"details": {
            "user_id": "test-user", 
            "permission": "execute_aider", 
            "granted": False
        }}
    )


@pytest.mark.asyncio
async def test_log_security_event(security_service, mock_logger_factory):
    """Test logging a security event."""
    # Log a security event
    event_type = "custom_event"
    details = {"key": "value", "another_key": 123}
    
    await security_service.log_security_event(event_type, details)
    
    # Check logging calls
    mock_logger_factory.return_value.info.assert_any_call(
        f"Security event: {event_type}", 
        extra={"details": details}
    )
    
    mock_logger_factory.return_value.info.assert_any_call(
        f"Security event {event_type}: {details}"
    )


@pytest.mark.asyncio
async def test_log_security_event_auth_failure(security_service, mock_logger_factory):
    """Test logging an auth failure event with warning level."""
    # Log an auth failure event
    event_type = "auth_failure"
    details = {"token_prefix": "INVALID_", "reason": "token_not_found"}
    
    await security_service.log_security_event(event_type, details)
    
    # Check warning log call
    mock_logger_factory.return_value.warning.assert_any_call(
        f"Authentication failure: {details}"
    )


@pytest.mark.asyncio
async def test_log_security_event_permission_denied(security_service, mock_logger_factory):
    """Test logging a permission denied event with warning level."""
    # Log a permission denied event
    event_type = "permission_denied"
    details = {"user_id": "test-user", "permission": "execute_aider"}
    
    await security_service.log_security_event(event_type, details)
    
    # Check warning log call
    mock_logger_factory.return_value.warning.assert_any_call(
        f"Permission denied: {details}"
    )