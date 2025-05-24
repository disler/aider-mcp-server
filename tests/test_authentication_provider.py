"""
Tests for authentication provider interface and implementations.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from aider_mcp_server.atoms.security.errors import (
    InvalidCredentialsError,
)
from aider_mcp_server.molecules.security.auth_provider import DefaultAuthenticationProvider
from aider_mcp_server.interfaces.authentication_provider import (
    AuthToken,
    IAuthenticationProvider,
    UserInfo,
)
from aider_mcp_server.atoms.security.context import Permissions


@pytest.fixture
def logger_factory():
    """Create a logger factory for testing."""
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
def default_auth_provider(logger_factory):
    """Create a default authentication provider for testing."""
    return DefaultAuthenticationProvider(logger_factory)


@pytest.mark.asyncio
async def test_auth_token_initialization():
    """Test AuthToken initialization and methods."""
    token = AuthToken(
        token="test-token",
        user_id="test-user",
        issued_at=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=1),
        permissions={Permissions.EXECUTE_AIDER},
    )

    assert token.token == "test-token"
    assert token.user_id == "test-user"
    assert not token.is_expired()
    assert token.has_permission(Permissions.EXECUTE_AIDER)
    assert not token.has_permission(Permissions.VIEW_CONFIG)


@pytest.mark.asyncio
async def test_auth_token_expiration():
    """Test AuthToken expiration checking."""
    # Test expired token
    token = AuthToken(
        token="expired-token",
        user_id="test-user",
        issued_at=datetime.now() - timedelta(hours=2),
        expires_at=datetime.now() - timedelta(hours=1),
    )
    assert token.is_expired()

    # Test non-expiring token
    token = AuthToken(
        token="non-expiring",
        user_id="test-user",
        issued_at=datetime.now(),
        expires_at=None,
    )
    assert not token.is_expired()


@pytest.mark.asyncio
async def test_user_info_initialization():
    """Test UserInfo initialization."""
    user_info = UserInfo(
        user_id="test-user",
        username="testuser",
        email="test@example.com",
        permissions={Permissions.EXECUTE_AIDER, Permissions.LIST_MODELS},
        metadata={"role": "developer"},
    )

    assert user_info.user_id == "test-user"
    assert user_info.username == "testuser"
    assert user_info.email == "test@example.com"
    assert len(user_info.permissions) == 2
    assert user_info.metadata["role"] == "developer"


@pytest.mark.asyncio
async def test_user_info_defaults():
    """Test UserInfo default values."""
    user_info = UserInfo(user_id="minimal-user")
    assert user_info.username is None
    assert user_info.email is None
    assert user_info.permissions == set()
    assert user_info.metadata == {}


@pytest.mark.asyncio
async def test_default_auth_provider_authenticate_success(default_auth_provider):
    """Test successful authentication with default provider."""
    credentials = {"api_key": "test-api-key"}
    auth_token = await default_auth_provider.authenticate(credentials)

    assert auth_token is not None
    assert auth_token.user_id == "test-user"
    assert Permissions.EXECUTE_AIDER in auth_token.permissions
    assert not auth_token.is_expired()


@pytest.mark.asyncio
async def test_default_auth_provider_authenticate_admin(default_auth_provider):
    """Test admin authentication with default provider."""
    credentials = {"api_key": "admin-api-key"}
    auth_token = await default_auth_provider.authenticate(credentials)

    assert auth_token.user_id == "admin-user"
    assert Permissions.EXECUTE_AIDER in auth_token.permissions
    assert Permissions.LIST_MODELS in auth_token.permissions
    assert Permissions.VIEW_CONFIG in auth_token.permissions


@pytest.mark.asyncio
async def test_default_auth_provider_authenticate_invalid_key(default_auth_provider):
    """Test authentication with invalid API key."""
    credentials = {"api_key": "invalid-key"}
    with pytest.raises(InvalidCredentialsError, match="Invalid API key"):
        await default_auth_provider.authenticate(credentials)


@pytest.mark.asyncio
async def test_default_auth_provider_authenticate_missing_key(default_auth_provider):
    """Test authentication with missing API key."""
    credentials = {}
    with pytest.raises(InvalidCredentialsError, match="API key is required"):
        await default_auth_provider.authenticate(credentials)


@pytest.mark.asyncio
async def test_default_auth_provider_validate_token(default_auth_provider):
    """Test token validation."""
    # First authenticate to get a token
    credentials = {"api_key": "test-api-key"}
    auth_token = await default_auth_provider.authenticate(credentials)

    # Validate the token
    is_valid = await default_auth_provider.validate_token(auth_token.token)
    assert is_valid

    # Test invalid token
    is_valid = await default_auth_provider.validate_token("invalid-token")
    assert not is_valid


@pytest.mark.asyncio
async def test_default_auth_provider_get_user_info(default_auth_provider):
    """Test getting user info from token."""
    # First authenticate to get a token
    credentials = {"api_key": "test-api-key"}
    auth_token = await default_auth_provider.authenticate(credentials)

    # Get user info
    user_info = await default_auth_provider.get_user_info(auth_token.token)
    assert user_info is not None
    assert user_info.user_id == auth_token.user_id
    assert user_info.permissions == auth_token.permissions

    # Test invalid token
    user_info = await default_auth_provider.get_user_info("invalid-token")
    assert user_info is None


@pytest.mark.asyncio
async def test_default_auth_provider_revoke_token(default_auth_provider):
    """Test token revocation."""
    # First authenticate to get a token
    credentials = {"api_key": "test-api-key"}
    auth_token = await default_auth_provider.authenticate(credentials)

    # Revoke the token
    revoked = await default_auth_provider.revoke_token(auth_token.token)
    assert revoked

    # Token should now be invalid
    is_valid = await default_auth_provider.validate_token(auth_token.token)
    assert not is_valid

    # Try to revoke non-existent token
    revoked = await default_auth_provider.revoke_token("non-existent-token")
    assert not revoked


@pytest.mark.asyncio
async def test_default_auth_provider_refresh_token(default_auth_provider):
    """Test token refresh."""
    # First authenticate to get a token
    credentials = {"api_key": "test-api-key"}
    auth_token = await default_auth_provider.authenticate(credentials)
    old_token = auth_token.token

    # Refresh the token
    new_auth_token = await default_auth_provider.refresh_token(auth_token.token)
    assert new_auth_token is not None
    assert new_auth_token.token != old_token
    assert new_auth_token.user_id == auth_token.user_id
    assert new_auth_token.permissions == auth_token.permissions

    # Old token should be invalid
    is_valid = await default_auth_provider.validate_token(old_token)
    assert not is_valid

    # New token should be valid
    is_valid = await default_auth_provider.validate_token(new_auth_token.token)
    assert is_valid

    # Try to refresh invalid token
    new_auth_token = await default_auth_provider.refresh_token("invalid-token")
    assert new_auth_token is None


@pytest.mark.asyncio
async def test_auth_provider_interface_compliance(default_auth_provider):
    """Test that DefaultAuthenticationProvider implements IAuthenticationProvider interface."""
    assert isinstance(default_auth_provider, IAuthenticationProvider)

    # Check all abstract methods are implemented
    assert hasattr(default_auth_provider, "authenticate")
    assert hasattr(default_auth_provider, "validate_token")
    assert hasattr(default_auth_provider, "get_user_info")
    assert hasattr(default_auth_provider, "revoke_token")
    assert hasattr(default_auth_provider, "refresh_token")


class MockAuthenticationProvider(IAuthenticationProvider):
    """Mock authentication provider for testing interface compliance."""

    def __init__(self):
        self.authenticate_called = False
        self.validate_token_called = False
        self.get_user_info_called = False
        self.revoke_token_called = False
        self.refresh_token_called = False

    async def authenticate(self, credentials):
        self.authenticate_called = True
        return AuthToken(
            token="mock-token",
            user_id="mock-user",
            issued_at=datetime.now(),
            permissions=set(),
        )

    async def validate_token(self, token):
        self.validate_token_called = True
        return True

    async def get_user_info(self, token):
        self.get_user_info_called = True
        return UserInfo(user_id="mock-user")

    async def revoke_token(self, token):
        self.revoke_token_called = True
        return True

    async def refresh_token(self, token):
        self.refresh_token_called = True
        return AuthToken(
            token="new-mock-token",
            user_id="mock-user",
            issued_at=datetime.now(),
            permissions=set(),
        )


@pytest.mark.asyncio
async def test_mock_auth_provider_interface():
    """Test mock authentication provider interface usage."""
    mock_provider = MockAuthenticationProvider()

    # Test all interface methods
    auth_token = await mock_provider.authenticate({"username": "test"})
    assert mock_provider.authenticate_called
    assert auth_token.user_id == "mock-user"

    is_valid = await mock_provider.validate_token("some-token")
    assert mock_provider.validate_token_called
    assert is_valid

    user_info = await mock_provider.get_user_info("some-token")
    assert mock_provider.get_user_info_called
    assert user_info.user_id == "mock-user"

    revoked = await mock_provider.revoke_token("some-token")
    assert mock_provider.revoke_token_called
    assert revoked

    new_token = await mock_provider.refresh_token("some-token")
    assert mock_provider.refresh_token_called
    assert new_token.token == "new-mock-token"
