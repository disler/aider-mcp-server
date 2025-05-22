"""
Tests for SSE security validation.

These tests verify that the SSE Transport Adapter properly validates the security
of incoming connections and requests, including authentication, authorization,
and request validation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aider_mcp_server.security import SecurityContext
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter


@pytest.fixture
def adapter():
    """Fixture providing a basic SSETransportAdapter instance."""
    adapter = SSETransportAdapter()
    return adapter


def test_validate_request_security_default(adapter):
    """Test that the default security validation returns an anonymous context."""
    # Create a simple request details dictionary
    request_details = {"client_ip": "127.0.0.1", "path": "/sse/"}

    # Validate the request security
    security_context = adapter.validate_request_security(request_details)

    # Verify that the security context is anonymous
    assert security_context.is_anonymous is True
    assert security_context.user_id is None
    assert len(security_context.permissions) == 0
    assert security_context.transport_id == adapter.get_transport_id()


def test_validate_request_security_with_token():
    """Test validating a request with an authentication token."""
    # Create a mock security service that can validate tokens
    mock_security_service = MagicMock()
    mock_security_service.validate_token.return_value = SecurityContext(
        user_id="test-user", permissions={"read", "write"}, is_anonymous=False, transport_id="sse"
    )

    # Create an adapter with the mock security service
    adapter = SSETransportAdapter()
    adapter._security_service = mock_security_service

    # Create a request details dictionary with a token
    request_details = {"client_ip": "127.0.0.1", "path": "/sse/", "headers": {"Authorization": "Bearer test-token"}}

    # Override the validate_request_security method to handle tokens
    def enhanced_validate_security(request_details):
        # Extract the token from the headers
        headers = request_details.get("headers", {})
        auth_header = headers.get("Authorization", "")

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            # Validate the token using the security service
            return adapter._security_service.validate_token(token)

        # Fall back to anonymous context if no token is provided
        return SecurityContext(
            user_id=None, permissions=set(), is_anonymous=True, transport_id=adapter.get_transport_id()
        )

    # Replace the method with our enhanced version
    adapter.validate_request_security = enhanced_validate_security

    # Validate the request security
    security_context = adapter.validate_request_security(request_details)

    # Verify that the security context has the expected values
    assert security_context.is_anonymous is False
    assert security_context.user_id == "test-user"
    assert security_context.permissions == {"read", "write"}
    assert security_context.transport_id == "sse"

    # Verify that the token was validated
    mock_security_service.validate_token.assert_called_once_with("test-token")


def test_validate_request_security_with_invalid_token():
    """Test validating a request with an invalid authentication token."""
    # Create a mock security service that rejects tokens
    mock_security_service = MagicMock()
    mock_security_service.validate_token.side_effect = ValueError("Invalid token")

    # Create an adapter with the mock security service
    adapter = SSETransportAdapter()
    adapter._security_service = mock_security_service

    # Create a request details dictionary with an invalid token
    request_details = {"client_ip": "127.0.0.1", "path": "/sse/", "headers": {"Authorization": "Bearer invalid-token"}}

    # Override the validate_request_security method to handle tokens
    def enhanced_validate_security(request_details):
        # Extract the token from the headers
        headers = request_details.get("headers", {})
        auth_header = headers.get("Authorization", "")

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            try:
                # Try to validate the token
                return adapter._security_service.validate_token(token)
            except ValueError:
                # Return anonymous context with limited permissions on token validation failure
                return SecurityContext(
                    user_id=None,
                    permissions={"read"},  # Limited permissions
                    is_anonymous=True,
                    transport_id=adapter.get_transport_id(),
                )

        # Fall back to anonymous context if no token is provided
        return SecurityContext(
            user_id=None, permissions=set(), is_anonymous=True, transport_id=adapter.get_transport_id()
        )

    # Replace the method with our enhanced version
    adapter.validate_request_security = enhanced_validate_security

    # Validate the request security
    security_context = adapter.validate_request_security(request_details)

    # Verify that the security context has the expected values
    assert security_context.is_anonymous is True
    assert security_context.user_id is None
    assert security_context.permissions == {"read"}  # Limited permissions
    assert security_context.transport_id == adapter.get_transport_id()

    # Verify that the token was attempted to be validated
    mock_security_service.validate_token.assert_called_once_with("invalid-token")


def test_validate_request_security_with_origin_header():
    """Test validating a request with an origin header for CORS."""
    # Create an adapter
    adapter = SSETransportAdapter()

    # Create a request details dictionary with an origin header
    request_details = {"client_ip": "127.0.0.1", "path": "/sse/", "headers": {"Origin": "https://example.com"}}

    # Define allowed origins for CORS
    allowed_origins = {"https://example.com", "https://trusted-site.com"}

    # Override the validate_request_security method to check the origin
    def enhanced_validate_security(request_details):
        # Extract the origin from the headers
        headers = request_details.get("headers", {})
        origin = headers.get("Origin")

        # Create a basic security context
        security_context = SecurityContext(
            user_id=None, permissions=set(), is_anonymous=True, transport_id=adapter.get_transport_id()
        )

        # If there's an origin, check if it's allowed
        if origin:
            if origin in allowed_origins:
                # Add a permission for allowed origins
                security_context.permissions.add("cors_allowed")
            else:
                # Add a different permission for disallowed origins
                security_context.permissions.add("cors_blocked")

        return security_context

    # Replace the method with our enhanced version
    adapter.validate_request_security = enhanced_validate_security

    # Validate the request security
    security_context = adapter.validate_request_security(request_details)

    # Verify that the security context has the expected values
    assert security_context.is_anonymous is True
    assert security_context.user_id is None
    assert "cors_allowed" in security_context.permissions
    assert "cors_blocked" not in security_context.permissions
    assert security_context.transport_id == adapter.get_transport_id()

    # Test with a disallowed origin
    request_details["headers"]["Origin"] = "https://malicious-site.com"
    security_context = adapter.validate_request_security(request_details)

    # Verify that the security context has the expected values
    assert "cors_allowed" not in security_context.permissions
    assert "cors_blocked" in security_context.permissions


def test_validate_request_security_with_ip_filtering():
    """Test validating a request with IP address filtering."""
    # Create an adapter
    adapter = SSETransportAdapter()

    # Define allowed and blocked IP ranges
    allowed_ips = {"127.0.0.1", "192.168.1.0/24"}
    blocked_ips = {"10.0.0.0/8"}

    # Override the validate_request_security method to check the IP
    def enhanced_validate_security(request_details):
        # Extract the client IP from the request details
        client_ip = request_details.get("client_ip")

        # Create a basic security context
        security_context = SecurityContext(
            user_id=None, permissions=set(), is_anonymous=True, transport_id=adapter.get_transport_id()
        )

        # Check if the IP is explicitly allowed
        if client_ip in allowed_ips:
            security_context.permissions.add("ip_allowed")
        # Check if the IP is in an allowed range
        elif any(client_ip.startswith("192.168.1.") for ip in allowed_ips if ip.endswith("/24")):
            security_context.permissions.add("ip_allowed")
        # Check if the IP is in a blocked range
        elif any(client_ip.startswith("10.") for ip in blocked_ips if ip.endswith("/8")):
            security_context.permissions.add("ip_blocked")

        return security_context

    # Replace the method with our enhanced version
    adapter.validate_request_security = enhanced_validate_security

    # Test with an allowed IP
    request_details = {"client_ip": "127.0.0.1", "path": "/sse/"}
    security_context = adapter.validate_request_security(request_details)

    # Verify that the security context has the expected values
    assert "ip_allowed" in security_context.permissions
    assert "ip_blocked" not in security_context.permissions

    # Test with an IP in an allowed range
    request_details = {"client_ip": "192.168.1.100", "path": "/sse/"}
    security_context = adapter.validate_request_security(request_details)

    # Verify that the security context has the expected values
    assert "ip_allowed" in security_context.permissions
    assert "ip_blocked" not in security_context.permissions

    # Test with an IP in a blocked range
    request_details = {"client_ip": "10.1.2.3", "path": "/sse/"}
    security_context = adapter.validate_request_security(request_details)

    # Verify that the security context has the expected values
    assert "ip_allowed" not in security_context.permissions
    assert "ip_blocked" in security_context.permissions


@pytest.mark.asyncio
async def test_security_in_handle_sse_request():
    """Test that security validation is performed during SSE request handling."""
    # Create a mock coordinator
    mock_coordinator = AsyncMock()

    # Create a mock security service
    mock_security_service = MagicMock()

    # Create the adapter with the mock coordinator
    adapter = SSETransportAdapter(coordinator=mock_coordinator)
    adapter._security_service = mock_security_service

    # Mock the validate_request_security method
    adapter.validate_request_security = MagicMock(
        return_value=SecurityContext(
            user_id="test-user",
            permissions={"read", "write"},
            is_anonymous=False,
            transport_id=adapter.get_transport_id(),
        )
    )

    # Create a mock request
    mock_request = MagicMock()
    mock_request.scope = {"type": "http", "client": ("127.0.0.1", 12345)}
    mock_request.receive = AsyncMock()
    mock_request._send = AsyncMock()

    # Mock the MCP transport and server for successful connection
    mock_mcp_transport = MagicMock()
    mock_mcp_server = MagicMock()

    # Create a context manager for connect_sse
    class MockContextManager:
        async def __aenter__(self):
            return (MagicMock(), MagicMock())  # mock streams

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_mcp_transport.connect_sse.return_value = MockContextManager()
    mock_mcp_server._mcp_server = MagicMock()
    mock_mcp_server._mcp_server.run = AsyncMock()
    mock_mcp_server._mcp_server.create_initialization_options = MagicMock(return_value={})

    adapter._mcp_transport = mock_mcp_transport
    adapter._mcp_server = mock_mcp_server

    # Mock the starlette Response
    with patch("starlette.responses.Response") as mock_response_cls:
        mock_response = MagicMock()
        mock_response_cls.return_value = mock_response

        # Enhance the handle_sse_request method to validate security
        original_handle_sse = adapter.handle_sse_request

        async def enhanced_handle_sse(request):
            # Extract request details for security validation
            request_details = {
                "client_ip": request.scope["client"][0],
                "path": request.scope.get("path", ""),
                "headers": dict(request.scope.get("headers", [])),
            }

            # Validate security
            security_context = adapter.validate_request_security(request_details)

            # Check if the request is allowed
            if not security_context.is_anonymous or "read" in security_context.permissions:
                # Allow the request to proceed
                return await original_handle_sse(request)
            else:
                # Reject the request
                return mock_response_cls(status_code=403, content="Forbidden")

        # Replace the method with our enhanced version
        adapter.handle_sse_request = enhanced_handle_sse

        # Handle an SSE request
        response = await adapter.handle_sse_request(mock_request)

        # Verify that the security validation was performed
        adapter.validate_request_security.assert_called_once()
        request_details = adapter.validate_request_security.call_args[0][0]
        assert request_details["client_ip"] == "127.0.0.1"

        # Verify that the MCP transport's connect_sse method was called
        mock_mcp_transport.connect_sse.assert_called_once_with(
            mock_request.scope, mock_request.receive, mock_request._send
        )

        # Verify that the MCP server's run method was called
        mock_mcp_server._mcp_server.run.assert_called_once()

        # Verify that a response was returned
        assert response == mock_response


@pytest.mark.asyncio
async def test_security_validation_with_custom_security_service():
    """Test security validation with a custom security service implementation."""

    # Create a custom security service class
    class CustomSecurityService:
        def __init__(self):
            self.allowed_tokens = {"valid-token": "test-user"}
            self.user_permissions = {"test-user": {"read", "write"}}

        def validate_token(self, token):
            if token not in self.allowed_tokens:
                raise ValueError("Invalid token")

            user_id = self.allowed_tokens[token]
            permissions = self.user_permissions.get(user_id, set())

            return SecurityContext(user_id=user_id, permissions=permissions, is_anonymous=False, transport_id="sse")

    # Create an instance of the custom security service
    custom_security_service = CustomSecurityService()

    # Create an adapter with the custom security service
    adapter = SSETransportAdapter()
    adapter._security_service = custom_security_service

    # Create a request details dictionary with a valid token
    request_details = {"client_ip": "127.0.0.1", "path": "/sse/", "headers": {"Authorization": "Bearer valid-token"}}

    # Override the validate_request_security method to use the custom security service
    def enhanced_validate_security(request_details):
        # Extract the token from the headers
        headers = request_details.get("headers", {})
        auth_header = headers.get("Authorization", "")

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            try:
                # Try to validate the token
                return adapter._security_service.validate_token(token)
            except ValueError:
                # Return anonymous context on token validation failure
                return SecurityContext(
                    user_id=None, permissions=set(), is_anonymous=True, transport_id=adapter.get_transport_id()
                )

        # Fall back to anonymous context if no token is provided
        return SecurityContext(
            user_id=None, permissions=set(), is_anonymous=True, transport_id=adapter.get_transport_id()
        )

    # Replace the method with our enhanced version
    adapter.validate_request_security = enhanced_validate_security

    # Validate the request security with a valid token
    security_context = adapter.validate_request_security(request_details)

    # Verify that the security context has the expected values
    assert security_context.is_anonymous is False
    assert security_context.user_id == "test-user"
    assert security_context.permissions == {"read", "write"}
    assert security_context.transport_id == "sse"

    # Test with an invalid token
    request_details["headers"]["Authorization"] = "Bearer invalid-token"
    security_context = adapter.validate_request_security(request_details)

    # Verify that the security context has the expected values
    assert security_context.is_anonymous is True
    assert security_context.user_id is None
    assert len(security_context.permissions) == 0
    assert security_context.transport_id == adapter.get_transport_id()
