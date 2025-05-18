import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.mcp_types import OperationResult
from aider_mcp_server.request_processor import RequestProcessor
from aider_mcp_server.security import Permissions


@pytest.fixture
def security_service():
    service = MagicMock()
    service.check_permission = AsyncMock(return_value=True)
    service.validate_token = AsyncMock()
    service.log_security_event = AsyncMock()
    return service


@pytest.fixture
def request_processor(
    event_coordinator,
    session_manager,
    logger_factory,
    handler_registry,
    response_formatter,
    security_service,
):
    return RequestProcessor(
        event_coordinator,
        session_manager,
        logger_factory,
        handler_registry,
        response_formatter,
        security_service,
    )


@pytest.fixture
def handler_registry():
    handler_registry_mock = MagicMock()

    async def test_handler(
        request_id: str,
        transport_id: str,
        request_data: dict,
        security_context,
        use_diff_cache: bool,
        clear_cached_for_unchanged: bool,
    ) -> OperationResult:
        return {
            "success": True,
            "request_id": request_id,
            "transport_id": transport_id,
            "result": {"message": "Test Result"},
        }

    handler_registry_mock.get_handler = AsyncMock(return_value=test_handler)
    handler_registry_mock.get_required_permission = AsyncMock(return_value=Permissions.EXECUTE_AIDER)
    handler_registry_mock.register_handler = AsyncMock()

    return handler_registry_mock


@pytest.fixture
def response_formatter():
    response_formatter_mock = MagicMock()

    def format_success_response(request_id: str, transport_id: str, result: dict) -> OperationResult:
        return {
            "success": True,
            "request_id": request_id,
            "transport_id": transport_id,
            "result": result,
        }

    def format_error_response(
        request_id: str,
        transport_id: str,
        error_message: str,
        error_code=None,
        details=None,
    ) -> OperationResult:
        error_data = {
            "message": error_message,
        }

        if error_code:
            error_data["code"] = error_code

        if details:
            error_data["details"] = details

        return {
            "success": False,
            "request_id": request_id,
            "transport_id": transport_id,
            "error": error_data,
        }

    response_formatter_mock.format_success_response = MagicMock(side_effect=format_success_response)
    response_formatter_mock.format_error_response = MagicMock(side_effect=format_error_response)

    return response_formatter_mock


@pytest.fixture
def event_coordinator():
    coordinator = MagicMock()
    coordinator.send_event_to_transport = AsyncMock()
    coordinator.broadcast_event = AsyncMock()
    return coordinator


@pytest.fixture
def session_manager():
    manager = AsyncMock()
    manager.check_permission = AsyncMock(return_value=True)
    manager.get_transport_security_context = AsyncMock()

    # Mock security context
    from aider_mcp_server.security import Permissions, SecurityContext

    mock_context = SecurityContext(user_id="test-user", permissions={Permissions.EXECUTE_AIDER})
    manager.get_transport_security_context.return_value = mock_context

    return manager


@pytest.fixture
def logger_factory():
    return MagicMock()


@pytest.mark.asyncio
async def test_fail_request(request_processor, handler_registry, response_formatter):
    # Test failing a request
    async def test_handler(
        request_id: str,
        transport_id: str,
        request_data: dict,
        security_context,
        use_diff_cache: bool,
        clear_cached_for_unchanged: bool,
    ) -> OperationResult:
        raise Exception("Test Error")

    await handler_registry.register_handler("test_operation", test_handler, Permissions.EXECUTE_AIDER)

    # Set a specific return value for response_formatter.format_error_response
    response_formatter.format_error_response.return_value = {
        "success": False,
        "request_id": "123",
        "transport_id": "transport_id",
        "error": {"message": "Error Details", "code": "Error Message"},
    }

    # Directly call fail_request since process_request will run in the background
    await request_processor.fail_request("123", "test_operation", "Error Message", "Error Details", "transport_id")

    # Check that send_event_to_transport was called with appropriate error info
    request_processor._event_coordinator.send_event_to_transport.assert_any_call(
        "transport_id",
        EventTypes.STATUS,
        {
            "request_id": "123",
            "status": "failed",
            "message": "Operation test_operation failed: Error Message",
        },
    )


@pytest.mark.asyncio
async def test_register_handler(request_processor, handler_registry):
    # Test registering a handler
    async def test_handler(
        request_id: str,
        transport_id: str,
        request_data: dict,
        security_context,
        use_diff_cache: bool,
        clear_cached_for_unchanged: bool,
    ) -> OperationResult:
        return {
            "success": True,
            "request_id": request_id,
            "transport_id": transport_id,
            "result": {"message": "Test Result"},
        }

    await handler_registry.register_handler("test_operation", test_handler, Permissions.EXECUTE_AIDER)

    handler_registry.register_handler.assert_called_once_with("test_operation", test_handler, Permissions.EXECUTE_AIDER)


@pytest.mark.asyncio
async def test_process_request(request_processor, handler_registry):
    # Test processing a request
    request_data = {"parameters": {"param1": "value1", "param2": "value2"}}

    await request_processor.process_request("123", "transport_id", "test_operation", request_data)

    # Give the task a chance to complete
    await asyncio.sleep(0.01)

    handler_registry.get_handler.assert_called_once_with("test_operation")


@pytest.mark.asyncio
async def test_cleanup_request(request_processor):
    # Test cleaning up a request
    await request_processor._cleanup_request("123")

    assert "123" not in request_processor._active_requests
