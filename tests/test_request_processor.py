from unittest.mock import AsyncMock, MagicMock

import pytest

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.mcp_types import OperationResult
from aider_mcp_server.request_processor import RequestProcessor
from aider_mcp_server.security import Permissions


@pytest.fixture
def request_processor():
    event_coordinator = MagicMock()
    # Configure the send_event_to_transport method to be awaitable
    event_coordinator.send_event_to_transport = AsyncMock()
    event_coordinator.broadcast_event = AsyncMock()

    session_manager = MagicMock()
    # Configure the check_permission method to be awaitable
    session_manager.check_permission = AsyncMock(return_value=True)

    logger_factory = MagicMock(return_value=MagicMock())
    return RequestProcessor(event_coordinator, session_manager, logger_factory)


@pytest.mark.asyncio
async def test_register_handler(request_processor):
    # Test registering a handler
    async def test_handler(
        request_id: str, transport_id: str, request_data: dict
    ) -> OperationResult:
        return {
            "success": True,
            "request_id": request_id,
            "transport_id": transport_id,
            "result": {"message": "Test Result"},
        }

    await request_processor.register_handler(
        "test_operation", test_handler, Permissions.EXECUTE_AIDER
    )

    assert "test_operation" in request_processor._handlers
    assert request_processor._handlers["test_operation"][0] == test_handler
    assert request_processor._handlers["test_operation"][1] == Permissions.EXECUTE_AIDER


@pytest.mark.asyncio
async def test_process_request(request_processor):
    # Test processing a request
    async def test_handler(
        request_id: str, transport_id: str, request_data: dict
    ) -> OperationResult:
        return {
            "success": True,
            "request_id": request_id,
            "transport_id": transport_id,
            "result": {"message": "Test Result"},
        }

    await request_processor.register_handler(
        "test_operation", test_handler, Permissions.EXECUTE_AIDER
    )

    request_data = {"key": "value"}
    await request_processor.process_request(
        "123", "transport_id", "test_operation", request_data
    )

    assert "123" in request_processor._active_requests
    assert (
        request_processor._active_requests["123"]["operation_name"] == "test_operation"
    )
    assert request_processor._active_requests["123"]["status"] == "processing"
    assert request_processor._active_requests["123"]["details"] == request_data


@pytest.mark.asyncio
async def test_fail_request(request_processor):
    # Test failing a request
    async def test_handler(
        request_id: str, transport_id: str, request_data: dict
    ) -> OperationResult:
        raise Exception("Test Error")

    await request_processor.register_handler(
        "test_operation", test_handler, Permissions.EXECUTE_AIDER
    )

    # Directly call fail_request since process_request will run in the background
    await request_processor.fail_request(
        "123", "test_operation", "Error Message", "Error Details", "transport_id"
    )

    # Check that send_event_to_transport was called with appropriate error info
    request_processor._event_coordinator.send_event_to_transport.assert_any_call(
        "transport_id",
        EventTypes.TOOL_RESULT,
        {
            "request_id": "123",
            "tool_name": "test_operation",
            "result": {
                "success": False,
                "request_id": "123",
                "transport_id": "transport_id",
                "error": {"message": "Error Details", "code": "Error Message"},
            },
        },
    )


@pytest.mark.asyncio
async def test_cleanup_request(request_processor):
    # Test cleaning up a request
    request_data = {"key": "value"}
    await request_processor.process_request(
        "123", "transport_id", "test_operation", request_data
    )

    await request_processor._cleanup_request("123")

    assert "123" not in request_processor._active_requests
