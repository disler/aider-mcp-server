"""Test that working directory is properly passed to handlers in SSE mode."""

import json
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from aider_mcp_server.handlers import process_aider_ai_code_request
from aider_mcp_server.security import SecurityContext
from aider_mcp_server.sse_server import run_sse_server
from aider_mcp_server.sse_transport_adapter import SSETransportAdapter
from aider_mcp_server.transport_coordinator import ApplicationCoordinator


@pytest.mark.asyncio
async def test_sse_adapter_stores_config():
    """Test that SSE adapter stores configuration correctly."""
    # Create mock coordinator
    mock_coordinator = MagicMock(spec=ApplicationCoordinator)

    # Create SSE adapter with configuration
    editor_model = "test-model"
    current_working_dir = "/test/working/dir"

    adapter = SSETransportAdapter(
        coordinator=mock_coordinator,
        host="127.0.0.1",
        port=8765,
        editor_model=editor_model,
        current_working_dir=current_working_dir,
    )

    # Verify configuration is stored
    assert adapter._editor_model == editor_model
    assert adapter._current_working_dir == current_working_dir


@pytest.mark.asyncio
async def test_sse_server_passes_config_to_adapter(free_port):
    """Test that SSE server passes configuration to SSE adapter."""
    editor_model = "test-model"
    current_working_dir = str(Path(__file__).parent.parent)  # Use actual project dir

    # Mock is_git_repository to return True
    with patch("aider_mcp_server.sse_server.is_git_repository") as mock_is_git:
        mock_is_git.return_value = (True, None)

        # Mock the ApplicationCoordinator
        with patch("aider_mcp_server.sse_server.ApplicationCoordinator") as mock_coordinator_class:
            # Create async context manager mock
            mock_coordinator = AsyncMock(spec=ApplicationCoordinator)
            mock_coordinator.__aenter__ = AsyncMock(return_value=mock_coordinator)
            mock_coordinator.__aexit__ = AsyncMock(return_value=None)

            # getInstance returns the coordinator
            mock_coordinator_class.getInstance = AsyncMock(return_value=mock_coordinator)

            # Mock the SSETransportAdapter
            with patch("aider_mcp_server.sse_server.SSETransportAdapter") as mock_adapter_class:
                mock_adapter = AsyncMock(spec=SSETransportAdapter)
                mock_adapter_class.return_value = mock_adapter

                # Mock the adapter initialization
                mock_adapter.initialize = AsyncMock()
                mock_adapter.start_listening = AsyncMock()

                # Mock asyncio.Event for shutdown handling
                with patch("asyncio.Event") as mock_event_class:
                    mock_event = AsyncMock()
                    mock_event_class.return_value = mock_event
                    mock_event.wait = AsyncMock()
                    mock_event.is_set = MagicMock(return_value=True)  # Shutdown immediately

                    # Run the server
                    await run_sse_server(
                        host="127.0.0.1",
                        port=free_port,
                        editor_model=editor_model,
                        current_working_dir=current_working_dir,
                    )

                    # Verify SSETransportAdapter was created with the right params
                    mock_adapter_class.assert_called_once_with(
                        coordinator=mock_coordinator,
                        host="127.0.0.1",
                        port=free_port,
                        get_logger=ANY,
                        editor_model=editor_model,
                        current_working_dir=current_working_dir,
                    )


@pytest.mark.asyncio
async def test_aider_handler_receives_working_dir():
    """Test that the aider handler actually receives and uses working_dir."""
    with patch("aider_mcp_server.handlers.code_with_aider") as mock_code_with_aider:
        # Set up mock response
        mock_response = {"success": True, "diff": "test diff", "is_cached_diff": False}
        mock_code_with_aider.return_value = json.dumps(mock_response)

        # Create request
        request_id = "test-request"
        transport_id = "sse"
        params = {
            "ai_coding_prompt": "test prompt",
            "relative_editable_files": ["test.py"],
            "relative_readonly_files": [],
        }
        security_context = SecurityContext(
            user_id=None, permissions=set(), is_anonymous=True, transport_id=transport_id
        )

        # Call handler with working dir
        test_working_dir = "/test/working/dir"
        result = await process_aider_ai_code_request(
            request_id=request_id,
            transport_id=transport_id,
            params=params,
            security_context=security_context,
            editor_model="test-model",
            current_working_dir=test_working_dir,
        )

        # Verify code_with_aider was called with working_dir
        mock_code_with_aider.assert_called_once()
        call_args = mock_code_with_aider.call_args

        # Check that working_dir was passed
        assert call_args.kwargs["working_dir"] == test_working_dir

        # Verify result
        assert result["success"] is True
        assert result["diff"] == "test diff"
