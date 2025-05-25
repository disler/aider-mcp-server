"""
Tests for RequestMonitor integration with AIDER tools.

Tests the Phase 3.1 implementation of automatic request tracking and throttling detection
during AIDER execution.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.molecules.monitoring.request_monitor import RequestMonitor
from aider_mcp_server.molecules.tools.aider_ai_code import code_with_aider


@pytest.fixture
def mock_coordinator():
    """Create a mock ApplicationCoordinator for testing."""
    coordinator = AsyncMock()
    coordinator.broadcast_event = AsyncMock()
    return coordinator


@pytest.fixture
def mock_request_monitor():
    """Create a mock RequestMonitor for testing."""
    monitor = AsyncMock(spec=RequestMonitor)
    monitor.track_request = AsyncMock(return_value="test_request_123")
    monitor.update_request_progress = AsyncMock()
    monitor.complete_request = AsyncMock()
    return monitor


class TestRequestMonitorIntegration:
    """Test suite for RequestMonitor integration with AIDER tools."""

    @pytest.mark.asyncio
    @patch("aider_mcp_server.molecules.tools.aider_ai_code.RequestMonitor")
    @patch("aider_mcp_server.molecules.tools.aider_ai_code._execute_aider_with_coordination")
    @patch("aider_mcp_server.molecules.tools.aider_ai_code._validate_working_dir_and_api_keys")
    @patch("aider_mcp_server.molecules.tools.aider_ai_code._broadcast_session_start")
    @patch("aider_mcp_server.molecules.tools.aider_ai_code._broadcast_session_completed")
    @patch("aider_mcp_server.molecules.tools.aider_ai_code._broadcast_changes_summary")
    async def test_request_monitor_lifecycle(
        self,
        mock_broadcast_changes_summary,
        mock_broadcast_session_completed,
        mock_broadcast_session_start,
        mock_validate_working_dir,
        mock_execute_aider,
        mock_request_monitor_class,
        mock_coordinator,
    ):
        """Test that RequestMonitor is properly integrated in the AIDER lifecycle."""
        # Setup mocks
        mock_validate_working_dir.return_value = None  # No validation error
        mock_monitor_instance = AsyncMock()
        mock_monitor_instance.track_request = AsyncMock(return_value="test_request_123")
        mock_monitor_instance.update_request_progress = AsyncMock()
        mock_monitor_instance.complete_request = AsyncMock()
        mock_request_monitor_class.return_value = mock_monitor_instance

        # Mock successful AIDER execution
        mock_response = {
            "success": True,
            "changes_summary": {"files": ["test.py"]},
            "rate_limit_info": None,
        }
        mock_execute_aider.return_value = mock_response

        # Mock API key check
        with patch(
            "aider_mcp_server.molecules.tools.aider_ai_code._handle_api_key_checks_and_warnings"
        ) as mock_api_check:
            mock_api_check.return_value = ({}, None)

            with patch("aider_mcp_server.molecules.tools.aider_ai_code._finalize_aider_response"):
                # Execute the function
                await code_with_aider(
                    ai_coding_prompt="Test prompt",
                    relative_editable_files=["test.py"],
                    relative_readonly_files=[],
                    model="test-model",
                    working_dir="/tmp/test",
                    coordinator=mock_coordinator,
                )

        # Verify RequestMonitor was created
        mock_request_monitor_class.assert_called_once_with(mock_coordinator)

        # Verify request tracking was initiated
        mock_monitor_instance.track_request.assert_called_once()
        call_args = mock_monitor_instance.track_request.call_args[1]
        assert "context" in call_args
        context = call_args["context"]
        assert context["model"] == "test-model"
        assert context["editable_files"] == 1
        assert context["readonly_files"] == 0
        assert context["working_dir"] == "/tmp/test"

        # Verify progress update was called
        mock_monitor_instance.update_request_progress.assert_called_once_with(
            "test_request_123",
            {
                "stage": "executing_aider",
                "model": "gemini/test-model",  # Model gets normalized
                "files_count": 1,
            },
        )

        # Verify request completion was called
        mock_monitor_instance.complete_request.assert_called_once()
        complete_args = mock_monitor_instance.complete_request.call_args
        assert complete_args[0][0] == "test_request_123"  # request_id
        assert complete_args[1]["success"] is True
        assert "result" in complete_args[1]

    @pytest.mark.asyncio
    @patch("aider_mcp_server.molecules.tools.aider_ai_code.RequestMonitor")
    @patch("aider_mcp_server.molecules.tools.aider_ai_code._execute_aider_with_coordination")
    @patch("aider_mcp_server.molecules.tools.aider_ai_code._validate_working_dir_and_api_keys")
    async def test_request_monitor_error_handling(
        self,
        mock_validate_working_dir,
        mock_execute_aider,
        mock_request_monitor_class,
        mock_coordinator,
    ):
        """Test that RequestMonitor properly handles errors during execution."""
        # Setup mocks
        mock_validate_working_dir.return_value = None
        mock_monitor_instance = AsyncMock()
        mock_monitor_instance.track_request = AsyncMock(return_value="test_request_123")
        mock_monitor_instance.update_request_progress = AsyncMock()
        mock_monitor_instance.complete_request = AsyncMock()
        mock_request_monitor_class.return_value = mock_monitor_instance

        # Make AIDER execution fail
        test_error = Exception("AIDER execution failed")
        mock_execute_aider.side_effect = test_error

        # Execute and expect exception
        with pytest.raises(Exception, match="AIDER execution failed"):
            await code_with_aider(
                ai_coding_prompt="Test prompt",
                relative_editable_files=["test.py"],
                working_dir="/tmp/test",
                coordinator=mock_coordinator,
            )

        # Verify request was marked as failed
        mock_monitor_instance.complete_request.assert_called_once()
        complete_args = mock_monitor_instance.complete_request.call_args
        assert complete_args[0][0] == "test_request_123"
        assert complete_args[1]["success"] is False
        assert "result" in complete_args[1]
        result = complete_args[1]["result"]
        assert result["success"] is False
        assert "error" in result
        assert result["error_type"] == "Exception"

    @pytest.mark.asyncio
    async def test_request_monitor_without_coordinator(self):
        """Test that AIDER works normally when no coordinator is provided."""
        with patch(
            "aider_mcp_server.molecules.tools.aider_ai_code._validate_working_dir_and_api_keys"
        ) as mock_validate:
            mock_validate.return_value = None

            with patch(
                "aider_mcp_server.molecules.tools.aider_ai_code._execute_aider_with_coordination"
            ) as mock_execute:
                mock_execute.return_value = {"success": True, "changes_summary": {"files": []}}

                with patch(
                    "aider_mcp_server.molecules.tools.aider_ai_code._handle_api_key_checks_and_warnings"
                ) as mock_api_check:
                    mock_api_check.return_value = ({}, None)

                    with patch("aider_mcp_server.molecules.tools.aider_ai_code._finalize_aider_response"):
                        # Execute without coordinator
                        result = await code_with_aider(
                            ai_coding_prompt="Test prompt",
                            relative_editable_files=["test.py"],
                            working_dir="/tmp/test",
                            coordinator=None,  # No coordinator
                        )

        # Should complete successfully without monitoring
        assert '"success": true' in result.lower()

    @pytest.mark.asyncio
    async def test_request_monitor_events_emitted(self, mock_coordinator):
        """Test that RequestMonitor emits appropriate events during request lifecycle."""
        # Create real RequestMonitor instance to test event emission
        monitor = RequestMonitor(mock_coordinator)

        # Track a request
        request_id = await monitor.track_request(context={"test": "context"})

        # Verify session started event was emitted
        mock_coordinator.broadcast_event.assert_called_with(
            EventTypes.AIDER_SESSION_STARTED,
            {
                "request_id": request_id,
                "start_time": mock_coordinator.broadcast_event.call_args[0][1]["start_time"],
                "context": {"test": "context"},
                "timestamp": mock_coordinator.broadcast_event.call_args[0][1]["timestamp"],
            },
        )

        # Update progress
        await monitor.update_request_progress(request_id, {"stage": "processing"})

        # Complete the request
        await monitor.complete_request(request_id, success=True, result={"files": 2})

        # Verify completion event was emitted
        assert mock_coordinator.broadcast_event.call_count >= 2
        completion_call = [
            call
            for call in mock_coordinator.broadcast_event.call_args_list
            if call[0][0] == EventTypes.AIDER_SESSION_COMPLETED
        ][0]
        assert completion_call[0][1]["success"] is True
        assert completion_call[0][1]["result"]["files"] == 2


class TestThrottlingDetection:
    """Test throttling detection functionality."""

    @pytest.mark.asyncio
    async def test_throttling_warning_events(self, mock_coordinator):
        """Test that warning and throttling events are emitted for long-running requests."""
        # Create monitor with very short thresholds for testing
        monitor = RequestMonitor(
            coordinator=mock_coordinator,
            warning_threshold=0.1,  # 100ms
            throttling_threshold=0.2,  # 200ms
        )

        # Track a request
        request_id = await monitor.track_request(context={"test": "long_running"})

        # Wait for warning threshold
        await asyncio.sleep(0.15)  # Wait past warning threshold

        # Check if warning event was emitted
        warning_calls = [
            call
            for call in mock_coordinator.broadcast_event.call_args_list
            if call[0][0] == EventTypes.AIDER_OPERATION_STATUS
        ]
        assert len(warning_calls) >= 1
        warning_event = warning_calls[0][0][1]
        assert warning_event["status"] == "long_running_warning"

        # Wait for throttling threshold
        await asyncio.sleep(0.1)  # Wait past throttling threshold

        # Check if throttling event was emitted
        throttling_calls = [
            call
            for call in mock_coordinator.broadcast_event.call_args_list
            if call[0][0] == EventTypes.AIDER_THROTTLING_DETECTED
        ]
        assert len(throttling_calls) >= 1
        throttling_event = throttling_calls[0][0][1]
        assert throttling_event["status"] == "throttled"

        # Complete the request to clean up
        await monitor.complete_request(request_id, success=True)
        await monitor.shutdown()

