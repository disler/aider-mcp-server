"""Test Phase 1.3 enhanced progress monitoring with lightweight changes summary."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from aider_mcp_server.atoms.types.event_types import EventTypes
from aider_mcp_server.atoms.types.streaming_types import AiderChangesSummary, ChangeType
from aider_mcp_server.molecules.tools.aider_ai_code import _broadcast_changes_summary
from aider_mcp_server.pages.application.coordinator import ApplicationCoordinator


@pytest_asyncio.fixture
async def mock_coordinator():
    """Create a mock coordinator for testing Phase 1.3 functionality."""
    coordinator = AsyncMock(spec=ApplicationCoordinator)
    coordinator.broadcast_event = AsyncMock()
    return coordinator


@pytest.mark.asyncio
async def test_changes_summary_broadcasting():
    """Test that changes summary is properly converted and broadcasted."""
    mock_coordinator = AsyncMock()
    mock_coordinator.broadcast_event = AsyncMock()

    # Mock response with changes_summary
    response = {
        "success": True,
        "changes_summary": {
            "summary": "Added new feature and fixed bug in user authentication",
            "files": [
                {"name": "src/auth.py", "operation": "modified"},
                {"name": "src/utils.py", "operation": "created"},
                {"name": "tests/test_auth.py", "operation": "modified"},
            ],
            "stats": {"lines_added": 45, "lines_removed": 12, "files_changed": 3},
        },
        "file_status": {"has_changes": True},
    }

    session_id = "test_session_123"
    relative_editable_files = ["src/auth.py", "src/utils.py", "tests/test_auth.py"]

    # Call the broadcast function
    await _broadcast_changes_summary(mock_coordinator, response, session_id, relative_editable_files)

    # Verify broadcast_event was called with correct event type
    mock_coordinator.broadcast_event.assert_called_once()
    call_args = mock_coordinator.broadcast_event.call_args

    assert call_args[0][0] == EventTypes.AIDER_CHANGES_SUMMARY

    # Verify the payload structure
    payload = call_args[0][1]
    assert payload["session_id"] == session_id
    assert payload["total_files_changed"] == 3
    assert payload["files_created"] == 1  # src/utils.py was created
    assert payload["files_modified"] == 2  # src/auth.py and tests/test_auth.py were modified
    assert payload["files_deleted"] == 0
    assert payload["total_lines_added"] == 45
    assert payload["total_lines_removed"] == 12
    assert "feature_addition" in payload["change_categories"]
    assert "bug_fix" in payload["change_categories"]
    assert payload["estimated_complexity"] == "simple"  # < 5 files and < 100 lines


@pytest.mark.asyncio
async def test_changes_summary_complexity_estimation():
    """Test complexity estimation based on files and lines changed."""
    mock_coordinator = AsyncMock()
    mock_coordinator.broadcast_event = AsyncMock()

    # Test complex change scenario
    response = {
        "success": True,
        "changes_summary": {
            "summary": "Major refactoring of authentication system",
            "files": [{"name": f"src/module_{i}.py", "operation": "modified"} for i in range(12)],
            "stats": {"lines_added": 250, "lines_removed": 180, "files_changed": 12},
        },
    }

    await _broadcast_changes_summary(mock_coordinator, response, "test_complex", [])

    payload = mock_coordinator.broadcast_event.call_args[0][1]
    assert payload["estimated_complexity"] == "complex"  # > 10 files and > 300 total lines
    assert payload["total_files_changed"] == 12


@pytest.mark.asyncio
async def test_changes_summary_category_detection():
    """Test automatic categorization of changes based on summary text."""
    mock_coordinator = AsyncMock()

    test_cases = [
        ("Fixed critical bug in payment processing", ["bug_fix"]),
        ("Added new user dashboard feature", ["feature_addition"]),
        ("Refactored database connection logic", ["refactoring"]),
        ("Added comprehensive test coverage", ["testing"]),
        ("Updated user interface styling", ["general_update"]),
        ("Fixed authentication bug and added new login feature", ["bug_fix", "feature_addition"]),
    ]

    for summary_text, expected_categories in test_cases:
        mock_coordinator.broadcast_event = AsyncMock()

        response = {
            "success": True,
            "changes_summary": {
                "summary": summary_text,
                "files": [{"name": "test.py", "operation": "modified"}],
                "stats": {"lines_added": 10, "lines_removed": 5},
            },
        }

        await _broadcast_changes_summary(mock_coordinator, response, "test_session", [])

        payload = mock_coordinator.broadcast_event.call_args[0][1]
        for category in expected_categories:
            assert category in payload["change_categories"], (
                f"Expected {category} in {payload['change_categories']} for '{summary_text}'"
            )


@pytest.mark.asyncio
async def test_changes_summary_with_no_changes():
    """Test handling when no changes are detected."""
    mock_coordinator = AsyncMock()
    mock_coordinator.broadcast_event = AsyncMock()

    response = {"success": False, "changes_summary": {"summary": "No changes detected", "files": [], "stats": {}}}

    await _broadcast_changes_summary(mock_coordinator, response, "test_no_changes", [])

    payload = mock_coordinator.broadcast_event.call_args[0][1]
    assert payload["total_files_changed"] == 0
    assert payload["files_created"] == 0
    assert payload["files_modified"] == 0
    assert payload["files_deleted"] == 0
    assert payload["estimated_complexity"] == "simple"


@pytest.mark.asyncio
async def test_changes_summary_error_handling():
    """Test that broadcast function handles errors gracefully."""
    mock_coordinator = AsyncMock()
    mock_coordinator.broadcast_event = AsyncMock(side_effect=Exception("Broadcast failed"))

    response = {"success": True, "changes_summary": {"summary": "Test changes"}, "file_status": {}}

    # Should not raise exception even if broadcast fails
    await _broadcast_changes_summary(mock_coordinator, response, "test_error", [])

    # Verify broadcast was attempted
    mock_coordinator.broadcast_event.assert_called_once()


@pytest.mark.asyncio
async def test_stdio_transport_phase_1_3_subscription(clean_coordinator):
    """Test that STDIO transport subscribes to Phase 1.3 events."""
    coordinator = clean_coordinator

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        from aider_mcp_server.organisms.transports.stdio.stdio_transport_adapter import StdioTransportAdapter

        stdio_adapter = StdioTransportAdapter(
            coordinator=coordinator, discovery_file=temp_path / "coordinator_discovery.json"
        )

        try:
            await stdio_adapter.initialize()

            # Test broadcasting Phase 1.3 events
            test_changes_summary: AiderChangesSummary = {
                "session_id": "test_phase_1_3",
                "timestamp": 1234567890.0,
                "total_files_changed": 2,
                "files_created": 1,
                "files_modified": 1,
                "files_deleted": 0,
                "total_lines_added": 25,
                "total_lines_removed": 5,
                "file_summaries": [
                    {
                        "file_path": "src/test.py",
                        "change_type": ChangeType.CREATED.value,
                        "lines_added": 20,
                        "lines_removed": 0,
                        "lines_modified": 0,
                        "change_description": "Created new test file",
                        "estimated_impact": "medium",
                    }
                ],
                "change_categories": ["testing"],
                "estimated_complexity": "simple",
            }

            # Broadcast Phase 1.3 event
            await coordinator.broadcast_event(EventTypes.AIDER_CHANGES_SUMMARY, test_changes_summary)

            # Give time for event processing
            await asyncio.sleep(0.1)

            # Test passes if no exceptions are raised

        finally:
            await stdio_adapter.shutdown()


@pytest_asyncio.fixture
async def clean_coordinator():
    """Provide a clean coordinator instance for each test."""
    # Reset singleton state for testing
    ApplicationCoordinator._instance = None
    ApplicationCoordinator._initialized = False

    # Get fresh instance
    coordinator = await ApplicationCoordinator.getInstance()
    await coordinator.__aenter__()

    yield coordinator

    # Clean up after test
    await coordinator.__aexit__(None, None, None)

    # Reset singleton state for next test
    ApplicationCoordinator._instance = None
    ApplicationCoordinator._initialized = False
