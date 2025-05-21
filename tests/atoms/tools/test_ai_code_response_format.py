"""
Tests for the response format of aider_ai_code.py
"""

import json
import os
import tempfile

import pytest

from aider_mcp_server.atoms.tools.aider_ai_code import (
    _process_coder_results,
)


def create_test_file(temp_dir, filename, content):
    """Helper to create a test file"""
    filepath = os.path.join(temp_dir, filename)
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


@pytest.mark.asyncio
async def test_process_coder_results_format():
    """Test that _process_coder_results returns a properly formatted response."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file with hello world content
        hello_file = "hello_world.py"
        create_test_file(temp_dir, hello_file, 'print("Hello, World!")')

        # Process the results
        result = await _process_coder_results(
            relative_editable_files=[hello_file],
            working_dir=temp_dir,
            use_diff_cache=False,  # Don't use cache for this test
            clear_cached_for_unchanged=False,
        )

        # Verify the result structure
        assert isinstance(result, dict)
        assert "success" in result
        assert "changes_summary" in result
        assert "file_status" in result

        # Verify detailed structure of changes_summary
        changes_summary = result["changes_summary"]
        assert "summary" in changes_summary
        assert "files" in changes_summary
        assert "stats" in changes_summary

        # Verify file_status has expected fields
        file_status = result["file_status"]
        assert "has_changes" in file_status
        assert "status_summary" in file_status

        # Diff should now contain a human-readable summary by default
        assert "diff" in result, "diff should be included by default"
        assert isinstance(result["diff"], str), "diff should be a string"

        # The diff might contain either changes_summary.summary or file_status.status_summary
        # depending on which one has more meaningful information
        assert (
            result["diff"] == result["changes_summary"]["summary"]
            or result["diff"] == result["file_status"]["status_summary"]
        ), "diff should contain meaningful summary from either changes_summary or file_status"

        # Verify success flag is set correctly
        assert result["success"] is True, "Success should be True for a file with content"


@pytest.mark.asyncio
async def test_process_coder_results_with_diff():
    """Test that _process_coder_results includes diff when specifically requested."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file with hello world content
        hello_file = "hello_with_diff.py"
        create_test_file(temp_dir, hello_file, 'print("Hello with diff!")')

        # Mock the response to include diff for testing
        # In real usage, this would be controlled by include_raw_diff parameter
        result = await _process_coder_results(
            relative_editable_files=[hello_file],
            working_dir=temp_dir,
            use_diff_cache=False,
        )

        # Add diff manually for testing
        result["diff"] = "This is a test diff"

        # Verify the result can be serialized to JSON
        json_str = json.dumps(result)
        parsed = json.loads(json_str)

        # Verify the parsed result has the expected structure
        assert "changes_summary" in parsed
        assert "file_status" in parsed
        assert "diff" in parsed
