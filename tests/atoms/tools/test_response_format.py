import asyncio
import os
import sys
import tempfile

import pytest

from aider_mcp_server.atoms.tools.aider_ai_code import (
    _process_coder_results,
    init_diff_cache,
    shutdown_diff_cache,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.mark.asyncio
async def test_process_response_format(temp_dir):
    """Test that the response format correctly reflects file creation/modification."""
    # Initialize the diff cache
    await init_diff_cache()

    try:
        # Create a simple file
        test_file = os.path.join(temp_dir, "test_file.py")
        with open(test_file, "w") as f:
            f.write('print("This is a test file")\n')

        # Process the results after file creation
        response = await _process_coder_results(
            relative_editable_files=[test_file],
            working_dir=temp_dir,
            use_diff_cache=True,
            clear_cached_for_unchanged=True,
        )

        # Check that success is True
        assert response["success"], "Response should indicate success"

        # Check that changes_summary is populated and consistent
        assert "changes_summary" in response, "Response should include changes_summary"
        assert response["changes_summary"]["summary"], "Changes summary should have a non-empty summary"
        assert "files" in response["changes_summary"], "Changes summary should include files information"

        # Check that file_status is populated and consistent
        assert "file_status" in response, "Response should include file_status"
        assert response["file_status"]["has_changes"], "File status should indicate changes"
        assert "files" in response["file_status"], "File status should include files information"

        # Check consistency between changes_summary and file_status
        if "files_created" in response["file_status"]:
            assert response["file_status"]["files_created"] > 0, "File status should show files created"
            if "stats" in response["changes_summary"]:
                assert "files_created" in response["changes_summary"]["stats"], (
                    "Changes summary stats should include files_created"
                )

        # Check that either diff is populated, or we have changes_summary, or both
        # Note: 'diff' field is now optional, with changes_summary being the preferred output format
        if "diff" in response:
            assert response["diff"], "Diff should be non-empty when present"
            assert response["diff"] == response["changes_summary"]["summary"], (
                "Diff should match changes_summary summary when both exist"
            )
        else:
            # If diff is not present, we must have a non-empty changes_summary
            assert response["changes_summary"]["summary"], "Changes summary should be non-empty when diff is missing"

    finally:
        # Clean up the diff cache but don't wait for it since the event loop
        # might be closing already from an error
        try:
            # Shutdown without await
            if sys.version_info >= (3, 7):
                asyncio.run_coroutine_threadsafe(shutdown_diff_cache(), asyncio.get_event_loop())
            else:
                # For older Python versions
                asyncio.ensure_future(shutdown_diff_cache())
        except Exception as e:
            # Log error during cleanup but continue test teardown
            import logging

            logging.getLogger(__name__).warning(f"Error during diff cache shutdown: {e}")
