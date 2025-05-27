"""
Tests for aider_ai_code changes handling, focusing on diff generation and file change detection.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from aider_mcp_server.molecules.tools.aider_ai_code import (
    _check_for_meaningful_changes,
    _get_changes_diff_or_content,
    _process_coder_results,
)
from aider_mcp_server.molecules.tools.changes_summarizer import (
    get_file_status_summary,
    summarize_changes,
)


class TestAiderChangesHandling:
    """Tests for aider_ai_code changes handling."""

    def test_check_for_meaningful_changes_new_file(self):
        """Test that _check_for_meaningful_changes correctly detects new files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a new file with simple content
            test_file = os.path.join(temp_dir, "test_file.py")
            with open(test_file, "w") as f:
                f.write("def hello():\n    print('Hello')\n")

            # Relative path for the function
            relative_path = "test_file.py"

            # Check with working directory
            has_meaningful_changes = _check_for_meaningful_changes([relative_path], temp_dir)
            assert has_meaningful_changes is True

    def test_check_for_meaningful_changes_empty_file(self):
        """Test that _check_for_meaningful_changes correctly handles empty files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create an empty file
            test_file = os.path.join(temp_dir, "empty_file.py")
            open(test_file, "w").close()  # Create empty file without unused variable

            # Relative path for the function
            relative_path = "empty_file.py"

            # Check with working directory
            has_meaningful_changes = _check_for_meaningful_changes([relative_path], temp_dir)
            assert has_meaningful_changes is False  # Empty files aren't meaningful changes

    def test_check_for_meaningful_changes_minimal_content(self):
        """Test that _check_for_meaningful_changes correctly handles files with minimal content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with minimal content that might not be considered meaningful
            test_file = os.path.join(temp_dir, "minimal_file.py")
            with open(test_file, "w") as f:
                f.write("# Just a comment\n")

            # Relative path for the function
            relative_path = "minimal_file.py"

            # Check with working directory
            has_meaningful_changes = _check_for_meaningful_changes([relative_path], temp_dir)
            assert has_meaningful_changes is False  # Just a comment isn't meaningful

    def test_check_for_meaningful_changes_code_patterns(self):
        """Test that _check_for_meaningful_changes recognizes code patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_files = [
                ("def_file.py", "def function():\n    pass\n"),
                ("class_file.py", "class TestClass:\n    pass\n"),
                ("import_file.py", "import os\nimport sys\n"),
                ("indented_file.py", "if True:\n    do_something()\n"),
            ]

            for filename, content in test_files:
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, "w") as f:
                    f.write(content)

                # Test each file individually
                has_meaningful_changes = _check_for_meaningful_changes([filename], temp_dir)
                assert has_meaningful_changes is True, f"Failed for {filename}"

    def test_get_changes_diff_or_content_git(self):
        """Test that _get_changes_diff_or_content gets git diff when possible."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = os.path.join(temp_dir, "test_file.py")
            with open(test_file, "w") as f:
                f.write("def hello():\n    print('Hello')\n")

            # Mock git diff command
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "mock git diff output"

            with patch("subprocess.run", return_value=mock_result) as mock_run:
                diff_output = _get_changes_diff_or_content(["test_file.py"], temp_dir)

                # Verify git command was called
                mock_run.assert_called_once()
                # Check that git diff output was returned
                assert diff_output == "mock git diff output"

    # Re-enabled after verifying exception handling is properly implemented
    def test_get_changes_diff_or_content_fallback(self):
        """Test that _get_changes_diff_or_content falls back to reading file contents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = os.path.join(temp_dir, "test_file.py")
            with open(test_file, "w") as f:
                f.write("def hello():\n    print('Hello')\n")

            # Mock git diff command to fail
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "git command failed"

            with patch("subprocess.run", side_effect=Exception("git error")) as mock_run:
                diff_output = _get_changes_diff_or_content(["test_file.py"], temp_dir)

                # Verify git command was called
                mock_run.assert_called_once()
                # Check that file content fallback was used
                assert "File contents after editing" in diff_output
                assert "def hello():" in diff_output

    @pytest.mark.asyncio
    async def test_process_coder_results_with_changes(self):
        """Test _process_coder_results with meaningful changes."""
        with patch("aider_mcp_server.molecules.tools.aider_ai_code.get_changes_diff_or_content") as mock_get_diff:
            mock_get_diff.return_value = "mock diff output"

            with patch("aider_mcp_server.molecules.tools.aider_ai_code._check_for_meaningful_changes") as mock_check:
                mock_check.return_value = True

                with patch("aider_mcp_server.molecules.tools.aider_ai_code.diff_cache") as mock_cache:
                    mock_cache.compare_and_cache.return_value = {"diff": "mock diff output"}

                    # Call the function
                    result = await _process_coder_results(["test_file.py"], "/test/dir", False, False)

                    # Verify the result
                    assert result["success"] is True
                    # Updated test to check for diff field presence but not exact content
                    assert "diff" in result
                    # Skip checking the exact text since we now use a summary instead

    @pytest.mark.asyncio
    async def test_process_coder_results_no_changes(self):
        """Test _process_coder_results with no meaningful changes."""
        with patch("aider_mcp_server.molecules.tools.aider_ai_code.get_changes_diff_or_content") as mock_get_diff:
            mock_get_diff.return_value = "mock diff output"

            with patch("aider_mcp_server.molecules.tools.aider_ai_code._check_for_meaningful_changes") as mock_check:
                mock_check.return_value = False

                # Also mock get_file_status_summary
                with patch("aider_mcp_server.molecules.tools.aider_ai_code.get_file_status_summary") as mock_status:
                    mock_status.return_value = {
                        "has_changes": False,
                        "status_summary": "No changes detected.",
                        "files_created": 0,
                        "files_modified": 0,
                    }

                    with patch("aider_mcp_server.molecules.tools.aider_ai_code.diff_cache") as mock_cache:
                        mock_cache.compare_and_cache.return_value = {"diff": "mock diff output"}

                        # Call the function
                        result = await _process_coder_results(["test_file.py"], "/test/dir", False, False)

                        # Verify the result
                        assert result["success"] is False

                        # Check that we have the summary fields
                        assert "changes_summary" in result
                        assert "file_status" in result

    def test_get_file_status_summary_integration(self):
        """Test get_file_status_summary with real files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with different characteristics
            files = [
                ("empty.py", ""),
                ("minimal.py", "# Just a comment"),
                ("code.py", "def function():\n    return True"),
            ]

            for filename, content in files:
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, "w") as f:
                    f.write(content)

            # Get status summary for all files
            result = get_file_status_summary([f[0] for f in files], temp_dir)

            # Check the results
            assert result["has_changes"] is True
            assert result["files_created"] > 0  # At least the empty file should be counted as created
            assert result["files_modified"] > 0  # At least the code file should be counted as modified

    def test_summarize_changes_integration(self):
        """Test summarize_changes with realistic diff content."""
        # Git diff style content
        git_diff = """diff --git a/test_file.py b/test_file.py
index 1234567..abcdef0 100644
--- a/test_file.py
+++ b/test_file.py
@@ -1,5 +1,6 @@
 def hello():
     print("Hello")
+    print("World")

 def goodbye():
     print("Goodbye")
"""
        # File contents fallback style
        file_contents = """File contents after editing (git not used):

--- test_file.py ---
def hello():
    print("Hello")
    print("World")

def goodbye():
    print("Goodbye")
"""

        # Test both formats
        git_result = summarize_changes(git_diff)
        file_result = summarize_changes(file_contents)

        # Verify git diff summary
        assert "Changed 1 files" in git_result["summary"]
        assert git_result["stats"]["lines_added"] == 1

        # Check for test_file.py in git result files array
        found_file = False
        for file_entry in git_result["files"]:
            if file_entry["name"] == "test_file.py":
                found_file = True
                break
        assert found_file, "test_file.py should be in the files array"

        # Verify file contents summary
        assert "Processed 1 files" in file_result["summary"]

        # Check for test_file.py in file result files array
        found_file = False
        for file_entry in file_result["files"]:
            if file_entry["name"] == "test_file.py":
                found_file = True
                break
        assert found_file, "test_file.py should be in the file_result files array"

    def test_summarize_changes_new_files(self):
        """Test summarize_changes correctly handles new files."""
        # Git diff style for new file
        git_diff = """diff --git a/new_file.py b/new_file.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+def new_function():
+    print("This is new")
+    return True
"""
        # Test summarization
        result = summarize_changes(git_diff)

        # Verify new file is correctly identified
        assert result["stats"]["files_created"] == 1
        assert result["stats"]["lines_added"] == 3

        # Find the new file in the array
        new_file_entry = None
        for file_entry in result["files"]:
            if file_entry["name"] == "new_file.py":
                new_file_entry = file_entry
                break

        assert new_file_entry is not None, "new_file.py should be in the files array"
        assert new_file_entry["operation"] == "created", "Operation should be 'created'"
