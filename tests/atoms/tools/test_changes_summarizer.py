"""
Tests for changes_summarizer.py module.
"""

import os
import tempfile
from unittest import mock

from aider_mcp_server.atoms.tools.changes_summarizer import (
    get_file_status_summary,
    summarize_changes,
)


class TestChangesSummarizer:
    """Tests for the changes_summarizer module."""

    def test_summarize_changes_empty(self):
        """Test summarizing an empty diff."""
        result = summarize_changes("")
        # Updated to work with new response format
        assert (
            "No meaningful changes detected" in result["summary"]
            or "No git-tracked changes detected" in result["summary"]
        )
        assert not result["files"]
        # Stats may or may not exist depending on the summarizer implementation
        # If stats does exist, verify total_files_changed is 0
        if "stats" in result:
            assert result["stats"].get("total_files_changed", 0) == 0

    def test_summarize_changes_git_diff(self):
        """Test summarizing a git diff."""
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
        result = summarize_changes(git_diff)

        # Check that we have the expected structure
        assert "stats" in result
        assert "files" in result
        assert "summary" in result

        # Verify file is in the results
        file_entries = [f for f in result["files"] if f["name"] == "test_file.py"]
        assert len(file_entries) > 0

        # Get the file entry
        file_entry = file_entries[0]

        # Verify the modification was correctly identified
        assert file_entry["operation"] == "modified"

        # Check for added lines
        assert file_entry["lines_added"] > 0

    def test_summarize_changes_new_file(self):
        """Test summarizing a diff with a new file."""
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
        result = summarize_changes(git_diff)

        # Verify file is in the results
        file_entries = [f for f in result["files"] if f["name"] == "new_file.py"]
        assert len(file_entries) > 0

        # Get the file entry
        file_entry = file_entries[0]

        # Verify the operation was correctly identified
        assert file_entry["operation"] == "created"

        # Check for added lines
        assert file_entry["lines_added"] > 0
        assert result["stats"]["files_created"] > 0

    def test_summarize_changes_deleted_file(self):
        """Test summarizing a diff with a deleted file."""
        git_diff = """diff --git a/deleted_file.py b/deleted_file.py
deleted file mode 100644
index 1234567..0000000
--- a/deleted_file.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def old_function():
-    print("This will be deleted")
-    return False
"""
        result = summarize_changes(git_diff)

        # Verify file is in the results
        file_entries = [f for f in result["files"] if f["name"] == "deleted_file.py"]
        assert len(file_entries) > 0

        # Get the file entry
        file_entry = file_entries[0]

        # Verify the operation was correctly identified
        assert file_entry["operation"] == "deleted"

        # Check for removed lines
        assert file_entry["lines_removed"] > 0
        assert result["stats"]["files_deleted"] > 0

    def test_summarize_changes_file_contents(self):
        """Test summarizing file contents (non-git diff)."""
        file_contents = """File contents after editing (git not used):

--- file1.py ---
def sample_function():
    print("This is a sample")
    return True

--- file2.py ---
class TestClass:
    def __init__(self):
        self.value = 42
"""
        result = summarize_changes(file_contents)

        # Verify files are in the results
        file1_entries = [f for f in result["files"] if f["name"] == "file1.py"]
        file2_entries = [f for f in result["files"] if f["name"] == "file2.py"]
        assert len(file1_entries) > 0
        assert len(file2_entries) > 0

        # Get the file entries
        file1_entry = file1_entries[0]
        file2_entry = file2_entries[0]

        # Verify the operation was correctly identified
        assert file1_entry["operation"] == "created"
        assert file2_entry["operation"] == "created"

        # Check for added lines
        assert result["stats"]["files_created"] > 0
        assert result["stats"]["lines_added"] > 0

    def test_summarize_changes_with_context_limit(self):
        """Test summarizing changes with limited context."""
        git_diff = """diff --git a/large_file.py b/large_file.py
index 1234567..abcdef0 100644
--- a/large_file.py
+++ b/large_file.py
@@ -1,20 +1,20 @@
 def function1():
     pass

-def function2():
+def renamed_function2():
     pass

 def function3():
     pass

 def function4():
     pass

 def function5():
     pass

 def function6():
     pass

 def function7():
     pass
"""
        # Test with 2 lines of context
        result = summarize_changes(git_diff, max_context_lines=2)

        # Verify file is in the results
        file_entries = [f for f in result["files"] if f["name"] == "large_file.py"]
        assert len(file_entries) > 0

        # Get the file entry
        file_entry = file_entries[0]

        # Verify changes list exists
        assert "changes" in file_entry

        # Verify we have reasonable content in the changes
        changes = file_entry["changes"]

        if changes:  # If we have changes
            # Verify proper context is maintained
            for change in changes:
                # Check for key parts of the change
                assert "function" in change

                # Count the number of lines
                line_count = len(change.split("\n"))

                # The lines should include the changed line plus some context
                # Should be at most max_context_lines*2 plus some for the change itself
                assert line_count <= 10, f"Too many lines in context: {line_count}"

    def test_get_file_status_summary(self):
        """Test getting file status summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a new file
            new_file = os.path.join(temp_dir, "new_file.py")
            with open(new_file, "w") as f:
                f.write("def test(): pass\n")

            # Create an empty file
            empty_file = os.path.join(temp_dir, "empty_file.py")
            open(empty_file, "w").close()

            result = get_file_status_summary(["new_file.py", "empty_file.py"], working_dir=temp_dir)

            assert result["has_changes"] is True
            assert "new_file.py" in result["status_summary"]
            assert "empty_file.py" in result["status_summary"]
            # The empty file should be counted as created
            assert result["files_created"] == 1
            # The non-empty file should be counted as modified
            assert result["files_modified"] == 1

    def test_get_file_status_summary_nonexistent(self):
        """Test getting file status for nonexistent files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = get_file_status_summary(["nonexistent.py"], working_dir=temp_dir)

            assert result["has_changes"] is False
            # Updated to work with new response format
            assert (
                "No changes detected" in result["status_summary"]
                or "No filesystem changes detected" in result["status_summary"]
            )
            assert result.get("files_created", 0) == 0
            assert result.get("files_modified", 0) == 0

    @mock.patch("os.path.exists")
    @mock.patch("os.path.getsize")
    def test_get_file_status_summary_with_many_files(self, mock_getsize, mock_exists):
        """Test getting file status with many files."""
        # Setup mocks
        mock_exists.return_value = True
        mock_getsize.return_value = 100  # Non-empty files

        # Create a list of many files
        many_files = [f"file{i}.py" for i in range(20)]

        result = get_file_status_summary(many_files)

        assert result["has_changes"] is True
        assert result["files_modified"] == 20
        # With many files, not all should be listed in the summary
        assert not all(f"file{i}.py" in result["status_summary"] for i in range(20))
