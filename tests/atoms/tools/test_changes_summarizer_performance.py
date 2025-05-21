"""
Performance tests for the changes summarizer.
These tests verify that the summarizer effectively reduces token usage
while maintaining essential information about changes.
"""

import json
import os
import tempfile
import time

import pytest

from aider_mcp_server.atoms.tools.changes_summarizer import (
    get_file_status_summary,
    summarize_changes,
)


class TestChangesSummarizerPerformance:
    """Performance and utility tests for the changes summarizer."""

    def test_summarizer_handles_large_diffs(self):
        """Test that the summarizer can handle large diffs efficiently."""
        # Create a large multi-file diff
        large_diff = ""

        # Add 50 files to the diff
        for i in range(50):
            file_diff = f"""diff --git a/file{i}.py b/file{i}.py
index 1234{i}..abcde{i} 100644
--- a/file{i}.py
+++ b/file{i}.py
@@ -1,10 +1,15 @@
 # File {i} header

 def function_{i}_1():
+    # Added comment
     return {i} * 2

 def function_{i}_2():
     return {i} + 10

+def new_function_{i}():
+    print("New function added")
+    return {i} * 5

 # End of file {i}
"""
            large_diff += file_diff

        # Measure the time to summarize
        start_time = time.time()
        summary = summarize_changes(large_diff)
        end_time = time.time()

        # The summarizer should be reasonably fast (under 1 second for this test)
        processing_time = end_time - start_time
        assert processing_time < 1.0, f"Summarization took too long: {processing_time:.2f} seconds"

        # Verify the file count is capped at the default max_files
        assert len(summary["files"]) <= 10, "Should respect max_files parameter"

        # Verify the summary has essential stats
        assert summary["stats"]["total_files_changed"] == 50
        assert summary["stats"]["files_modified"] > 0

        # Verify token reduction
        diff_size = len(large_diff)
        summary_size = len(json.dumps(summary))
        reduction_ratio = 1 - (summary_size / diff_size)

        # Should have at least 80% reduction for large diffs
        assert reduction_ratio > 0.8, f"Token reduction too small: {reduction_ratio:.2%}"

    def test_summarizer_with_file_modifications(self):
        """Test that the summarizer correctly handles file modifications."""
        diff = """diff --git a/test_file.py b/test_file.py
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
        # Summarize the changes
        summary = summarize_changes(diff)

        # Verify the file is detected and operation is correct
        file_entries = [f for f in summary["files"] if f["name"] == "test_file.py"]
        assert len(file_entries) > 0
        file_entry = file_entries[0]
        assert file_entry["operation"] == "modified"

        # Check lines added is at least 1
        assert file_entry["lines_added"] >= 1

        # Check changes contain the added line in some form
        changes = file_entry["changes"]
        assert any('+    print("World")' in change for change in changes)

    def test_file_status_summary_performance(self):
        """Test that file status summary performs well with many files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create 100 files
            file_count = 100
            file_paths = []

            for i in range(file_count):
                file_path = os.path.join(temp_dir, f"file_{i}.py")
                with open(file_path, "w") as f:
                    # Half empty, half with content
                    if i % 2 == 0:
                        f.write("")
                    else:
                        f.write(f"def function_{i}():\n    return {i}\n")

                file_paths.append(f"file_{i}.py")

            # Measure performance
            start_time = time.time()
            result = get_file_status_summary(file_paths, temp_dir)
            end_time = time.time()

            processing_time = end_time - start_time
            assert processing_time < 1.0, f"File status check took too long: {processing_time:.2f} seconds"

            # Verify results
            assert result["has_changes"] is True
            assert result["files_created"] == file_count // 2  # Half are empty
            assert result["files_modified"] == file_count // 2  # Half have content

    def test_summarizer_with_line_limits(self):
        """Test that max_context_lines parameter affects output."""
        # Create a diff
        diff = """diff --git a/test_file.py b/test_file.py
index 1234567..abcdef0 100644
--- a/test_file.py
+++ b/test_file.py
@@ -1,10 +1,12 @@
 def hello():
     print("Hello")
+    print("World")
+    print("!")

 def goodbye():
     print("Goodbye")

 # Additional content
 # More content
 # Even more content
"""

        # Test with different context line settings
        summary_small = summarize_changes(diff, max_context_lines=1)
        summary_large = summarize_changes(diff, max_context_lines=5)

        # Both should have the file
        small_entries = [f for f in summary_small["files"] if f["name"] == "test_file.py"]
        large_entries = [f for f in summary_large["files"] if f["name"] == "test_file.py"]
        assert len(small_entries) > 0
        assert len(large_entries) > 0

        # Both should have changes
        assert len(small_entries[0]["changes"]) > 0
        assert len(large_entries[0]["changes"]) > 0

    def test_summarizer_with_file_size_limits(self):
        """Test that max_file_kb parameter has some effect."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = os.path.join(temp_dir, "test_file.py")
            with open(test_file, "w") as f:
                # Write a file with several lines
                for i in range(1, 50):
                    f.write(f"def function_{i}():\n    return {i}\n")

            # Format as file contents
            with open(test_file, "r") as f:
                content = f.read()

            file_contents = f"""File contents after editing (git not used):

--- test_file.py ---
{content}
"""

            # Summarize
            summary = summarize_changes(file_contents)

            # Verify we got a summary
            file_entries = [f for f in summary["files"] if f["name"] == "test_file.py"]
            assert len(file_entries) > 0
            file_entry = file_entries[0]
            assert file_entry["operation"] == "created"

            # Should have some changes to display
            assert len(file_entry["changes"]) > 0

    @pytest.mark.asyncio
    async def test_summarizer_token_reduction(self):
        """Test that summarized changes use significantly fewer tokens than full diffs."""
        # Create a large diff with many changes
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a large Python file with many lines
            large_file = os.path.join(temp_dir, "large_file.py")
            with open(large_file, "w") as f:
                # Write ~1000 lines of code
                for i in range(1, 501):
                    f.write(f"def function_{i}():\n    return {i}\n")

            # Read the file content
            with open(large_file, "r") as f:
                file_content = f.read()

            # Create a diff-like format
            diff = f"diff --git a/large_file.py b/large_file.py\n--- a/large_file.py\n+++ b/large_file.py\n@@ -1,1000 +1,1000 @@\n{file_content}"

            # Get the size of the original diff (as a proxy for token count)
            diff_size = len(diff)

            # Summarize the changes
            summary = summarize_changes(diff)
            summary_json = json.dumps(summary)
            summary_size = len(summary_json)

            # The summary should be significantly smaller (we expect at least 80% reduction)
            reduction_ratio = 1 - (summary_size / diff_size)
            assert reduction_ratio > 0.8, f"Summary reduction only {reduction_ratio:%}"

            # Verify that the summary still contains useful information
            assert summary["summary"] is not None and len(summary["summary"]) > 0
            file_entries = [f for f in summary["files"] if f["name"] == "large_file.py"]
            assert len(file_entries) > 0
