"""
Module for summarizing code changes to reduce token usage.
"""

import os
import re
from typing import Dict, List, Optional, TypedDict, Union


class FileEntry(TypedDict, total=False):
    """TypedDict for file entries in the changes summary."""

    name: str
    operation: str
    lines_added: int
    lines_removed: int
    changes: List[str]


class ChangesStats(TypedDict, total=False):
    """TypedDict for statistics in the changes summary."""

    total_files_changed: int
    files_created: int
    files_modified: int
    files_deleted: int
    lines_added: int
    lines_removed: int


class ChangesSummary(TypedDict):
    """TypedDict for the changes summary result."""

    summary: str
    files: List[FileEntry]
    stats: ChangesStats


class FileStatusEntry(TypedDict):
    """TypedDict for file status entries."""

    name: str
    operation: str


class FileStatus(TypedDict, total=False):
    """TypedDict for file status result."""

    has_changes: bool
    status_summary: str
    files: List[FileStatusEntry]
    files_created: int
    files_modified: int


def summarize_changes(
    diff_content: str,
    max_context_lines: int = 3,
    max_files: int = 10,
    max_file_kb: int = 5,
) -> Dict[str, Union[str, List[Dict[str, Union[str, int, List[str]]]], Dict[str, int]]]:
    """
    Create a high-level summary of code changes from a diff string.

    Args:
        diff_content: The full diff content (either git diff output or file contents)
        max_context_lines: Maximum number of context lines to include per change
        max_files: Maximum number of files to include in the summary
        max_file_kb: Maximum size of each file's summary in KB

    Returns:
        A dictionary with high-level summary of changes, including:
        - summary: A text summary of changes across all files
        - files: An array of file summaries with details about changes
        - stats: Statistics about the changes (only non-zero values)
    """
    # Initialize the result structure with minimal data - stats will be added only if needed
    result: Dict[str, Union[str, List[Dict[str, Union[str, int, List[str]]]], Dict[str, int]]] = {
        "summary": "",
        "files": [],
    }

    # Check if input is empty
    if not diff_content or diff_content.strip() in [
        "No meaningful changes detected.",
        "No git-tracked changes detected.",
        "No git-tracked changes detected by cache comparison.",
    ]:
        result["summary"] = "No git-tracked changes detected."
        return result

    # Determine if this is a git diff or file contents
    is_git_diff = "diff --git" in diff_content

    if is_git_diff:
        return _summarize_git_diff(diff_content, max_context_lines, max_files, max_file_kb)
    else:
        return _summarize_file_contents(diff_content, max_context_lines, max_files, max_file_kb)


def _extract_filename_from_git_diff(diff_line: str) -> Optional[str]:
    """Extract filename from a git diff header line."""
    match = re.match(r"diff --git a/([^ ]+) b/([^ ]+)", diff_line)
    if match:
        return match.group(2)  # Use the 'b' path (new file path)
    return None


def _summarize_git_diff(
    diff_content: str,
    max_context_lines: int,
    max_files: int,
    max_file_kb: int,
) -> Dict[str, Union[str, List[Dict[str, Union[str, int, List[str]]]], Dict[str, int]]]:
    """Summarize changes from a git diff output."""
    result = {
        "summary": "",
        "files": [],
        # stats will be added only if there are non-zero values
    }

    # Stats counter (will only include non-zero values in final result)
    stats_counter = {
        "total_files_changed": 0,
        "files_created": 0,
        "files_modified": 0,
        "files_deleted": 0,
        "lines_added": 0,
        "lines_removed": 0,
    }

    # Process each diff section separately
    file_sections: List[str] = []
    current_section: List[str] = []

    for line in diff_content.splitlines(True):  # Keep line endings
        if line.startswith("diff --git") and current_section:
            file_sections.append("".join(current_section))
            current_section = [line]
        else:
            current_section.append(line)

    # Add the last section if it exists
    if current_section:
        file_sections.append("".join(current_section))

    # Process each file's diff
    file_count = 0
    for file_diff in file_sections:
        # Extract the file name from the first line
        diff_lines = file_diff.splitlines()
        if not diff_lines:
            continue

        file_name = _extract_filename_from_git_diff(diff_lines[0])
        if not file_name:
            continue

        file_count += 1
        stats_counter["total_files_changed"] += 1

        # Determine file operation
        if "new file mode" in file_diff:
            operation = "created"
            stats_counter["files_created"] += 1
        elif "deleted file mode" in file_diff:
            operation = "deleted"
            stats_counter["files_deleted"] += 1
        else:
            operation = "modified"
            stats_counter["files_modified"] += 1

        # Count added and removed lines
        lines_added = 0
        lines_removed = 0

        for line in diff_lines:
            if line.startswith("+") and not line.startswith("+++"):
                lines_added += 1
            elif line.startswith("-") and not line.startswith("---"):
                lines_removed += 1

        stats_counter["lines_added"] += lines_added
        stats_counter["lines_removed"] += lines_removed

        # Extract and summarize changes
        changes = []
        # Get the diff chunks - text between the @@ markers
        chunk_pattern = r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@"
        chunk_matches = list(re.finditer(chunk_pattern, file_diff))

        for i, match in enumerate(chunk_matches):
            chunk_start = match.end()
            chunk_end = chunk_matches[i + 1].start() if i + 1 < len(chunk_matches) else len(file_diff)
            chunk = file_diff[chunk_start:chunk_end]

            # Truncate the chunk if it's too large
            if len(chunk) > max_file_kb * 1024:
                chunk = chunk[: max_file_kb * 1024] + "\n... (truncated)"

            # Extract the context with limited lines before and after changes
            context_lines = []
            change_found = False

            for line in chunk.split("\n"):
                if line.startswith("+") or line.startswith("-"):
                    change_found = True
                    context_lines.append(line)
                elif change_found and len(context_lines) < max_context_lines * 2:
                    context_lines.append(line)
                elif not change_found and len(context_lines) < max_context_lines:
                    context_lines.append(line)
                else:
                    # Keep the last few lines as context
                    if len(context_lines) >= max_context_lines:
                        # Remove the oldest context line
                        context_lines.pop(0)
                    context_lines.append(line)

            if context_lines and any(line for line in context_lines):
                changes.append("\n".join(line for line in context_lines if line))

        # Add file summary to result if within max files limit
        if file_count <= max_files:
            file_summary = {
                "name": file_name,
                "operation": operation,
            }

            # Only include non-zero values
            if lines_added > 0:
                file_summary["lines_added"] = lines_added  # type: ignore # TypedDict allows int value
            if lines_removed > 0:
                file_summary["lines_removed"] = lines_removed  # type: ignore # TypedDict allows int value
            if changes:
                file_summary["changes"] = changes[:max_context_lines]  # type: ignore # Limit number of change chunks

            # Make sure files exists and is a list
            if "files" in result and isinstance(result["files"], list):
                result["files"].append(file_summary)

    # Filter out zero values from stats and only add stats if there are non-zero values
    non_zero_stats = {key: value for key, value in stats_counter.items() if value > 0}
    if non_zero_stats:
        # Cast to meet type expectations
        result["stats"] = non_zero_stats

    # Create overall summary - only include non-zero values
    summary_parts = []

    if stats_counter["total_files_changed"] > 0:
        summary_parts.append(f"Changed {stats_counter['total_files_changed']} files")

    file_details = []
    if stats_counter["files_created"] > 0:
        file_details.append(f"{stats_counter['files_created']} created")
    if stats_counter["files_modified"] > 0:
        file_details.append(f"{stats_counter['files_modified']} modified")
    if stats_counter["files_deleted"] > 0:
        file_details.append(f"{stats_counter['files_deleted']} deleted")

    if file_details:
        summary_parts.append(f"{', '.join(file_details)}")

    line_details = []
    if stats_counter["lines_added"] > 0:
        line_details.append(f"added {stats_counter['lines_added']} lines")
    if stats_counter["lines_removed"] > 0:
        line_details.append(f"removed {stats_counter['lines_removed']} lines")

    if line_details:
        summary_parts.append(f"{', '.join(line_details)}")

    result["summary"] = ". ".join(summary_parts)

    return result


def _summarize_file_contents(
    content: str,
    max_context_lines: int,
    max_files: int,
    max_file_kb: int,
) -> Dict[str, Union[str, List[Dict[str, Union[str, int, List[str]]]], Dict[str, int]]]:
    """Summarize changes from raw file contents."""
    result: Dict[str, Union[str, List[Dict[str, Union[str, int, List[str]]]], Dict[str, int]]] = {
        "summary": "",
        "files": [],
        # stats will be added only if there are non-zero values
    }

    # Stats counter (will only include non-zero values in final result)
    stats_counter = {
        "total_files_changed": 0,
        "files_created": 0,
        "files_modified": 0,
        "files_deleted": 0,
        "lines_added": 0,
        "lines_removed": 0,
    }

    # Look for the "File contents after editing" pattern
    if "File contents after editing" in content:
        # Split by file markers
        file_pattern = r"--- (.*?) ---\n"

        # We don't need to extract file names here as they're extracted from file_sections

        # Reset the matcher
        file_sections = re.split(file_pattern, content)

        # First section is header, skip it
        if len(file_sections) > 0:
            file_sections = file_sections[1:]

        # Process content for each file
        file_count = 0
        for i in range(0, len(file_sections), 2):
            if i >= len(file_sections) or i + 1 >= len(file_sections):
                break

            file_name = file_sections[i]
            file_content = file_sections[i + 1]

            file_count += 1
            stats_counter["total_files_changed"] += 1

            # If file content indicates error or not found, handle accordingly
            if "(Error reading file)" in file_content or "(File not found)" in file_content:
                if "(File not found)" in file_content:
                    stats_counter["files_deleted"] += 1
                    operation = "deleted"
                else:
                    operation = "error"
                line_count = 0
            else:
                # Consider it as created or modified (created is more likely with this fallback)
                stats_counter["files_created"] += 1
                operation = "created"

                # Count lines as added
                line_count = file_content.count("\n")
                stats_counter["lines_added"] += line_count

            # Truncate content if too large
            if len(file_content) > max_file_kb * 1024:
                file_content = file_content[: max_file_kb * 1024] + "\n... (truncated)"

            # Extract a sample of the content
            content_lines = file_content.split("\n")
            sample_lines = content_lines[:max_context_lines]

            # Add file summary to result if within max files
            if file_count <= max_files:
                file_summary = {
                    "name": file_name,
                    "operation": operation,
                }

                # Only include non-zero values
                if line_count > 0:
                    file_summary["lines_added"] = line_count

                if sample_lines and any(sample_lines):
                    file_summary["changes"] = ["\n".join(sample_lines)]

                # Make sure files exists and is a list
                if "files" in result and isinstance(result["files"], list):
                    result["files"].append(file_summary)

    # Filter out zero values from stats and only add stats if there are non-zero values
    non_zero_stats = {key: value for key, value in stats_counter.items() if value > 0}
    if non_zero_stats:
        # Cast to meet type expectations
        result["stats"] = non_zero_stats

    # Create overall summary - only include non-zero values
    summary_parts = []

    if stats_counter["total_files_changed"] > 0:
        summary_parts.append(f"Changed {stats_counter['total_files_changed']} files")

    file_details = []
    if stats_counter["files_created"] > 0:
        file_details.append(f"{stats_counter['files_created']} created")
    if stats_counter["files_modified"] > 0:
        file_details.append(f"{stats_counter['files_modified']} modified")
    if stats_counter["files_deleted"] > 0:
        file_details.append(f"{stats_counter['files_deleted']} deleted")

    if file_details:
        summary_parts.append(f"{', '.join(file_details)}")

    line_details = []
    if stats_counter["lines_added"] > 0:
        line_details.append(f"added {stats_counter['lines_added']} lines")
    if stats_counter["lines_removed"] > 0:
        line_details.append(f"removed {stats_counter['lines_removed']} lines")

    if line_details:
        summary_parts.append(f"{', '.join(line_details)}")

    if not summary_parts:
        result["summary"] = "No file changes detected or could not parse file content format."
    else:
        result["summary"] = ". ".join(summary_parts)

    return result


def get_file_status_summary(
    relative_editable_files: List[str], working_dir: Optional[str] = None
) -> Dict[str, Union[str, bool, int, List[Dict[str, str]]]]:
    """
    Check file status without using git diff to detect new and modified files.

    Args:
        relative_editable_files: List of files to check
        working_dir: Working directory path

    Returns:
        Dictionary with file status information including an array of changed files
    """
    result: Dict[str, Union[str, bool, int, List[Dict[str, str]]]] = {
        "has_changes": False,
        "status_summary": "",
    }

    # Stats counters
    files_created = 0
    files_modified = 0
    changed_files = []  # Will only add to result if there are actual changes

    for file_path in relative_editable_files:
        # Get absolute path if working_dir is provided
        full_path = file_path
        if working_dir and not os.path.isabs(file_path):
            full_path = os.path.join(working_dir, file_path)

        # Check if file exists
        if os.path.exists(full_path):
            # Check if file is empty (likely new)
            file_size = os.path.getsize(full_path)
            if file_size == 0:
                files_created += 1
                changed_files.append(
                    {
                        "name": file_path,
                        "operation": "created",
                    }
                )
            else:
                # We can't reliably know if it was modified without git tracking,
                # so consider all non-empty files as modified
                files_modified += 1
                changed_files.append(
                    {
                        "name": file_path,
                        "operation": "modified",
                    }
                )

    # Update has_changes flag
    result["has_changes"] = files_created > 0 or files_modified > 0

    # Only add non-zero counts and files to the result if there are actual changes
    if result["has_changes"]:
        # Add files array only if there are actually changes
        if changed_files:
            result["files"] = changed_files

        # Add non-zero counts
        if files_created > 0:
            result["files_created"] = files_created
        if files_modified > 0:
            result["files_modified"] = files_modified

        # Create summary
        summary_parts = []

        if files_created > 0:
            created_names = [f["name"] for f in changed_files if f["operation"] == "created"]
            if len(created_names) <= 3:  # Only list if not too many
                summary_parts.append(f"{files_created} files created ({', '.join(created_names)})")
            else:
                summary_parts.append(f"{files_created} files created")

        if files_modified > 0:
            modified_names = [f["name"] for f in changed_files if f["operation"] == "modified"]
            if len(modified_names) <= 3:  # Only list if not too many
                summary_parts.append(f"{files_modified} files modified ({', '.join(modified_names)})")
            else:
                summary_parts.append(f"{files_modified} files modified")

        result["status_summary"] = "Filesystem changes detected: " + ", ".join(summary_parts)
    else:
        result["status_summary"] = "No filesystem changes detected in the specified files."

    return result
