"""
Module for summarizing code changes to reduce token usage.
"""

import os
import re
from typing import Any, Dict, List, Optional, TypedDict, Union


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
) -> Dict[str, Any]:  # Using Any to avoid complex nested type definitions
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


def _determine_file_operation_and_stats(file_diff: str, stats_counter: Dict[str, int]) -> str:
    """Determines file operation and updates stats."""
    if "new file mode" in file_diff:
        operation = "created"
        stats_counter["files_created"] += 1
    elif "deleted file mode" in file_diff:
        operation = "deleted"
        stats_counter["files_deleted"] += 1
    else:
        operation = "modified"
        stats_counter["files_modified"] += 1
    return operation


def _count_added_removed_lines(diff_lines: List[str], stats_counter: Dict[str, int]) -> tuple[int, int]:
    """Counts added and removed lines and updates stats."""
    lines_added = 0
    lines_removed = 0
    for line in diff_lines:
        if line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1
    stats_counter["lines_added"] += lines_added
    stats_counter["lines_removed"] += lines_removed
    return lines_added, lines_removed


def _extract_changes_from_chunk(chunk: str, max_context_lines: int) -> List[str]:
    """Extracts context lines from a diff chunk."""
    context_lines: List[str] = []
    change_found = False
    # Ensure lines are split correctly and iterated
    for line_content in chunk.splitlines(): # Use splitlines to handle different line endings
        line = line_content # Keep original line for appending
        if line.startswith("+") or line.startswith("-"):
            change_found = True
            context_lines.append(line)
        elif change_found: # If a change was found, keep adding context lines up to the limit
            if len(context_lines) < (max_context_lines * 2) + sum(1 for l in context_lines if l.startswith("+") or l.startswith("-")): # Heuristic
                context_lines.append(line)
            # else: if context is full after a change, stop adding
        elif not change_found: # If no change found yet, manage pre-context
            if len(context_lines) >= max_context_lines:
                context_lines.pop(0) # Remove oldest pre-context line
            context_lines.append(line)
        # This logic aims to keep `max_context_lines` before and after the change block.
        # A more robust approach might involve identifying change blocks first.
    return context_lines


def _process_diff_chunks(file_diff: str, max_context_lines: int, max_file_kb: int) -> List[str]:
    """Processes diff chunks to extract changes."""
    changes: List[str] = []
    chunk_pattern = r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*" # Include rest of line
    chunk_matches = list(re.finditer(chunk_pattern, file_diff))

    for i, match in enumerate(chunk_matches):
        # Find the start of the next chunk header, or end of file_diff
        chunk_content_end = chunk_matches[i + 1].start() if i + 1 < len(chunk_matches) else len(file_diff)
        
        # Extract the content part of the chunk (lines after the @@ line)
        chunk_content_with_header = file_diff[match.start():chunk_content_end]
        # Remove the header line itself from the content to be processed for context
        header_line_end = chunk_content_with_header.find('\n') + 1 if chunk_content_with_header.find('\n') != -1 else len(chunk_content_with_header)
        chunk_actual_content = chunk_content_with_header[header_line_end:]


        if len(chunk_actual_content) > max_file_kb * 1024:
            chunk_actual_content = chunk_actual_content[: max_file_kb * 1024] + "\n... (truncated)"

        context_lines = _extract_changes_from_chunk(chunk_actual_content, max_context_lines)
        if context_lines and any(line.strip() for line in context_lines): # Check for non-empty lines
            # Add the @@ header to the context for clarity
            header = match.group(0)
            changes.append(header + "\n" + "\n".join(line for line in context_lines if line.strip()))
    return changes


def _build_git_diff_summary_string(stats_counter: Dict[str, int]) -> str:
    """Builds the summary string for git diff."""
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
        summary_parts.append(f"({', '.join(file_details)})") # Wrap in parentheses

    line_details = []
    if stats_counter["lines_added"] > 0:
        line_details.append(f"added {stats_counter['lines_added']} lines")
    if stats_counter["lines_removed"] > 0:
        line_details.append(f"removed {stats_counter['lines_removed']} lines")

    if line_details:
        summary_parts.append(f"with {', '.join(line_details)}") # "with" for line changes

    return ". ".join(filter(None, summary_parts)) # Filter ensures no empty strings from missing parts


def _process_git_diff_file_section(
    file_diff: str,
    stats_counter: Dict[str, int],
    max_context_lines: int,
    max_file_kb: int,
) -> Optional[Dict[str, Any]]:
    """Processes a single file section from a git diff."""
    diff_lines = file_diff.splitlines()
    if not diff_lines:
        return None

    file_name = _extract_filename_from_git_diff(diff_lines[0])
    if not file_name:
        return None

    stats_counter["total_files_changed"] += 1
    operation = _determine_file_operation_and_stats(file_diff, stats_counter)
    lines_added, lines_removed = _count_added_removed_lines(diff_lines, stats_counter)
    changes = _process_diff_chunks(file_diff, max_context_lines, max_file_kb)

    file_summary: Dict[str, Any] = {
        "name": file_name,
        "operation": operation,
    }
    if lines_added > 0:
        file_summary["lines_added"] = lines_added
    if lines_removed > 0:
        file_summary["lines_removed"] = lines_removed
    if changes:
        file_summary["changes"] = changes
    return file_summary


def _process_git_diff_file_sections(
    file_sections: List[str],
    stats_counter: Dict[str, int],
    result_files_list: List[Dict[str, Any]],
    max_files: int,
    max_context_lines: int,
    max_file_kb: int,
) -> None:
    """Processes all file sections from a git diff."""
    file_count = 0
    for file_diff in file_sections:
        file_summary_entry = _process_git_diff_file_section(
            file_diff, stats_counter, max_context_lines, max_file_kb
        )
        if file_summary_entry:
            file_count += 1
            if file_count <= max_files:
                result_files_list.append(file_summary_entry)


def _summarize_git_diff(
    diff_content: str,
    max_context_lines: int,
    max_files: int,
    max_file_kb: int,
) -> Dict[str, Any]:
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
    if "files" in result and isinstance(result["files"], list):
        _process_git_diff_file_sections(
            file_sections,
            stats_counter,
            result["files"],
            max_files,
            max_context_lines,
            max_file_kb,
        )

    non_zero_stats: Dict[str, int] = {key: value for key, value in stats_counter.items() if value > 0}
    if non_zero_stats:
        result["stats"] = non_zero_stats # type: ignore[assignment]

    result["summary"] = _build_git_diff_summary_string(stats_counter)
    return result


def _process_single_file_content(
    file_name: str,
    file_content: str,
    stats_counter: Dict[str, int],
    max_file_kb: int,
    max_context_lines: int,
) -> Optional[Dict[str, Any]]:
    """Processes a single file's content for summarization."""
    stats_counter["total_files_changed"] += 1
    operation: str
    line_count = 0

    if "(Error reading file)" in file_content or "(File not found)" in file_content:
        if "(File not found)" in file_content:
            stats_counter["files_deleted"] += 1
            operation = "deleted"
        else:
            operation = "error" # File exists but error reading
    else:
        # For non-git diff, assume new/changed content is 'created' or 'modified'
        # Defaulting to 'created' as it's a common case for this function's input.
        stats_counter["files_created"] += 1 # Or files_modified, depending on context
        operation = "created" # Or "modified"
        line_count = file_content.count("\n") + 1 # Count lines
        stats_counter["lines_added"] += line_count


    if len(file_content) > max_file_kb * 1024:
        file_content = file_content[: max_file_kb * 1024] + "\n... (truncated)"

    content_lines = file_content.split("\n")
    sample_lines = content_lines[:max_context_lines]

    file_summary: Dict[str, Any] = {
        "name": file_name,
        "operation": operation,
    }
    if line_count > 0 and operation not in ["deleted", "error"]: # Only add lines_added if relevant
        file_summary["lines_added"] = line_count
    if sample_lines and any(s.strip() for s in sample_lines) and operation not in ["deleted", "error"]:
        file_summary["changes"] = ["\n".join(s for s in sample_lines if s.strip())]
    return file_summary


def _build_file_content_summary_string(stats_counter: Dict[str, int]) -> str:
    """Builds the summary string for file contents."""
    summary_parts = []
    if stats_counter["total_files_changed"] > 0:
        summary_parts.append(f"Processed {stats_counter['total_files_changed']} files")

    file_details = []
    if stats_counter["files_created"] > 0:
        file_details.append(f"{stats_counter['files_created']} created/modified")
    if stats_counter["files_deleted"] > 0:
        file_details.append(f"{stats_counter['files_deleted']} not found")
    # files_modified is not explicitly tracked here, but created covers new/changed

    if file_details:
        summary_parts.append(f"({', '.join(file_details)})")

    line_details = []
    if stats_counter["lines_added"] > 0:
        line_details.append(f"with approx. {stats_counter['lines_added']} lines")
    # lines_removed is not tracked in this function

    if line_details:
        summary_parts.append(" ".join(line_details))


    if not summary_parts:
        return "No file changes detected or could not parse file content format."
    return ". ".join(filter(None, summary_parts))


def _summarize_file_contents(
    content: str,
    max_context_lines: int,
    max_files: int,
    max_file_kb: int,
) -> Dict[str, Any]:
    """Summarize changes from raw file contents."""
    result: Dict[str, Any] = {
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
            file_name = file_sections[i].strip()
            file_content = file_sections[i + 1].strip()
            file_count += 1

            if file_count > max_files:
                # Optionally, log that max_files limit was reached
                break

            file_summary_entry = _process_single_file_content(
                file_name, file_content, stats_counter, max_file_kb, max_context_lines
            )
            if file_summary_entry:
                if "files" in result and isinstance(result["files"], list):
                    result["files"].append(file_summary_entry)

    non_zero_stats: Dict[str, int] = {key: value for key, value in stats_counter.items() if value > 0}
    if non_zero_stats:
        result["stats"] = non_zero_stats

    result["summary"] = _build_file_content_summary_string(stats_counter)
    return result


def _check_single_file_status(
    file_path: str, working_dir: Optional[str]
) -> Optional[FileStatusEntry]:
    """Checks the status of a single file."""
    full_path = file_path
    if working_dir and not os.path.isabs(file_path):
        full_path = os.path.join(working_dir, file_path)

    if os.path.exists(full_path):
        file_size = os.path.getsize(full_path)
        if file_size == 0: # Empty file often means new or cleared
            return {"name": file_path, "operation": "created"} # Or "empty"
        else: # Non-empty means modified or existing
            return {"name": file_path, "operation": "modified"}
    return None # File does not exist


def _build_file_status_summary_string(
    files_created: int, files_modified: int, changed_files: List[FileStatusEntry]
) -> str:
    """Builds the summary string for file status."""
    if not files_created and not files_modified:
        return "No filesystem changes detected in the specified files."

    summary_parts = []
    if files_created > 0:
        created_names = [f["name"] for f in changed_files if f["operation"] == "created"]
        # Limit listed names to avoid overly long summaries
        display_created_names = created_names[:3]
        etc_created = "..." if len(created_names) > 3 else ""
        summary_parts.append(f"{files_created} files created/empty ({', '.join(display_created_names)}{etc_created})")


    if files_modified > 0:
        modified_names = [f["name"] for f in changed_files if f["operation"] == "modified"]
        # Limit listed names
        display_modified_names = modified_names[:3]
        etc_modified = "..." if len(modified_names) > 3 else ""
        summary_parts.append(f"{files_modified} files modified/existing ({', '.join(display_modified_names)}{etc_modified})")

    return "Filesystem changes detected: " + ", ".join(summary_parts)


def get_file_status_summary(relative_editable_files: List[str], working_dir: Optional[str] = None) -> Dict[str, Any]:
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
    changed_file_entries: List[FileStatusEntry] = []

    for file_path in relative_editable_files:
        status_entry = _check_single_file_status(file_path, working_dir)
        if status_entry:
            changed_file_entries.append(status_entry)
            if status_entry["operation"] == "created":
                files_created += 1
            elif status_entry["operation"] == "modified":
                files_modified += 1
    
    has_any_changes = files_created > 0 or files_modified > 0
    result["has_changes"] = has_any_changes

    if has_any_changes:
        if changed_file_entries: # Ensure list is not empty before assigning
            result["files"] = changed_file_entries
        if files_created > 0:
            result["files_created"] = files_created
        if files_modified > 0:
            result["files_modified"] = files_modified
    
    result["status_summary"] = _build_file_status_summary_string(files_created, files_modified, changed_file_entries)
    return result
