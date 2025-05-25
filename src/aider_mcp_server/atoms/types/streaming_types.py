"""
Phase 1.3 streaming data types for lightweight real-time monitoring.

These types are designed to provide maximum insight with minimal payload size,
avoiding the bottleneck of streaming full diffs while maintaining rich progress tracking.
"""

from typing import Dict, List, Optional, TypedDict, Union
from enum import Enum


class ChangeType(Enum):
    """Types of changes that can be made to files."""
    CREATED = "created"
    MODIFIED = "modified" 
    DELETED = "deleted"
    RENAMED = "renamed"


class OperationType(Enum):
    """Types of operations AIDER can perform."""
    ANALYZING = "analyzing"
    PLANNING = "planning"
    GENERATING = "generating"
    APPLYING = "applying"
    VALIDATING = "validating"
    COMPLETING = "completing"


class FileChangesSummary(TypedDict):
    """Lightweight summary of changes to a single file."""
    file_path: str
    change_type: str  # ChangeType enum value
    lines_added: int
    lines_removed: int
    lines_modified: int
    change_description: str  # Brief human-readable summary
    estimated_impact: str    # "low", "medium", "high"


class AiderChangesSummary(TypedDict):
    """Comprehensive but lightweight summary of all changes in a session."""
    session_id: str
    timestamp: float
    total_files_changed: int
    files_created: int
    files_modified: int
    files_deleted: int
    total_lines_added: int
    total_lines_removed: int
    file_summaries: List[FileChangesSummary]
    change_categories: List[str]  # e.g., ["bug_fix", "feature_addition", "refactoring"]
    estimated_complexity: str     # "simple", "moderate", "complex"
    

class AiderFileProgress(TypedDict):
    """Progress tracking for individual file processing."""
    session_id: str
    file_path: str
    status: str              # "queued", "processing", "completed", "failed"
    progress_percentage: int # 0-100
    current_operation: str   # OperationType enum value
    estimated_completion_ms: Optional[int]
    error_message: Optional[str]


class AiderOperationStatus(TypedDict):
    """Current operation status for real-time monitoring."""
    session_id: str
    operation: str           # OperationType enum value
    status: str              # "starting", "in_progress", "completed", "failed"
    progress_percentage: int # Overall session progress 0-100
    current_step: str        # Human-readable current step
    files_remaining: int
    estimated_completion_ms: Optional[int]
    performance_metrics: Dict[str, Union[int, float, str]]  # tokens/sec, etc.


# Type aliases for convenience
StreamingEventData = Union[AiderChangesSummary, AiderFileProgress, AiderOperationStatus]