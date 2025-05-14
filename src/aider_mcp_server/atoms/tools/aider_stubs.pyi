"""
Type stubs for the aider library to be used with mypy.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union

# Response dictionary from aider processing
class ResponseDict(TypedDict, total=False):
    success: bool
    diff: str
    is_cached_diff: bool
    rate_limit_info: Optional[Dict[str, Union[bool, int, str, None]]]

# Stub for aider.repo module
class Repo:
    """Stub for the aider.repo.Repo class."""

    def __init__(
        self,
        repo_path: str,
        main_branch: Optional[str] = None,
        io: Optional[Any] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the Repo object."""
        pass

    def abs_path(self, path: str) -> str:
        """Convert a repo-relative path to an absolute path."""
        pass

    def rel_path(self, path: str) -> str:
        """Convert an absolute path to a repo-relative path."""
        pass

    def is_in_repo(self, path: str) -> bool:
        """Check if a path is in the repo."""
        pass

    def is_repo_path(self, path: str) -> bool:
        """Check if a path is valid in the repo."""
        pass

    def is_code_file(self, path: str) -> bool:
        """Check if a path points to a code file."""
        pass

    def read_file(self, path: str) -> str:
        """Read a file from the repo."""
        pass

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the repo."""
        pass

# Stubs for aider.coders
class BaseCoder:
    """Stub for the aider.coders.base_coder.BaseCoder class."""

    def __init__(
        self, repo: Optional[Repo] = None, model: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Initialize the BaseCoder."""
        pass

    async def run_code_edit(self, prompt: str, files: List[str]) -> Dict[str, Any]:
        """Run the code edit operation."""
        pass

class PromptBasedCoder(BaseCoder):
    """Stub for PromptBasedCoder."""

    pass

class Coder(PromptBasedCoder):
    """Stub for the main Coder class."""

    pass

# Stubs for other aider modules as needed
class Commands:
    """Stub for aider.commands.Commands."""

    pass

class Models:
    """Stub for aider.models.Models."""

    @staticmethod
    def get_model_names() -> List[str]:
        """Get available model names."""
        pass

async def code_with_aider(
    ai_coding_prompt: str,
    relative_editable_files: List[str],
    relative_readonly_files: Optional[List[str]] = None,
    model: str = "gemini/gemini-2.5-flash-preview-04-17",
    working_dir: Optional[str] = None,
    use_diff_cache: bool = True,
    clear_cached_for_unchanged: bool = True,
    architect_mode: bool = False,
    editor_model: Optional[str] = None,
    auto_accept_architect: bool = True,
) -> str: ...
async def init_diff_cache() -> None: ...
async def shutdown_diff_cache() -> None: ...
