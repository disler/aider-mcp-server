"""
Type stubs for the aider library to be used with mypy.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Stub for aider.repo module
class Repo:
    """Stub for the aider.repo.Repo class."""
    
    def __init__(self, 
                 repo_path: str, 
                 main_branch: Optional[str] = None, 
                 io: Optional[Any] = None, 
                 verbose: bool = False) -> None:
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
    
    def __init__(self, 
                 repo: Optional[Repo] = None, 
                 model: Optional[str] = None, 
                 **kwargs: Any) -> None:
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