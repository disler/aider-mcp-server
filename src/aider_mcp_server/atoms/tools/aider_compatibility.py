"""
Compatibility layer for handling different versions of aider library.
This module provides utilities to ensure compatibility across aider versions.
"""
import inspect
import logging
from typing import Any, Dict, Optional, Set

from packaging import version

logger = logging.getLogger(__name__)


def get_aider_version() -> Optional[str]:
    """Get the installed aider version."""
    try:
        import aider
        return getattr(aider, "__version__", None)
    except ImportError:
        return None


def get_supported_coder_params() -> Set[str]:
    """
    Get parameters supported by the current aider Coder implementation.
    
    Returns:
        Set of parameter names supported by the current Coder implementation.
    """
    try:
        from aider.coders import Coder
        sig = inspect.signature(Coder.__init__)
        return set(sig.parameters.keys()) - {"self"}  # Remove self parameter
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to get Coder parameters: {e}")
        return set()


def get_supported_coder_create_params() -> Set[str]:
    """
    Get parameters supported by the Coder.create method.
    
    Returns:
        Set of parameter names supported by the Coder.create method.
    """
    try:
        from aider.coders import Coder
        sig = inspect.signature(Coder.create)
        return set(sig.parameters.keys()) - {"cls"}  # Remove cls parameter
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to get Coder.create parameters: {e}")
        return set()


def filter_supported_params(params: Dict[str, Any], supported_params: Set[str]) -> Dict[str, Any]:
    """
    Filter parameters to include only those supported by the current implementation.
    
    Args:
        params: Dictionary of parameters to filter
        supported_params: Set of supported parameter names
    
    Returns:
        Filtered dictionary containing only supported parameters
    """
    filtered = {k: v for k, v in params.items() if k in supported_params}
    unsupported = set(params.keys()) - supported_params
    
    if unsupported:
        logger.warning(f"Removing unsupported parameters: {unsupported}")
    
    return filtered


def get_version_specific_params(aider_version_str: Optional[str]) -> Dict[str, Any]:
    """
    Get version-specific parameter overrides based on aider version.
    
    Args:
        aider_version_str: Aider version string or None
    
    Returns:
        Dictionary of version-specific parameter overrides
    """
    if not aider_version_str:
        return {}
    
    try:
        aider_version = version.parse(aider_version_str)
    except Exception as e:
        logger.error(f"Failed to parse aider version '{aider_version_str}': {e}")
        return {}
    
    overrides: Dict[str, Any] = {}
    
    # Based on actual inspection of aider 0.83.1, these parameters are not supported:
    # - quiet (not in Coder.__init__)
    # - edit_format (handled separately in create method)
    # - auto_accept_architect (present in Coder.__init__)
    
    # Parameters to remove for all versions (not supported in 0.83.1)
    params_to_remove = ["quiet"]
    
    for param in params_to_remove:
        overrides[param] = None  # Will be filtered out
    
    return overrides


def check_aider_compatibility(expected_version: str) -> None:
    """
    Check if the installed aider version matches the expected version.
    
    Args:
        expected_version: Expected aider version from pyproject.toml
    
    Raises:
        AssertionError: If the versions don't match or if parameters are incompatible
    """
    actual_version = get_aider_version()
    if not actual_version:
        raise AssertionError("aider library not installed")
    
    # Compare versions
    expected = version.parse(expected_version.lstrip(">="))
    actual = version.parse(actual_version)
    
    if actual < expected:
        raise AssertionError(
            f"Installed aider version {actual_version} is older than expected {expected_version}"
        )
    
    # Check parameter compatibility
    supported_params = get_supported_coder_create_params()
    our_params = {
        "main_model", "io", "fnames", "read_only_fnames", "repo",
        "show_diffs", "auto_commits", "dirty_commits", "use_git",
        "stream", "suggest_shell_commands", "detect_urls",
        "verbose", "quiet", "edit_format", "auto_accept_architect"
    }
    
    unsupported = our_params - supported_params
    if unsupported:
        logger.warning(f"Parameters not supported by aider {actual_version}: {unsupported}")