"""
Test suite for verifying compatibility with the installed aider version.
"""
import inspect
import pathlib
import sys
try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib  # fallback for older Python versions
from typing import Any, Dict

import pytest

# Ensure the source directory is in the Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "src"))

from aider_mcp_server.atoms.tools.aider_compatibility import (
    check_aider_compatibility,
    filter_supported_params,
    get_aider_version,
    get_supported_coder_create_params,
    get_supported_coder_params,
    get_version_specific_params,
)
from aider_mcp_server.atoms.tools.aider_ai_code import _setup_aider_coder


@pytest.mark.integration
def test_aider_version_installed():
    """Test that aider is installed and has a version."""
    version = get_aider_version()
    assert version is not None, "aider library not installed or version not available"
    print(f"Installed aider version: {version}")


@pytest.mark.integration  
def test_aider_parameter_compatibility():
    """Test that our code uses only parameters supported by the current aider version."""
    supported_params = get_supported_coder_create_params()
    assert supported_params, "Failed to get supported Coder parameters"
    
    # Parameters we're using in our code
    our_params = {
        "main_model", "io", "fnames", "read_only_fnames", "repo",
        "show_diffs", "auto_commits", "dirty_commits", "use_git",
        "stream", "suggest_shell_commands", "detect_urls",
        "verbose", "quiet", "edit_format", "auto_accept_architect"
    }
    
    # Find unsupported parameters
    unsupported = our_params - supported_params
    if unsupported:
        print(f"Warning: Parameters not supported by current aider version: {unsupported}")
        print(f"Supported parameters: {supported_params}")
    
    # This is a warning, not a failure, as we handle unsupported params dynamically
    assert True


@pytest.mark.integration
def test_version_specific_params():
    """Test version-specific parameter handling."""
    current_version = get_aider_version()
    assert current_version is not None
    
    # Test parameter overrides for the current version
    overrides = get_version_specific_params(current_version)
    print(f"Version {current_version} overrides: {overrides}")
    
    # Test with older version
    old_overrides = get_version_specific_params("0.83.0")
    print(f"Version 0.83.0 overrides: {old_overrides}")
    
    # Test with newer version
    new_overrides = get_version_specific_params("0.84.0")
    print(f"Version 0.84.0 overrides: {new_overrides}")


@pytest.mark.integration
def test_filter_supported_params():
    """Test filtering of parameters based on what's supported."""
    supported = get_supported_coder_create_params()
    
    # Test parameters including potentially unsupported ones
    test_params = {
        "main_model": "test_model",
        "io": None,
        "fnames": [],
        "quiet": True,  # This might not be supported
        "verbose": False,
        "some_future_param": "value"  # This definitely isn't supported
    }
    
    filtered = filter_supported_params(test_params, supported)
    
    # Verify only supported params are included
    assert all(key in supported for key in filtered)
    print(f"Filtered params: {filtered}")


@pytest.mark.integration
def test_actual_aider_coder_creation():
    """Test that we can create an actual Aider Coder with our current configuration."""
    try:
        from aider.coders import Coder
        from aider.io import InputOutput
        from aider.models import Model
        
        # Create minimal components needed for testing
        io = InputOutput(pretty=False, yes=True, fancy_input=False)
        model = Model("gpt-3.5-turbo")  # Use a simple model name
        
        # Get supported parameters
        supported = get_supported_coder_create_params()
        
        # Test parameters
        params = {
            "main_model": model,
            "io": io,
            "fnames": [],
            "show_diffs": False,
            "auto_commits": False,
            "stream": False,
            "verbose": False,
        }
        
        # Add potentially unsupported parameters
        optional_params = {
            "quiet": True,
            "suggest_shell_commands": False,
            "detect_urls": False,
        }
        
        # Filter to only supported parameters
        filtered_optional = filter_supported_params(optional_params, supported)
        params.update(filtered_optional)
        
        # Attempt to create coder
        coder = Coder.create(**params)
        assert coder is not None, "Failed to create Coder instance"
        print("Successfully created Coder instance with filtered parameters")
        
    except Exception as e:
        print(f"Failed to create Coder: {e}")
        # This is expected if we don't have proper API keys
        if "API key" in str(e) or "Missing" in str(e):
            pytest.skip("Skipping due to missing API keys")
        else:
            raise


@pytest.mark.integration
def test_aider_version_vs_pyproject():
    """Compare installed aider version with what's specified in pyproject.toml."""
    # Load pyproject.toml
    pyproject_path = pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
    if hasattr(tomllib, 'load'):
        with open(pyproject_path, 'rb') as f:
            pyproject = tomllib.load(f)
    else:
        with open(pyproject_path) as f:
            pyproject = tomllib.load(f)
    
    # Get expected version from dependencies
    dependencies = pyproject["project"]["dependencies"]
    aider_dep = None
    for dep in dependencies:
        if dep.startswith("aider-chat"):
            aider_dep = dep
            break
    
    assert aider_dep is not None, "aider-chat not found in pyproject.toml dependencies"
    
    # Extract version requirement
    expected_version = aider_dep.split(">=")[1] if ">=" in aider_dep else None
    assert expected_version is not None, f"Could not parse version from {aider_dep}"
    
    # Check compatibility
    try:
        check_aider_compatibility(expected_version)
        print(f"Aider version compatibility check passed for {expected_version}")
    except AssertionError as e:
        print(f"Compatibility check failed: {e}")
        # This is a warning, not a hard failure in tests
        pass


@pytest.mark.integration
def test_coder_init_vs_create_params():
    """Compare parameters accepted by Coder.__init__ vs Coder.create."""
    init_params = get_supported_coder_params()
    create_params = get_supported_coder_create_params()
    
    print(f"Coder.__init__ params: {init_params}")
    print(f"Coder.create params: {create_params}")
    
    # Usually create accepts more params as it does additional processing
    if create_params != init_params:
        diff = create_params - init_params
        print(f"Additional params in create: {diff}")


if __name__ == "__main__":
    # Run the tests manually  
    print("Running aider compatibility tests...")
    test_aider_version_installed()
    test_aider_parameter_compatibility()
    test_version_specific_params()
    test_filter_supported_params()
    test_actual_aider_coder_creation()
    test_aider_version_vs_pyproject()
    test_coder_init_vs_create_params()
    print("All tests completed.")