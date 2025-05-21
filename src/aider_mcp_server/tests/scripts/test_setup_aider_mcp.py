"""Tests for the setup_aider_mcp script."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from aider_mcp_server.setup_aider_mcp import (
    find_aider_mcp_server,
    is_git_repo,
    read_env_file,
    get_available_models,
    create_mcp_config,
    main,
)


def test_is_git_repo():
    """Test the is_git_repo function."""
    # The current directory should be a git repo for the tests to work
    current_dir = os.getcwd()
    assert is_git_repo(current_dir), "Expected current directory to be a git repo"
    
    # A temporary directory should not be a git repo
    with tempfile.TemporaryDirectory() as temp_dir:
        assert not is_git_repo(temp_dir), "Expected temp directory not to be a git repo"


def test_find_aider_mcp_server():
    """Test the find_aider_mcp_server function."""
    # When a valid path is given
    with mock.patch("pathlib.Path.exists", return_value=True):
        with mock.patch.object(Path, "__truediv__", return_value=Path("/fake/path")):
            result = find_aider_mcp_server("/some/path")
            assert result == "/some/path", "Expected to return the provided path"
    
    # When an invalid path is given
    with mock.patch("pathlib.Path.exists", return_value=False):
        result = find_aider_mcp_server("/invalid/path")
        assert result is None, "Expected None for invalid path"


def test_read_env_file():
    """Test the read_env_file function."""
    # Create a temporary env file
    with tempfile.TemporaryDirectory() as temp_dir:
        env_path = Path(temp_dir) / ".env"
        with open(env_path, "w") as f:
            f.write("TEST_KEY=test_value\n")
            f.write("# This is a comment\n")
            f.write("ANOTHER_KEY=another_value\n")
        
        # Mock os.environ to avoid interference from the actual environment
        with mock.patch.dict(os.environ, {}, clear=True):
            result = read_env_file(temp_dir)
            # Check if our test keys are in the result
            assert "TEST_KEY" in result and result["TEST_KEY"] == "test_value", "TEST_KEY not found or incorrect value"
            assert "ANOTHER_KEY" in result and result["ANOTHER_KEY"] == "another_value", "ANOTHER_KEY not found or incorrect value"


def test_get_available_models():
    """Test the get_available_models function."""
    # We need to mock the DEFAULT_MODELS dictionary to ensure test stability
    with mock.patch("aider_mcp_server.setup_aider_mcp.DEFAULT_MODELS", {
        "gemini": "gemini/test-model",
        "openai": "openai/test-model",
        "anthropic": "anthropic/test-model"
    }):
        env_vars = {
            "GEMINI_API_KEY": "fake-key",
            "OPENAI_API_KEY": "your_openai_api_key_here",  # Should now be accepted as valid
            "ANTHROPIC_API_KEY": "fake-key",
        }
        
        result = get_available_models(env_vars)
        assert "gemini" in result, "Expected gemini provider to be available"
        assert "anthropic" in result, "Expected anthropic provider to be available"
        assert "openai" in result, "Expected openai provider to be available"


def test_create_mcp_config():
    """Test the create_mcp_config function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        aider_dir = "/path/to/aider"
        model = "test-model"
        target_dir = "/target/dir"
        
        # Mock user input for overwrite confirmation (not needed first time)
        with mock.patch("builtins.input", return_value="y"):
            config_path = Path(temp_dir) / ".mcp.json"
            result = create_mcp_config(config_path, aider_dir, model, target_dir)
            assert result is True, "Expected config creation to succeed"
            
            # Check the config file was created
            assert config_path.exists(), "Expected config file to be created"
            
            # Check the content of the config file
            with open(config_path, "r") as f:
                config = json.load(f)
                assert "mcpServers" in config, "Expected mcpServers in config"
                assert "aider-mcp-server" in config["mcpServers"], "Expected aider-mcp-server in config"
                assert config["mcpServers"]["aider-mcp-server"]["args"][5] == model, "Expected correct model in config"


def test_main_non_git_repository_error(capsys):
    """Test that the correct error message is displayed when run in a non-git repository."""
    # Mock command line arguments for a non-git directory
    test_args = ["setup_aider_mcp.py"]
    
    with mock.patch("sys.argv", test_args):
        # Mock is_git_repo to return False
        with mock.patch("aider_mcp_server.setup_aider_mcp.is_git_repo", return_value=False):
            # Mock Path.cwd to return a test directory
            test_dir = "/test/non-git-dir"
            with mock.patch("pathlib.Path.cwd", return_value=Path(test_dir)):
                # Run main and expect it to exit with code 1
                exit_code = main()
                assert exit_code == 1
                
                # Capture the output and check for the expected error message
                captured = capsys.readouterr()
                
                # Verify the error message contains all expected components
                assert f"Error: {test_dir} is not a git repository." in captured.out
                assert "git init" in captured.out
                assert "setup-aider-mcp setup" in captured.out
                assert "Alternatively, run this command from within an existing git repository" in captured.out