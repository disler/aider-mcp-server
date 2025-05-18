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
        
        result = read_env_file(temp_dir)
        assert result == {
            "TEST_KEY": "test_value",
            "ANOTHER_KEY": "another_value",
        }, "Expected to read key-value pairs from env file"


def test_get_available_models():
    """Test the get_available_models function."""
    env_vars = {
        "GEMINI_API_KEY": "fake-key",
        "OPENAI_API_KEY": "your_openai_api_key_here",  # Should be ignored
        "ANTHROPIC_API_KEY": "fake-key",
    }
    
    result = get_available_models(env_vars)
    assert "gemini" in result, "Expected gemini provider to be available"
    assert "anthropic" in result, "Expected anthropic provider to be available"
    assert "openai" not in result, "Expected openai provider to be unavailable (placeholder value)"


def test_create_mcp_config():
    """Test the create_mcp_config function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        aider_dir = "/path/to/aider"
        model = "test-model"
        
        # Mock user input for overwrite confirmation (not needed first time)
        with mock.patch("builtins.input", return_value="y"):
            result = create_mcp_config(temp_dir, aider_dir, model)
            assert result is True, "Expected config creation to succeed"
            
            # Check the config file was created
            config_path = Path(temp_dir) / ".mcp.json"
            assert config_path.exists(), "Expected config file to be created"
            
            # Check the content of the config file
            with open(config_path, "r") as f:
                config = json.load(f)
                assert "mcpServers" in config, "Expected mcpServers in config"
                assert "aider-mcp-server" in config["mcpServers"], "Expected aider-mcp-server in config"
                assert config["mcpServers"]["aider-mcp-server"]["args"][5] == model, "Expected correct model in config"