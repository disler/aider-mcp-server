"""
Tests for architect mode functionality in aider_ai_code tool.
"""

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Import the modules to test
from aider_mcp_server.atoms.tools.aider_ai_code import (
    _configure_model,
    _setup_aider_coder,
    code_with_aider,
)


def test_configure_model_architect_mode():
    """Test that _configure_model correctly configures for architect mode."""
    # Test with both architect_mode and editor_model
    model = _configure_model(
        "gemini/gemini-2.5-pro-exp-03-25",
        editor_model="gemini/gemini-2.5-flash-preview-04-17",
        architect_mode=True,
    )
    # Since we can't test actual models in unit tests, we're just checking
    # that the return type is correct
    assert model is not None

    # Test with architect_mode but no editor_model (should use same model for both)
    model = _configure_model(
        "gemini/gemini-2.5-pro-exp-03-25",
        architect_mode=True,
    )
    assert model is not None

    # Test with standard model (no architect mode)
    model = _configure_model("gemini/gemini-2.5-pro-exp-03-25")
    assert model is not None


@patch("aider_mcp_server.atoms.tools.aider_ai_code.Coder")
def test_setup_aider_coder_architect_mode(mock_coder):
    """Test that _setup_aider_coder correctly configures the Coder with architect mode."""
    # Setup mock
    mock_coder.create.return_value = MagicMock()
    mock_model = MagicMock()
    mock_model.commit_message_models.return_value = []

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with architect mode
        _setup_aider_coder(
            mock_model,
            temp_dir,
            [temp_dir + "/file.py"],
            [temp_dir + "/readonly.py"],
            architect_mode=True,
            auto_accept_architect=True,
        )

        # Check that Coder.create was called with architect-specific parameters
        call_args = mock_coder.create.call_args
        assert call_args is not None
        kwargs = call_args[1]

        # When architect_mode is True, it activates architect mode
        assert kwargs["edit_format"] == "architect"

        # Auto-accept should be enabled when specified
        assert kwargs["auto_accept_architect"] is True

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test without architect mode
        _setup_aider_coder(
            mock_model,
            temp_dir,
            [temp_dir + "/file.py"],
            [temp_dir + "/readonly.py"],
            architect_mode=False,
        )

        # Check that Coder.create was called without architect parameters
        call_args = mock_coder.create.call_args
        assert call_args is not None
        kwargs = call_args[1]

        # When architect_mode is False, it should not activate architect mode
        assert kwargs["edit_format"] is None
        assert kwargs["auto_accept_architect"] is True  # Default is True when not in architect mode

    # Test default behavior (no architect mode)
    with tempfile.TemporaryDirectory() as temp_dir:
        _setup_aider_coder(
            mock_model,
            temp_dir,
            [temp_dir + "/file.py"],
            [],
        )

        call_args = mock_coder.create.call_args
        assert call_args is not None
        kwargs = call_args[1]
        assert kwargs["edit_format"] is None
        assert kwargs["auto_accept_architect"] is True  # Default is True when not in architect mode


@pytest.mark.asyncio
@patch("aider_mcp_server.atoms.tools.aider_ai_code._configure_model")
@patch("aider_mcp_server.atoms.tools.aider_ai_code._setup_aider_coder")
@patch("aider_mcp_server.atoms.tools.aider_ai_code._run_aider_session")
async def test_code_with_aider_architect_mode(mock_run_aider, mock_setup_coder, mock_configure_model):
    """Test code_with_aider function with architect mode parameters."""
    # Setup mocks
    mock_model = MagicMock()
    mock_configure_model.return_value = mock_model

    mock_coder = MagicMock()
    mock_setup_coder.return_value = mock_coder

    mock_run_aider.return_value = {
        "success": True,
        "changes_summary": {"summary": "Some diff content"},
        "file_status": {"has_changes": True, "status_summary": "Changes detected"},
        "is_cached_diff": False,
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        # Call with architect mode parameters
        result = await code_with_aider(
            ai_coding_prompt="Create a calculator",
            relative_editable_files=["file.py"],
            relative_readonly_files=["readonly.py"],
            model="gemini/gemini-2.5-pro-exp-03-25",
            working_dir=temp_dir,
            architect_mode=True,
            editor_model="gemini/gemini-2.5-flash-preview-04-17",
            auto_accept_architect=True,
        )

        # Check that _configure_model was called with architect mode params
        mock_configure_model.assert_called_once_with(
            "gemini/gemini-2.5-pro-exp-03-25",
            "gemini/gemini-2.5-flash-preview-04-17",
            True,
        )

        # Check that _setup_aider_coder was called with architect mode
        call_args = mock_setup_coder.call_args
        assert call_args is not None
        args, kwargs = call_args
        # It's called with positional arguments
        assert len(args) == 6
        assert args[4] is True  # architect_mode (0-based index 4 is 5th argument)
        assert args[5] is True  # auto_accept_architect (0-based index 5 is 6th argument)

        # Check that the result is correct
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert "changes_summary" in result_data
        assert result_data["changes_summary"]["summary"] == "Some diff content"


@pytest.mark.asyncio
@patch("aider_mcp_server.atoms.tools.aider_ai_code._configure_model")
@patch("aider_mcp_server.atoms.tools.aider_ai_code._setup_aider_coder")
@patch("aider_mcp_server.atoms.tools.aider_ai_code._run_aider_session")
async def test_code_with_aider_default_mode(mock_run_aider, mock_setup_coder, mock_configure_model):
    """Test code_with_aider function with default (non-architect) mode."""
    # Setup mocks
    mock_model = MagicMock()
    mock_configure_model.return_value = mock_model

    mock_coder = MagicMock()
    mock_setup_coder.return_value = mock_coder

    mock_run_aider.return_value = {
        "success": True,
        "changes_summary": {"summary": "Some diff content"},
        "file_status": {"has_changes": True, "status_summary": "Changes detected"},
        "is_cached_diff": False,
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        # Call with default parameters (no architect mode)
        result = await code_with_aider(
            ai_coding_prompt="Create a calculator",
            relative_editable_files=["file.py"],
            model="gemini/gemini-2.5-pro-exp-03-25",
            working_dir=temp_dir,
        )

        # Check that _configure_model was called without architect mode
        mock_configure_model.assert_called_once_with(
            "gemini/gemini-2.5-pro-exp-03-25",
            None,
            False,
        )

        # Check that _setup_aider_coder was called without architect mode
        call_args = mock_setup_coder.call_args
        assert call_args is not None
        args, kwargs = call_args
        # It's called with positional arguments
        assert len(args) == 6
        assert args[4] is False  # architect_mode (0-based index 4 is 5th argument)
        assert args[5] is True  # auto_accept_architect (0-based index 5 is 6th argument)

        # Check that the result is correct
        result_data = json.loads(result)
        assert result_data["success"] is True
        assert "changes_summary" in result_data
        assert result_data["changes_summary"]["summary"] == "Some diff content"
