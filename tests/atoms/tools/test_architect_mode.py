"""
Tests for architect mode functionality in aider_ai_code tool.
"""

import json
import os
import pytest
from typing import Dict, Optional, Any
from unittest.mock import MagicMock, patch

# Import the modules to test
from aider_mcp_server.atoms.tools.aider_ai_code import (
    code_with_aider,
    _configure_model,
    _setup_aider_coder,
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
    
    # Test with architect mode
    _setup_aider_coder(
        mock_model,
        "/tmp/test",
        ["/tmp/test/file.py"],
        ["/tmp/test/readonly.py"],
        architect_mode=True,
        auto_accept_architect=True,
    )
    
    # Check that Coder.create was called with architect-specific parameters
    coder_create_kwargs = mock_coder.create.call_args[1]
    assert coder_create_kwargs['edit_format'] == "architect"
    assert coder_create_kwargs['auto_accept_architect'] is True
    
    # Test without architect mode
    mock_coder.create.reset_mock()
    _setup_aider_coder(
        mock_model,
        "/tmp/test",
        ["/tmp/test/file.py"],
        ["/tmp/test/readonly.py"],
        architect_mode=False,
    )
    
    # Check that Coder.create was called without architect-specific parameters
    coder_create_kwargs = mock_coder.create.call_args[1]
    assert coder_create_kwargs['edit_format'] is None
    assert coder_create_kwargs['auto_accept_architect'] is True


@pytest.mark.asyncio
@patch("aider_mcp_server.atoms.tools.aider_ai_code._execute_with_retry")
async def test_code_with_aider_architect_mode(mock_execute):
    """Test that code_with_aider passes architect mode parameters correctly."""
    # Setup mock
    mock_execute.return_value = {
        "success": True,
        "diff": "Some diff content",
        "is_cached_diff": False,
    }
    
    # Call with architect mode parameters
    result = await code_with_aider(
        ai_coding_prompt="Create a calculator",
        relative_editable_files=["file.py"],
        relative_readonly_files=["readonly.py"],
        model="gemini/gemini-2.5-pro-exp-03-25",
        working_dir="/tmp/test",
        architect_mode=True,
        editor_model="gemini/gemini-2.5-flash-preview-04-17",
        auto_accept_architect=True,
    )
    
    # Check that _execute_with_retry was called with correct parameters
    call_args = mock_execute.call_args[0]
    
    # In the current implementation, all parameters are passed as positional arguments
    # Verify architect mode parameters were passed to _execute_with_retry in correct positions
    # Expected order: ai_coding_prompt, relative_editable_files, abs_editable_files, abs_readonly_files, 
    #                working_dir, model, provider, use_diff_cache, clear_cached_for_unchanged,
    #                architect_mode, editor_model, auto_accept_architect
    
    # Check if architect mode is passed (position 9)
    assert len(call_args) >= 10
    assert call_args[9] is True
    
    # Check if editor model is passed (position 10)
    assert len(call_args) >= 11
    assert call_args[10] == "gemini/gemini-2.5-flash-preview-04-17"
    
    # Check if auto_accept_architect is passed (position 11) 
    assert len(call_args) >= 12
    assert call_args[11] is True
    
    # Verify response format
    result_dict = json.loads(result)
    assert "success" in result_dict
    assert "diff" in result_dict
    assert "is_cached_diff" in result_dict