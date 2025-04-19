import sys
import pytest
from unittest.mock import AsyncMock, patch
import argparse
from aider_mcp_server.__main__ import main
from aider_mcp_server.atoms.atom_utils import DEFAULT_EDITOR_MODEL

@pytest.fixture
def mock_serve():
    """Mock the serve function for testing."""
    with patch("aider_mcp_server.__main__.serve", new_callable=AsyncMock) as mock:
        yield mock

def test_argument_parsing(mock_serve, monkeypatch, tmp_path):
    """Test that arguments are parsed correctly and serve is called with them."""
    # Set up test arguments
    test_dir = str(tmp_path)
    test_args = ["prog", "--current-working-dir", test_dir]
    monkeypatch.setattr(sys, "argv", test_args)
    
    # Run the main function
    main()
    
    # Check that serve was called with the correct arguments
    mock_serve.assert_called_once()
    call_args = mock_serve.call_args[1]
    assert call_args["editor_model"] == DEFAULT_EDITOR_MODEL
    assert call_args["current_working_dir"] == test_dir

def test_custom_editor_model(mock_serve, monkeypatch, tmp_path):
    """Test that custom editor model is passed correctly."""
    # Set up test arguments with custom editor model
    test_dir = str(tmp_path)
    custom_model = "custom-model"
    test_args = ["prog", "--current-working-dir", test_dir, "--editor-model", custom_model]
    monkeypatch.setattr(sys, "argv", test_args)
    
    # Run the main function
    main()
    
    # Check that serve was called with the correct arguments
    mock_serve.assert_called_once()
    call_args = mock_serve.call_args[1]
    assert call_args["editor_model"] == custom_model
    assert call_args["current_working_dir"] == test_dir

@patch("aider_mcp_server.__main__.argparse.ArgumentParser.parse_args")
def test_missing_required_argument(mock_parse_args, mock_serve):
    """Test that the program exits when missing required arguments."""
    # Simulate the behavior when parse_args raises an error
    mock_parse_args.side_effect = SystemExit(2)
    
    # Check that SystemExit is raised
    with pytest.raises(SystemExit) as excinfo:
        main()
    
    # Verify that serve was not called
    mock_serve.assert_not_called()
    assert excinfo.value.code == 2
