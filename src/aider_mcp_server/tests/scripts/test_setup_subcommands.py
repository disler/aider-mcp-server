#!/usr/bin/env python3
"""Test the new subcommands in setup_aider_mcp.py"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from aider_mcp_server.setup_aider_mcp import (
    update_model_config,
    list_available_models,
    setup_command,
    change_model_command,
    list_models_command,
    main,
)


class TestSetupSubcommands(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            "mcpServers": {
                "aider": {
                    "type": "stdio",
                    "command": "uv",
                    "args": [
                        "--directory",
                        "/test/path",
                        "run",
                        "aider-mcp-server",
                        "--editor-model",
                        "gemini/gemini-2.5-pro-exp-03-25",
                        "--current-working-dir",
                        "/test/dir"
                    ]
                }
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_update_model_config(self):
        """Test updating model in existing configuration."""
        config_path = Path(self.temp_dir) / ".mcp.json"
        with open(config_path, 'w') as f:
            json.dump(self.test_config, f)
        
        # Update the model
        result = update_model_config(self.temp_dir, "openai/gpt-4o")
        self.assertTrue(result)
        
        # Verify the config was updated
        with open(config_path, 'r') as f:
            updated_config = json.load(f)
        
        args = updated_config["mcpServers"]["aider"]["args"]
        model_index = args.index("--editor-model") + 1
        self.assertEqual(args[model_index], "openai/gpt-4o")
    
    def test_update_model_config_no_file(self):
        """Test updating model when no config exists."""
        result = update_model_config(self.temp_dir, "openai/gpt-4o")
        self.assertFalse(result)
    
    @patch('builtins.print')
    def test_list_available_models_with_keys(self, mock_print):
        """Test listing models with API keys available."""
        env_vars = {
            "GEMINI_API_KEY": "test123",
            "OPENAI_API_KEY": "test456",
        }
        
        list_available_models(env_vars)
        
        # Verify it prints available models
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertIn("Available models based on your API keys:", print_calls)
    
    @patch('builtins.print')
    def test_list_available_models_no_keys(self, mock_print):
        """Test listing models with no API keys."""
        list_available_models({})
        
        # Verify it prints instructions for setting keys
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertIn("No API keys found in environment.", print_calls)
    
    @patch('sys.argv', ['setup_aider_mcp.py', 'list-models'])
    @patch('aider_mcp_server.setup_aider_mcp.find_aider_mcp_server')
    @patch('aider_mcp_server.setup_aider_mcp.read_env_file')
    def test_list_models_command_integration(self, mock_read_env, mock_find):
        """Test list-models command through main function."""
        mock_find.return_value = "/test/path"
        mock_read_env.return_value = {"GEMINI_API_KEY": "test"}
        
        result = main()
        self.assertEqual(result, 0)
        mock_find.assert_called_once()
        mock_read_env.assert_called_once()
    
    @patch('sys.argv', ['setup_aider_mcp.py', 'change-model', '--model', 'openai/gpt-4o'])
    @patch('aider_mcp_server.setup_aider_mcp.find_aider_mcp_server')
    @patch('aider_mcp_server.setup_aider_mcp.read_env_file')
    @patch('aider_mcp_server.setup_aider_mcp.update_model_config')
    def test_change_model_command_integration(self, mock_update, mock_read_env, mock_find):
        """Test change-model command through main function."""
        mock_find.return_value = "/test/path"
        mock_read_env.return_value = {"OPENAI_API_KEY": "test"}
        mock_update.return_value = True
        
        result = main()
        self.assertEqual(result, 0)
        mock_update.assert_called_once()
    
    @patch('sys.argv', ['setup_aider_mcp.py'])  # No command specified
    @patch('aider_mcp_server.setup_aider_mcp.setup_command')
    def test_default_to_setup_command(self, mock_setup):
        """Test that main defaults to setup command when no command specified."""
        mock_setup.return_value = 0
        
        result = main()
        self.assertEqual(result, 0)
        mock_setup.assert_called_once()


if __name__ == "__main__":
    unittest.main()