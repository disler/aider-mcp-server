#!/usr/bin/env python3
"""Test the just-prompt integration in setup_aider_mcp.py"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from aider_mcp_server.setup_aider_mcp import (
    find_just_prompt_server,
    read_just_prompt_env_file,
    get_just_prompt_providers,
    select_just_prompt_models,
    create_just_prompt_config,
    create_mcp_config,
    main,
)


class TestJustPromptIntegration(unittest.TestCase):
    def test_find_just_prompt_server_with_specific_path(self):
        """Test finding just-prompt with a specific path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake just-prompt structure
            just_prompt_dir = Path(temp_dir) / "just-prompt"
            (just_prompt_dir / "src" / "just_prompt").mkdir(parents=True)
            
            result = find_just_prompt_server(str(just_prompt_dir))
            self.assertEqual(result, str(just_prompt_dir))
    
    def test_find_just_prompt_server_default_location(self):
        """Test finding just-prompt at default location."""
        default_path = Path("/mnt/l/ToolNexusMCP_plugins/just-prompt")
        if default_path.exists() and (default_path / "src" / "just_prompt").exists():
            result = find_just_prompt_server()
            self.assertEqual(result, str(default_path))
    
    def test_read_just_prompt_env_file(self):
        """Test reading just-prompt env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("OPENAI_API_KEY=test123\nGEMINI_API_KEY=test456\n")
            
            result = read_just_prompt_env_file(temp_dir)
            self.assertEqual(result["OPENAI_API_KEY"], "test123")
            self.assertEqual(result["GEMINI_API_KEY"], "test456")
    
    def test_get_just_prompt_providers(self):
        """Test getting available providers based on env vars."""
        env_vars = {
            "OPENAI_API_KEY": "test123",
            "GEMINI_API_KEY": "test456",
            "UNRELATED_KEY": "value"
        }
        
        providers = get_just_prompt_providers(env_vars)
        self.assertIn("openai", providers)
        self.assertIn("gemini", providers)
        self.assertNotIn("anthropic", providers)
    
    def test_create_just_prompt_config(self):
        """Test creating just-prompt configuration."""
        config = create_just_prompt_config("/path/to/just-prompt", "o:gpt-4o,g:gemini-2.0-flash")
        
        self.assertIn("just-prompt", config)
        self.assertEqual(config["just-prompt"]["type"], "stdio")
        self.assertEqual(config["just-prompt"]["command"], "uv")
        self.assertIn("--default-models", config["just-prompt"]["args"])
        self.assertIn("o:gpt-4o,g:gemini-2.0-flash", config["just-prompt"]["args"])
    
    def test_create_mcp_config_with_just_prompt(self):
        """Test creating merged MCP configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            aider_dir = "/path/to/aider"
            model = "gemini/gemini-2.5-pro"
            just_prompt_config = {
                "just-prompt": {
                    "type": "stdio",
                    "command": "uv",
                    "args": ["--directory", "/path/to/just-prompt", "run", "just-prompt"],
                    "env": {}
                }
            }
            
            success = create_mcp_config(temp_dir, aider_dir, model, just_prompt_config)
            self.assertTrue(success)
            
            # Read the created config
            config_path = Path(temp_dir) / ".mcp.json"
            with open(config_path) as f:
                config = json.load(f)
            
            self.assertIn("aider-mcp-server", config["mcpServers"])
            self.assertIn("just-prompt", config["mcpServers"])
            self.assertEqual(config["mcpServers"]["aider-mcp-server"]["command"], "uv")
            self.assertEqual(config["mcpServers"]["just-prompt"]["command"], "uv")
    
    def test_no_just_prompt_with_flag(self):
        """Test behavior when --also-just-prompt is used but just-prompt is not installed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake aider-mcp-server structure
            aider_dir = Path(temp_dir) / "aider-mcp-server"
            (aider_dir / "src" / "aider_mcp_server").mkdir(parents=True)
            (aider_dir / ".env").write_text("GEMINI_API_KEY=test_key\n")
            
            # Mock necessary functions
            with patch('aider_mcp_server.setup_aider_mcp.find_just_prompt_server', return_value=None), \
                 patch('aider_mcp_server.setup_aider_mcp.find_aider_mcp_server', return_value=str(aider_dir)), \
                 patch('aider_mcp_server.setup_aider_mcp.is_git_repo', return_value=True), \
                 patch('aider_mcp_server.setup_aider_mcp.select_model', return_value="gemini/gemini-2.5-pro"), \
                 patch('sys.argv', ['setup_aider_mcp', '--current-dir', temp_dir, '--also-just-prompt']):
                
                # Run the main function
                result = main()
                
                # Should succeed with only aider setup
                self.assertEqual(result, 0)
                
                # Check that MCP config was created
                config_path = Path(temp_dir) / ".mcp.json"
                self.assertTrue(config_path.exists())
                
                # Read the config
                with open(config_path) as f:
                    config = json.load(f)
                
                # Should only have aider-mcp-server
                self.assertIn("aider-mcp-server", config["mcpServers"])
                self.assertNotIn("just-prompt", config["mcpServers"])
    
    def test_with_just_prompt_successful(self):
        """Test successful setup with both aider and just-prompt."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake aider-mcp-server structure
            aider_dir = Path(temp_dir) / "aider-mcp-server"
            (aider_dir / "src" / "aider_mcp_server").mkdir(parents=True)
            (aider_dir / ".env").write_text("GEMINI_API_KEY=test_key\n")
            
            # Create fake just-prompt structure
            just_prompt_dir = Path(temp_dir) / "just-prompt"
            (just_prompt_dir / "src" / "just_prompt").mkdir(parents=True)
            (just_prompt_dir / ".env").write_text("OPENAI_API_KEY=test_openai\nGEMINI_API_KEY=test_gemini\n")
            
            # Mock necessary functions
            with patch('aider_mcp_server.setup_aider_mcp.find_just_prompt_server', return_value=str(just_prompt_dir)), \
                 patch('aider_mcp_server.setup_aider_mcp.find_aider_mcp_server', return_value=str(aider_dir)), \
                 patch('aider_mcp_server.setup_aider_mcp.is_git_repo', return_value=True), \
                 patch('aider_mcp_server.setup_aider_mcp.select_model', return_value="gemini/gemini-2.5-pro"), \
                 patch('aider_mcp_server.setup_aider_mcp.select_just_prompt_models', return_value="o:gpt-4o,g:gemini-2.0-flash"), \
                 patch('sys.argv', ['setup_aider_mcp', '--current-dir', temp_dir, '--also-just-prompt']):
                
                # Run the main function
                result = main()
                
                # Should succeed
                self.assertEqual(result, 0)
                
                # Check that MCP config was created
                config_path = Path(temp_dir) / ".mcp.json"
                self.assertTrue(config_path.exists())
                
                # Read the config
                with open(config_path) as f:
                    config = json.load(f)
                
                # Should have both aider-mcp-server and just-prompt
                self.assertIn("aider-mcp-server", config["mcpServers"])
                self.assertIn("just-prompt", config["mcpServers"])
                
                # Check just-prompt configuration
                just_prompt_config = config["mcpServers"]["just-prompt"]
                self.assertEqual(just_prompt_config["type"], "stdio")
                self.assertEqual(just_prompt_config["command"], "uv")
                self.assertIn("--default-models", just_prompt_config["args"])
                self.assertIn("o:gpt-4o,g:gemini-2.0-flash", just_prompt_config["args"])


if __name__ == "__main__":
    unittest.main()