#!/usr/bin/env python3
"""Test script to verify JSON output from aider MCP server"""

import asyncio
import json
from unittest.mock import Mock, patch
from aider_mcp_server.atoms.tools.aider_ai_code import code_with_aider, init_diff_cache, shutdown_diff_cache


class MockCoder:
    """Mock Aider Coder that simulates the problematic output"""
    
    def __init__(self, *args, **kwargs):
        pass
        
    def run(self, prompt):
        # Simulate the problematic behavior where Aider prints to stdout
        print("Creating empty file test.py")
        return "Mock execution completed"
        
    @staticmethod
    def create(*args, **kwargs):
        return MockCoder()


class MockInputOutput:
    """Mock InputOutput class"""
    
    def __init__(self, *args, **kwargs):
        self.yes_to_all = True
        self.tool_error = False
        self.dry_run = False
        self.quiet = True
        self.output = None
        self.tool_output = None
        self.tool_error_output = None


class MockModel:
    """Mock Model class"""
    
    def __init__(self, *args, **kwargs):
        pass
        
    def commit_message_models(self):
        return []


async def test_json_output():
    """Test that only JSON is output from the code_with_aider function"""
    
    # Initialize diff cache
    await init_diff_cache()
    
    # Mock the Aider components
    with patch('aider_mcp_server.atoms.tools.aider_ai_code.Coder', MockCoder), \
         patch('aider_mcp_server.atoms.tools.aider_ai_code.InputOutput', MockInputOutput), \
         patch('aider_mcp_server.atoms.tools.aider_ai_code.Model', MockModel):
        
        # Test the code_with_aider function
        result = await code_with_aider(
            ai_coding_prompt="Add a test function",
            relative_editable_files=["test.py"],
            relative_readonly_files=[],
            model="test-model",
            working_dir="/tmp/test",
        )
        
        print("=== RAW RESULT ===")
        print(repr(result))
        print("=== END RAW RESULT ===")
        
        # Try to parse the result as JSON
        try:
            parsed = json.loads(result)
            print("SUCCESS: Result is valid JSON")
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            print(f"ERROR: Result is not valid JSON: {e}")
            print(f"Result: {result}")
            
            # Check if there's non-JSON data at the beginning
            if not result.startswith('{'):
                print(f"Non-JSON data detected at start: {result[:50]}...")
    
    # Shutdown diff cache
    await shutdown_diff_cache()


if __name__ == "__main__":
    asyncio.run(test_json_output())