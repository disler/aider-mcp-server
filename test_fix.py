#!/usr/bin/env python3
import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import the module
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir.absolute()))

from aider_mcp_server.atoms.tools.aider_ai_code import code_with_aider

async def main():
    print("Testing fixed code_with_aider function...")
    
    # Set up test parameters
    working_dir = os.getcwd()
    test_file = "test_file.py"
    
    # Create a test file if it doesn't exist
    if not os.path.exists(test_file):
        with open(test_file, "w") as f:
            f.write("# Test file for aider\n")
    
    # Run the function with simple parameters
    result = await code_with_aider(
        ai_coding_prompt="Add a function called hello_world that prints 'Hello, World!'",
        relative_editable_files=[test_file],
        working_dir=working_dir
    )
    
    print(f"Result: {result}")
    print("Test completed.")

if __name__ == "__main__":
    asyncio.run(main())