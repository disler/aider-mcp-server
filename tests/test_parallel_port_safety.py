"""Test that multiple SSE tests can run in parallel without port conflicts."""
import asyncio
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parallel_server_starts(free_port, server_process):
    """Test that a server can start with a dynamically allocated port."""
    # Use a test directory
    test_dir = Path(tempfile.gettempdir()) / f"test_aider_sse_{free_port}"
    test_dir.mkdir(exist_ok=True)
    
    # Initialize a git repo in the test directory
    subprocess.run(["git", "init"], cwd=test_dir, capture_output=True)
    
    # Start the SSE server
    process = server_process(
        [
            "python", "-m", "aider_mcp_server",
            "--server-mode", "sse",
            "--current-working-dir", str(test_dir),
            "--port", str(free_port),
            "--editor-model", "gpt-3.5-turbo"
        ],
        cwd=Path(__file__).parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"OPENAI_API_KEY": "test-key", **subprocess.os.environ}
    )
    
    # Wait briefly
    await asyncio.sleep(1)
    
    # Check if started
    assert process.poll() is None, "Server should still be running"
    
    # Cleanup is handled by the fixture


@pytest.mark.integration
@pytest.mark.asyncio
async def test_another_parallel_server(free_port, server_process):
    """Another test that runs in parallel to verify port isolation."""
    # Use a test directory
    test_dir = Path(tempfile.gettempdir()) / f"test_aider_sse_{free_port}"
    test_dir.mkdir(exist_ok=True)
    
    # Initialize a git repo in the test directory
    subprocess.run(["git", "init"], cwd=test_dir, capture_output=True)
    
    # Start the SSE server
    process = server_process(
        [
            "python", "-m", "aider_mcp_server",
            "--server-mode", "sse",
            "--current-working-dir", str(test_dir),
            "--port", str(free_port),
            "--editor-model", "gpt-3.5-turbo"
        ],
        cwd=Path(__file__).parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"OPENAI_API_KEY": "test-key", **subprocess.os.environ}
    )
    
    # Wait briefly
    await asyncio.sleep(1)
    
    # Check if started
    assert process.poll() is None, "Server should still be running"
    
    # Cleanup is handled by the fixture