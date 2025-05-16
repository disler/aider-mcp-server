"""Integration test to verify working directory is passed correctly in SSE mode."""
import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sse_working_directory_integration():
    """Test that SSE server correctly passes working directory to aider handlers."""
    # Use a test directory
    test_dir = Path(tempfile.gettempdir()) / "test_aider_sse"
    test_dir.mkdir(exist_ok=True)
    
    # Initialize a git repo in the test directory
    subprocess.run(["git", "init"], cwd=test_dir, capture_output=True)
    
    # Start the SSE server with the test directory
    server_process = subprocess.Popen(
        [
            "python", "-m", "aider_mcp_server",
            "--server-mode", "sse",
            "--current-working-dir", str(test_dir),
            "--port", "8767",
            "--editor-model", "gpt-3.5-turbo"
        ],
        cwd=Path(__file__).parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"OPENAI_API_KEY": "test-key", **subprocess.os.environ}
    )
    
    try:
        # Wait for server to start
        await asyncio.sleep(2)
        
        # Check if server is running
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            pytest.fail(f"Server failed to start. STDOUT: {stdout}\nSTDERR: {stderr}")
        
        # Try to connect to the SSE endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8767/sse/",
                headers={"Accept": "text/event-stream"},
                timeout=5
            )
            
            assert response.status_code == 200, f"SSE connection failed: {response.status_code}"
            
        # Check server logs for the working directory validation message
        # Give it a moment to process
        await asyncio.sleep(1)
        
        # Terminate server and get output
        server_process.terminate()
        stdout, stderr = server_process.communicate(timeout=5)
        
        # Verify the working directory was validated
        assert "Validated working directory (git repository):" in stdout or \
               "Validated working directory (git repository):" in stderr, \
               f"Working directory validation not found in logs.\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        
        # Verify the correct directory was used
        assert str(test_dir) in stdout or str(test_dir) in stderr, \
               f"Test directory {test_dir} not found in logs.\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        
    finally:
        # Ensure server is terminated
        if server_process.poll() is None:
            server_process.terminate()
            server_process.wait(timeout=5)
        
        # Cleanup
        subprocess.run(["rm", "-rf", str(test_dir)], capture_output=True)


if __name__ == "__main__":
    asyncio.run(test_sse_working_directory_integration())