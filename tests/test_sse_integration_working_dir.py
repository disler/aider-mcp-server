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
@pytest.mark.skip(reason="Integration test has timing issues with SSE connections")
async def test_sse_working_directory_integration():
    """Test that SSE server correctly passes working directory to aider handlers."""
    # Use a test directory
    test_dir = Path(tempfile.gettempdir()) / "test_aider_sse"
    test_dir.mkdir(exist_ok=True)
    
    # Initialize a git repo in the test directory
    result = subprocess.run(["git", "init"], cwd=test_dir, capture_output=True)
    if result.returncode != 0:
        # Might already exist, that's ok
        pass
    
    # Use a unique port for tests to avoid conflicts
    import random
    test_port = str(random.randint(9000, 9999))
    
    # Start the SSE server with the test directory
    server_process = subprocess.Popen(
        [
            "python", "-m", "aider_mcp_server",
            "--server-mode", "sse",
            "--current-working-dir", str(test_dir),
            "--port", test_port,
            "--editor-model", "gpt-3.5-turbo"
        ],
        cwd=Path(__file__).parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "OPENAI_API_KEY": "test-key",
            "TEST_MODE": "true",  # Add test mode flag
            **subprocess.os.environ
        }
    )
    
    try:
        # Wait longer for server to start (since it creates various resources)
        await asyncio.sleep(4)
        
        # Check if server is running
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            # Provide more detailed error information
            if stderr:
                pytest.fail(f"Server failed to start. STDERR: {stderr}")
            else:
                pytest.fail(f"Server failed to start. STDOUT: {stdout}")
        
        # Try to connect to the SSE endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:{test_port}/sse/",
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
        
        # Verify the working directory was validated - check for proper log message
        assert "Working directory" in stdout or "Working directory" in stderr, \
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