"""Integration test to verify working directory is passed correctly in SSE mode."""

import asyncio
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sse_working_directory_integration(free_port, server_process):
    """Test that SSE server starts correctly with a valid git repository."""
    # Use a test directory
    test_dir = Path(tempfile.gettempdir()) / "test_aider_sse"
    test_dir.mkdir(exist_ok=True)

    # Initialize a git repo in the test directory
    subprocess.run(["git", "init"], cwd=test_dir, capture_output=True)  # noqa: S603, S607

    # Start the SSE server with the test directory
    process = server_process(
        [
            "python",
            "-m",
            "aider_mcp_server",
            "--server-mode",
            "sse",
            "--current-working-dir",
            str(test_dir),
            "--port",
            str(free_port),
            "--editor-model",
            "gpt-3.5-turbo",
        ],
        cwd=Path(__file__).parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"OPENAI_API_KEY": "test-key", **subprocess.os.environ},
    )

    try:
        # Give server time to start
        await asyncio.sleep(2)

        # Check if server is still running (it should be)
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Server failed to start or exited early. STDOUT: {stdout}\nSTDERR: {stderr}")

        # Terminate server gracefully
        process.terminate()

        # Get output
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            # Force kill if needed
            process.kill()
            stdout, stderr = process.communicate()

        # The server should have started without error
        combined_output = stdout + stderr
        assert "not a valid git repository" not in combined_output, (
            f"Unexpected git repository error found.\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        )

    finally:
        # Cleanup
        subprocess.run(["rm", "-rf", str(test_dir)], capture_output=True)  # noqa: S603, S607


if __name__ == "__main__":
    asyncio.run(test_sse_working_directory_integration(8766, lambda *args, **kwargs: subprocess.Popen(*args, **kwargs)))  # noqa: S603
