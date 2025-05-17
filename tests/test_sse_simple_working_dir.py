"""Simple test to verify working directory configuration in SSE mode."""

import subprocess
import tempfile
import time
from pathlib import Path


def test_sse_working_directory_logs(free_port):
    """Test that SSE server starts correctly with a valid git repository."""
    # Use a test directory
    test_dir = Path(tempfile.gettempdir()) / "test_aider_sse"
    test_dir.mkdir(exist_ok=True)

    # Initialize a git repo in the test directory
    subprocess.run(["git", "init"], cwd=test_dir, capture_output=True)  # noqa: S603, S607

    # Start the SSE server with the test directory
    process = subprocess.Popen(  # noqa: S603
        [  # noqa: S607
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

    # Wait a bit for the server to start
    time.sleep(2)

    # Check if server is still running (it should be)
    return_code = process.poll()
    
    # Terminate the process
    process.terminate()
    
    # Get output
    try:
        stdout, stderr = process.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()

    # Check the output
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    
    # If the server exited early, that's an error
    if return_code is not None:
        assert False, f"Server exited early with code {return_code}.\nSTDOUT: {stdout}\nSTDERR: {stderr}"

    # The server should have started without error
    combined_output = stdout + stderr
    assert "not a valid git repository" not in combined_output, (
        f"Unexpected git repository error found.\nSTDOUT: {stdout}\nSTDERR: {stderr}"
    )

    # Cleanup
    subprocess.run(["rm", "-rf", str(test_dir)], capture_output=True)  # noqa: S603, S607

    print("Test passed! SSE server started correctly with git directory.")


if __name__ == "__main__":
    test_sse_working_directory_logs(8767)