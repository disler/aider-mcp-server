"""Simple test to verify working directory configuration in SSE mode."""

import subprocess
import tempfile
import time
from pathlib import Path


def test_sse_working_directory_logs(free_port):
    """Test that SSE server logs validate the working directory."""
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

    # Terminate and get output
    process.terminate()

    try:
        stdout, stderr = process.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()

    # Check the output
    print("STDOUT:", stdout)
    print("STDERR:", stderr)

    # Verify the working directory was validated
    output = stdout + stderr
    assert "Validated working directory (git repository):" in output, (
        f"Working directory validation not found in logs.\nOutput: {output}"
    )

    # Verify the correct directory was used
    assert str(test_dir) in output, f"Test directory {test_dir} not found in logs.\nOutput: {output}"

    # Cleanup
    subprocess.run(["rm", "-rf", str(test_dir)], capture_output=True)  # noqa: S603, S607

    print("Test passed! Working directory was properly validated.")


if __name__ == "__main__":
    test_sse_working_directory_logs()
