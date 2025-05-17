"""Test to verify working directory validation happens in SSE mode."""

import subprocess
import tempfile
from pathlib import Path

import pytest


def test_sse_validates_working_directory():
    """Test that SSE server validates working directory is a git repo."""
    # Use a non-git directory
    test_dir = Path(tempfile.gettempdir()) / "test_not_git"
    test_dir.mkdir(exist_ok=True)

    try:
        # Try to start the SSE server with a non-git directory
        # This should fail
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
                "8768",
                "--editor-model",
                "gpt-3.5-turbo",
            ],
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"OPENAI_API_KEY": "test-key", **subprocess.os.environ},
        )

        # Wait for error
        stdout, stderr = process.communicate(timeout=2)
        return_code = process.returncode

        # The process should have exited with an error
        assert return_code != 0, f"Expected non-zero exit code, got {return_code}"

        # Check for error message about not being a git repository
        combined_output = stdout + stderr
        assert "not a valid git repository" in combined_output or "not a git repository" in combined_output, (
            f"Expected git repository error message not found.\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        )

    finally:
        # Cleanup
        subprocess.run(["rm", "-rf", str(test_dir)], capture_output=True)  # noqa: S603, S607


def test_sse_accepts_git_directory():
    """Test that SSE server accepts a valid git directory."""
    # Use a git directory
    test_dir = Path(tempfile.gettempdir()) / "test_git"
    test_dir.mkdir(exist_ok=True)

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=test_dir, capture_output=True)  # noqa: S603, S607

    try:
        # Start the SSE server with a git directory
        # This should start successfully
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
                "8769",
                "--editor-model",
                "gpt-3.5-turbo",
            ],
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"OPENAI_API_KEY": "test-key", **subprocess.os.environ},
        )

        # Give it some time to start
        import time

        time.sleep(2)

        # Check if server is still running
        return_code = process.poll()

        # If it exited, get the output
        if return_code is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Server exited unexpectedly: {return_code}\nSTDOUT: {stdout}\nSTDERR: {stderr}")

        # Terminate the server
        process.terminate()
        stdout, stderr = process.communicate()

        # The process should not have any error about git repository
        combined_output = stdout + stderr
        assert "not a valid git repository" not in combined_output, (
            f"Unexpected git repository error.\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        )

    finally:
        # Cleanup
        subprocess.run(["rm", "-rf", str(test_dir)], capture_output=True)  # noqa: S603, S607


if __name__ == "__main__":
    test_sse_validates_working_directory()
    test_sse_accepts_git_directory()
    print("All tests passed!")
