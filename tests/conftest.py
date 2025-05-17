import os
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Generator

import pytest

# Add the src directory to the path for importing modules during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


@pytest.fixture(scope="session", autouse=False)
def mock_api_keys():
    """
    Set up mock API keys in the environment for tests.

    This is an optional fixture that can be used to avoid real API calls
    during testing. To use it, add it to your test function parameters.

    Example:
    def test_something(mock_api_keys):
        # Test with mock API keys
    """
    # Store original environment variables
    original_env = {}
    for key in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ]:
        original_env[key] = os.environ.get(key)

    # Set mock API keys
    os.environ["OPENAI_API_KEY"] = "sk-mock-openai-key"
    os.environ["ANTHROPIC_API_KEY"] = "sk-mock-anthropic-key"
    os.environ["GOOGLE_API_KEY"] = "mock-google-key"
    os.environ["GEMINI_API_KEY"] = "mock-gemini-key"

    yield

    # Restore original environment variables
    for key, value in original_env.items():
        if value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = value


@pytest.fixture
def temp_git_repo() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Get the full path to git executable
        git_executable = shutil.which("git")
        if not git_executable:
            pytest.skip("Git executable not found")

        # Use the full path in subprocess calls
        subprocess.run(  # noqa: S603
            [git_executable, "init"],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        # Create a README.md file
        with open(os.path.join(tmp_dir, "README.md"), "w") as f:
            f.write("# Test Repository\n\nThis is a test repository.\n")

        subprocess.run(  # noqa: S603
            [git_executable, "add", "README.md"],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(  # noqa: S603
            [git_executable, "commit", "-m", "Initial commit"],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        yield tmp_dir


@pytest.fixture
def free_port() -> int:
    """Get a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture
def server_process():
    """Fixture to manage server subprocess lifecycle."""
    process = None

    def _start_process(*args, **kwargs):
        nonlocal process
        process = subprocess.Popen(*args, **kwargs)  # noqa: S603
        return process

    yield _start_process

    # Cleanup
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
