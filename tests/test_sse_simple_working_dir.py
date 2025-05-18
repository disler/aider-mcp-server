"""Simple test to verify working directory configuration in SSE mode."""

import subprocess
import tempfile
from pathlib import Path


def test_sse_working_directory_logs():
    """Test that SSE server logs validate the working directory."""
    # Skip the test if the module cannot be imported
    import pytest
    try:
        import aider_mcp_server  # noqa: F401
    except ImportError:
        pytest.skip("aider_mcp_server module not available - likely installation issue")
    
    # Use a test directory with a unique name to avoid collisions
    import os
    import random
    
    test_base = os.path.join(tempfile.gettempdir(), f"test_aider_sse_{random.randint(1000, 9999)}")  # noqa: S311
    test_dir = Path(test_base)
    test_dir.mkdir(exist_ok=True)

    try:
        # Initialize a git repo in the test directory
        subprocess.run(["git", "init"], cwd=test_dir, capture_output=True)  # noqa: S603, S607

        # Use a random port to avoid collisions
        test_port = str(random.randint(9000, 9999))  # noqa: S311
        
        # Start the SSE server with the test directory
        try:
            result = subprocess.run(  # noqa: S603
                [  # noqa: S607
                    "python",
                    "-m",
                    "aider_mcp_server",
                    "--server-mode",
                    "sse",
                    "--current-working-dir",
                    str(test_dir),
                    "--port",
                    test_port,
                    "--editor-model",
                    "gpt-3.5-turbo",
                ],
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                timeout=10,  # Even more timeout for CI
                env={
                    "OPENAI_API_KEY": "test-key",
                    "TEST_MODE": "true",
                    "MCP_LOG_LEVEL": "DEBUG",  # Enable debug logging to see what's happening
                    **subprocess.os.environ
                },
            )
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired as e:
            # In CI, the server might take longer to start - get partial output
            # Handle both str and bytes outputs
            stdout = e.stdout
            stderr = e.stderr
            if isinstance(stdout, bytes):
                stdout = stdout.decode('utf-8', errors='replace')
            if isinstance(stderr, bytes):
                stderr = stderr.decode('utf-8', errors='replace')
            stdout = stdout or ""
            stderr = stderr or ""
            print(f"Process timed out after {e.timeout} seconds - checking partial output")
            
        # Check the output
        print("STDOUT:", stdout)
        print("STDERR:", stderr)

        # Verify the working directory was validated
        output = stdout + stderr
        
        # Check for validation message - it might have different formats
        validation_found = (
            "Validated working directory (git repository):" in output or
            "working directory" in output.lower() and str(test_dir) in output
        )
        
        assert validation_found, (
            f"Working directory validation not found in logs.\nOutput: {output}"
        )

        # Verify the correct directory was used
        assert str(test_dir) in output, f"Test directory {test_dir} not found in logs.\nOutput: {output}"

        print("Test passed! Working directory was properly validated.")
    
    finally:
        # Cleanup - ensure we clean up even if test fails
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    test_sse_working_directory_logs()
