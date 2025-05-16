"""Simple test to verify working directory configuration in SSE mode."""
import subprocess
import tempfile
import time
from pathlib import Path


def test_sse_working_directory_logs():
    """Test that SSE server logs validate the working directory."""
    # Use a test directory
    test_dir = Path(tempfile.gettempdir()) / "test_aider_sse"
    test_dir.mkdir(exist_ok=True)
    
    # Initialize a git repo in the test directory
    subprocess.run(["git", "init"], cwd=test_dir, capture_output=True)
    
    # Start the SSE server with the test directory
    result = subprocess.run(
        [
            "python", "-m", "aider_mcp_server",
            "--server-mode", "sse",
            "--current-working-dir", str(test_dir),
            "--port", "8767",
            "--editor-model", "gpt-3.5-turbo"
        ],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=3,  # Quick timeout just to see startup logs
        env={"OPENAI_API_KEY": "test-key", **subprocess.os.environ}
    )
    
    # Check the output
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    # Verify the working directory was validated
    output = result.stdout + result.stderr
    assert "Validated working directory (git repository):" in output, \
           f"Working directory validation not found in logs.\nOutput: {output}"
    
    # Verify the correct directory was used
    assert str(test_dir) in output, \
           f"Test directory {test_dir} not found in logs.\nOutput: {output}"
    
    # Cleanup
    subprocess.run(["rm", "-rf", str(test_dir)], capture_output=True)
    
    print("Test passed! Working directory was properly validated.")


if __name__ == "__main__":
    test_sse_working_directory_logs()