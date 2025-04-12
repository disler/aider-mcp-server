import os
import pytest
import sys
import tempfile
import shutil
import subprocess

# Add the src directory to the path for importing modules during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture
def temp_git_repo():
    """Create a temporary directory with an initialized Git repository for testing."""
    tmp_dir = tempfile.mkdtemp()
    
    # Initialize git repository in the temp directory
    subprocess.run(["git", "init"], cwd=tmp_dir, capture_output=True, text=True, check=True)
    
    # Configure git user for the repository
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_dir, capture_output=True, text=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_dir, capture_output=True, text=True, check=True)
    
    # Create and commit an initial file to have a valid git history
    with open(os.path.join(tmp_dir, "README.md"), "w") as f:
        f.write("# Test Repository\nThis is a test repository for Aider MCP Server tests.")
    
    subprocess.run(["git", "add", "README.md"], cwd=tmp_dir, capture_output=True, text=True, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_dir, capture_output=True, text=True, check=True)
    
    yield tmp_dir
    
    # Clean up
    shutil.rmtree(tmp_dir)
