import os
import shutil
import subprocess
import sys
import tempfile
from typing import Generator

import pytest

# Add the src directory to the path for importing modules during tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


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
