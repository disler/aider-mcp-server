#!/usr/bin/env python3
"""
Pre-commit hook script to check aider compatibility before committing.
"""

import pathlib
import subprocess
import sys

try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib  # fallback for older Python versions

import importlib.metadata


def check_aider_compatibility():
    """Check if our code is compatible with installed aider version."""
    # Get expected version from pyproject.toml
    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    if hasattr(tomllib, "load"):
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
    else:
        with open(pyproject_path) as f:
            config = tomllib.load(f)

    dependencies = config["project"]["dependencies"]
    aider_dep = None
    for dep in dependencies:
        if dep.startswith("aider-chat"):
            aider_dep = dep
            break

    if not aider_dep:
        print("Warning: aider-chat not found in pyproject.toml dependencies")
        return 0

    # Extract version requirement
    expected = aider_dep.split(">=")[1] if ">=" in aider_dep else None
    if not expected:
        print(f"Warning: Could not parse version from {aider_dep}")
        return 0

    # Get actual installed version
    try:
        actual = importlib.metadata.version("aider-chat")
    except importlib.metadata.PackageNotFoundError:
        print("Warning: aider-chat not installed")
        return 0

    print(f"Expected aider version: >={expected}")
    print(f"Actual aider version: {actual}")

    # Run compatibility test
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", "tests/compatibility/test_aider_compatibility.py", "-v"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Compatibility tests failed!")
        print(result.stdout)
        print(result.stderr)
        return 1

    print("Compatibility tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(check_aider_compatibility())
