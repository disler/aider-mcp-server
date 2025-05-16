#!/usr/bin/env python3
"""Apply formatting fixes to ensure pre-commit passes."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and return success status."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout}")
    return result.returncode == 0

def main():
    """Apply all formatting fixes."""
    # Change to the project directory
    project_dir = Path(__file__).parent
    
    # Run isort fixes
    run_command(f"cd {project_dir} && hatch -e dev run isort src/ tests/")
    
    # Run ruff format
    run_command(f"cd {project_dir} && hatch -e dev run ruff format src/ tests/")
    
    # Run ruff fixes
    run_command(f"cd {project_dir} && hatch -e dev run ruff check --fix src/ tests/")
    
    # Run pre-commit to verify
    result = run_command(f"cd {project_dir} && hatch -e dev run pre-commit run --all-files")
    
    if result:
        print("\nAll formatting issues fixed!")
    else:
        print("\nSome issues remain. Running pre-commit again...")
        run_command(f"cd {project_dir} && hatch -e dev run pre-commit run --all-files")

if __name__ == "__main__":
    main()