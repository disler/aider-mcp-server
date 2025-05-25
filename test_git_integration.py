#!/usr/bin/env python3
"""Test Git Integration"""

import sys
sys.path.insert(0, 'src')
import os
import tempfile

from src.aider_mcp_server.molecules.tools.aider_ai_code import (
    _normalize_file_paths,
    _get_git_diff,
    _check_for_meaningful_changes,
    get_changes_diff_or_content
)

def test_file_path_normalization():
    print("=== TESTING FILE PATH NORMALIZATION ===")
    
    # Test with working directory
    working_dir = "/tmp/test_project"
    
    test_cases = [
        # Relative paths
        (["src/main.py", "tests/test.py"], working_dir, ["src/main.py", "tests/test.py"]),
        
        # Absolute paths
        (["/tmp/test_project/src/main.py"], working_dir, ["src/main.py"]),
        
        # Mixed paths
        (["src/main.py", "/tmp/test_project/tests/test.py"], working_dir, ["src/main.py", "tests/test.py"]),
    ]
    
    for input_paths, wd, expected in test_cases:
        result = _normalize_file_paths(input_paths, wd)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {input_paths} -> {result}")

def test_git_diff():
    print("\n=== TESTING GIT DIFF ===")
    
    # Test in current directory (which is a git repo)
    working_dir = os.getcwd()
    
    # Test with a file that exists
    test_files = ["check_compatibility.py"]  # File we created earlier
    
    try:
        result = _get_git_diff(test_files, working_dir)
        print(f"  ✅ Git diff executed successfully")
        print(f"    Diff length: {len(result)} characters")
        if result.strip():
            print(f"    Has diff content: Yes")
        else:
            print(f"    Has diff content: No (file unchanged)")
    except Exception as e:
        print(f"  ❌ Git diff failed: {e}")

def test_meaningful_changes():
    print("\n=== TESTING MEANINGFUL CHANGE DETECTION ===")
    
    # Create temporary test files
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Test 1: Empty file
        empty_file = os.path.join(temp_dir, "empty.py")
        with open(empty_file, 'w') as f:
            f.write("")
        
        result = _check_for_meaningful_changes(["empty.py"], temp_dir)
        print(f"  Empty file -> meaningful: {result} (expected: False)")
        
        # Test 2: Comment-only file
        comment_file = os.path.join(temp_dir, "comment.py")
        with open(comment_file, 'w') as f:
            f.write("# Just a comment")
        
        result = _check_for_meaningful_changes(["comment.py"], temp_dir)
        print(f"  Comment-only file -> meaningful: {result} (expected: False)")
        
        # Test 3: File with actual code
        code_file = os.path.join(temp_dir, "code.py")
        with open(code_file, 'w') as f:
            f.write("def hello():\n    print('Hello')")
        
        result = _check_for_meaningful_changes(["code.py"], temp_dir)
        print(f"  Code file -> meaningful: {result} (expected: True)")
        
        # Test 4: Non-existent file
        result = _check_for_meaningful_changes(["nonexistent.py"], temp_dir)
        print(f"  Non-existent file -> meaningful: {result} (expected: False)")

def test_changes_diff_or_content():
    print("\n=== TESTING CHANGES DIFF OR CONTENT ===")
    
    # Test with current git repo
    working_dir = os.getcwd()
    test_files = ["check_compatibility.py", "test_architect_mode.py"]
    
    try:
        result = get_changes_diff_or_content(test_files, working_dir)
        print(f"  ✅ Changes diff/content retrieved successfully")
        print(f"    Result length: {len(result)} characters")
        if "git diff" in result.lower() or "file contents" in result.lower():
            print(f"    Contains expected content: Yes")
        else:
            print(f"    Content preview: {result[:100]}...")
    except Exception as e:
        print(f"  ❌ Changes diff/content failed: {e}")

def test_git_repo_detection():
    print("\n=== TESTING GIT REPO DETECTION ===")
    
    # Test current directory (should be a git repo)
    current_dir = os.getcwd()
    git_dir = os.path.join(current_dir, ".git")
    is_git_repo = os.path.isdir(git_dir)
    
    print(f"  Current directory: {current_dir}")
    print(f"  .git directory exists: {is_git_repo}")
    
    if is_git_repo:
        print(f"  ✅ Git repository detected correctly")
    else:
        print(f"  ❌ Git repository not detected")

if __name__ == "__main__":
    test_file_path_normalization()
    test_git_diff()
    test_meaningful_changes()
    test_changes_diff_or_content()
    test_git_repo_detection()