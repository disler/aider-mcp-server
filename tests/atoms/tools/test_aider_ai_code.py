import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import pathlib
from typing import Generator, Any

import pytest

from aider_mcp_server.atoms.tools.aider_ai_code import code_with_aider, load_env_files


def api_keys_missing() -> bool:
    """
    Check if required API keys are missing after loading .env files.
    Looks for any one of the common API keys.
    """
    # Determine the project root directory relative to this test file
    # This file is tests/atoms/tools/test_aider_ai_code.py
    # Project root is 4 levels up
    try:
        project_root = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
        # Load environment variables from .env files, starting search from project root
        load_env_files(working_dir=str(project_root))
    except Exception as e:
        # Log or handle error if path resolution or loading fails, but don't fail the check
        print(f"Warning: Could not load .env files for API key check: {e}")
        pass # Continue checking environment variables directly

    # List of potential API keys for different providers
    potential_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "VERTEX_AI_API_KEY",
    ]

    # Check if any of the potential keys are set in the environment
    # Return True if *none* are found, False if *at least one* is found
    return not any(os.environ.get(key) is not None for key in potential_keys)


@pytest.fixture
def git_repo_with_files() -> Generator[str, None, None]:
    """Create a temporary git repository with some files for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Get the full path to git executable
        git_executable = shutil.which("git")
        if not git_executable:
            pytest.skip("Git executable not found")

        # Initialize git repository in the temp directory
        subprocess.run(  # noqa: S603
            [git_executable, "init"],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        # Configure git user for the repository
        subprocess.run(  # noqa: S603
            [git_executable, "config", "user.name", "Test User"],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(  # noqa: S603
            [git_executable, "config", "user.email", "test@example.com"],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        # Create and commit an initial file to have a valid git history
        with open(os.path.join(tmp_dir, "README.md"), "w") as f:
            f.write(
                "# Test Repository\nThis is a test repository for Aider MCP Server tests."
            )

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
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory with an initialized Git repository for testing."""
    tmp_dir = tempfile.mkdtemp()

    # Get the full path to git executable
    git_executable = shutil.which("git")
    if not git_executable or git_executable is None:
        pytest.skip("Git executable not found")

    # Initialize git repository in the temp directory
    subprocess.run(  # noqa: S603
        [git_executable, "init"],
        cwd=tmp_dir,
        capture_output=True,
        text=True,
        check=True,
    )

    # Configure git user for the repository
    subprocess.run(  # noqa: S603
        [git_executable, "config", "user.name", "Test User"],
        cwd=tmp_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(  # noqa: S603
        [git_executable, "config", "user.email", "test@example.com"],
        cwd=tmp_dir,
        capture_output=True,
        text=True,
        check=True,
    )

    # Create and commit an initial file to have a valid git history
    with open(os.path.join(tmp_dir, "README.md"), "w") as f:
        f.write(
            "# Test Repository\nThis is a test repository for Aider MCP Server tests."
        )

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

    # Clean up
    shutil.rmtree(tmp_dir)


@pytest.mark.skipif(api_keys_missing(), reason="API keys required for this test")
def test_addition(temp_dir: str) -> None:
    """Test that code_with_aider can create a file that adds two numbers."""
    # Create the test file
    test_file = os.path.join(temp_dir, "math_add.py")
    with open(test_file, "w") as f:
        f.write("# This file should implement addition\n")

    prompt = "Implement a function add(a, b) that returns the sum of a and b in the math_add.py file."

    # Run code_with_aider with working_dir
    result = code_with_aider(
        ai_coding_prompt=prompt,
        relative_editable_files=[test_file],
        working_dir=temp_dir,  # Pass the temp directory as working_dir
    )

    # Parse the JSON result
    result_dict = json.loads(result)

    # Check that it succeeded
    assert result_dict["success"] is True, "Expected code_with_aider to succeed"
    assert "diff" in result_dict, "Expected diff to be in result"

    # Check that the file was modified correctly
    with open(test_file, "r") as f:
        content = f.read()

    assert any(x in content for x in ["def add(a, b):", "def add(a:"]), (
        "Expected to find add function in the file"
    )
    assert "return a + b" in content, "Expected to find return statement in the file"

    # Try to import and use the function
    import sys

    sys.path.append(temp_dir)
    from math_add import add  # type: ignore

    assert add(2, 3) == 5, "Expected add(2, 3) to return 5"


@pytest.mark.skipif(api_keys_missing(), reason="API keys required for this test")
def test_subtraction(temp_dir: str) -> None:
    """Test that code_with_aider can create a file that subtracts two numbers."""
    # Create the test file
    test_file = os.path.join(temp_dir, "math_subtract.py")
    with open(test_file, "w") as f:
        f.write("# This file should implement subtraction\n")

    prompt = "Implement a function subtract(a, b) that returns a minus b in the math_subtract.py file."

    # Run code_with_aider with working_dir
    result = code_with_aider(
        ai_coding_prompt=prompt,
        relative_editable_files=[test_file],
        working_dir=temp_dir,  # Pass the temp directory as working_dir
    )

    # Parse the JSON result
    result_dict = json.loads(result)

    # Check that it succeeded
    assert result_dict["success"] is True, "Expected code_with_aider to succeed"
    assert "diff" in result_dict, "Expected diff to be in result"

    # Check that the file was modified correctly
    with open(test_file, "r") as f:
        content = f.read()

    assert any(x in content for x in ["def subtract(a, b):", "def subtract(a:"]), (
        "Expected to find subtract function in the file"
    )
    assert "return a - b" in content, "Expected to find return statement in the file"

    # Try to import and use the function
    import sys

    sys.path.append(temp_dir)
    from math_subtract import subtract  # type: ignore

    assert subtract(5, 3) == 2, "Expected subtract(5, 3) to return 2"


@pytest.mark.skipif(api_keys_missing(), reason="API keys required for this test")
def test_multiplication(temp_dir: str) -> None:
    """Test that code_with_aider can create a file that multiplies two numbers."""
    # Create the test file
    test_file = os.path.join(temp_dir, "math_multiply.py")
    with open(test_file, "w") as f:
        f.write("# This file should implement multiplication\n")

    prompt = "Implement a function multiply(a, b) that returns the product of a and b in the math_multiply.py file."

    # Run code_with_aider with working_dir
    result = code_with_aider(
        ai_coding_prompt=prompt,
        relative_editable_files=[test_file],
        working_dir=temp_dir,  # Pass the temp directory as working_dir
    )

    # Parse the JSON result
    result_dict = json.loads(result)

    # Check that it succeeded
    assert result_dict["success"] is True, "Expected code_with_aider to succeed"
    assert "diff" in result_dict, "Expected diff to be in result"

    # Check that the file was modified correctly
    with open(test_file, "r") as f:
        content = f.read()

    assert any(x in content for x in ["def multiply(a, b):", "def multiply(a:"]), (
        "Expected to find multiply function in the file"
    )
    assert "return a * b" in content, "Expected to find return statement in the file"

    # Try to import and use the function
    import sys

    sys.path.append(temp_dir)
    from math_multiply import multiply  # type: ignore

    assert multiply(2, 3) == 6, "Expected multiply(2, 3) to return 6"


@pytest.mark.skipif(api_keys_missing(), reason="API keys required for this test")
def test_division(temp_dir: str) -> None:
    """Test that code_with_aider can create a file that divides two numbers."""
    # Create the test file
    test_file = os.path.join(temp_dir, "math_divide.py")
    with open(test_file, "w") as f:
        f.write("# This file should implement division\n")

    prompt = "Implement a function divide(a, b) that returns a divided by b in the math_divide.py file. Handle division by zero by returning None."

    # Run code_with_aider with working_dir
    result = code_with_aider(
        ai_coding_prompt=prompt,
        relative_editable_files=[test_file],
        working_dir=temp_dir,  # Pass the temp directory as working_dir
    )

    # Parse the JSON result
    result_dict = json.loads(result)

    # Check that it succeeded
    assert result_dict["success"] is True, "Expected code_with_aider to succeed"
    assert "diff" in result_dict, "Expected diff to be in result"

    # Check that the file was modified correctly
    with open(test_file, "r") as f:
        content = f.read()

    assert any(x in content for x in ["def divide(a, b):", "def divide(a:"]), (
        "Expected to find divide function in the file"
    )
    assert "return" in content, "Expected to find return statement in the file"

    # Try to import and use the function
    import sys

    sys.path.append(temp_dir)
    from math_divide import divide  # type: ignore

    assert divide(6, 3) == 2, "Expected divide(6, 3) to return 2"
    assert divide(1, 0) is None, "Expected divide(1, 0) to return None"


def test_failure_case(temp_dir: str) -> None:
    """Test that code_with_aider returns error information for a failure scenario."""

    # Save the original directory before changing it
    original_dir = os.getcwd()

    try:
        # Ensure this test runs in a non-git directory
        os.chdir(temp_dir)

        # Create a test file in the temp directory
        test_file = os.path.join(temp_dir, "failure_test.py")
        with open(test_file, "w") as f:
            f.write("# This file should trigger a failure\n")

        # Use an invalid model name to ensure a failure
        prompt = "This prompt should fail because we're using a non-existent model."

        # Run code_with_aider with an invalid model name
        result = code_with_aider(
            ai_coding_prompt=prompt,
            relative_editable_files=[test_file],
            model="non_existent_model_123456789",  # This model doesn't exist
            working_dir=temp_dir,  # Pass the temp directory as working_dir
        )

        # Parse the JSON result
        result_dict = json.loads(result)

        # Check the result - we're still expecting success=False but the important part
        # is that we get a diff that explains the error.
        # The diff should indicate that no meaningful changes were made,
        # often because the model couldn't be reached or produced no output.
        assert "diff" in result_dict, "Expected diff to be in result"
        diff_content = result_dict["diff"]
        assert (
            "File contents after editing (git not used):" in diff_content
            or "No meaningful changes detected" in diff_content
        ), (
            f"Expected error information like 'File contents after editing' or 'No meaningful changes' in diff, but got: {diff_content}"
        )
    finally:
        # Make sure we go back to the original directory
        try:
            os.chdir(original_dir)
        except FileNotFoundError:
            # If original directory is somehow no longer valid, use a safe fallback
            os.chdir(os.path.expanduser("~"))


@pytest.mark.skipif(api_keys_missing(), reason="API keys required for this test")
def test_complex_tasks(temp_dir: str) -> None:
    """Test that code_with_aider correctly implements more complex tasks."""
    # Create the test file for a calculator class
    test_file = os.path.join(temp_dir, "calculator.py")
    with open(test_file, "w") as f:
        f.write("# This file should implement a calculator class\n")

    # More complex prompt suitable for architect mode
    prompt = """
    Create a Calculator class with the following features:
    1. Basic operations: add, subtract, multiply, divide methods
    2. Memory functions: memory_store, memory_recall, memory_clear
    3. A history feature that keeps track of operations
    4. A method to show_history
    5. Error handling for division by zero

    All methods should be well-documented with docstrings.
    """

    # Run code_with_aider with an available and more stable model
    # Try multiple models in case one fails
    models_to_try = [
        "gemini/gemini-pro",  # More stable model as primary choice
        "gemini/gemini-1.5-pro",  # Alternative format
        "openai/gpt-3.5-turbo",  # Fallback to a different provider
    ]

    last_error = None
    result_dict: Any = None # Initialize result_dict

    for model in models_to_try:
        try:
            result = code_with_aider(
                ai_coding_prompt=prompt,
                relative_editable_files=[test_file],
                model=model,
                working_dir=temp_dir,
            )

            # Parse the JSON result
            result_dict = json.loads(result)

            # If this succeeded, break out of the loop
            if result_dict.get("success") is True:
                break

            # Otherwise, record the error but continue trying other models
            last_error = f"Model {model} did not produce successful changes. Result: {result_dict.get('diff', 'No diff provided')}"

        except Exception as e:
            last_error = f"Error with model {model}: {str(e)}"
            continue

    # Skip test if all models failed rather than failing the test
    if result_dict is None or result_dict.get("success") is False:
        pytest.skip(f"All models failed to generate code: {last_error}")

    # Check that it succeeded
    assert result_dict["success"] is True, (
        "Expected code_with_aider with architect mode to succeed"
    )
    assert "diff" in result_dict, "Expected diff to be in result"

    # Check that the file was modified correctly with expected elements
    with open(test_file, "r") as f:
        content = f.read()

    # Check for class definition and methods - relaxed assertions to accommodate type hints
    assert "class Calculator" in content, "Expected to find Calculator class definition"
    assert "add" in content, "Expected to find add method"
    assert "subtract" in content, "Expected to find subtract method"
    assert "multiply" in content, "Expected to find multiply method"
    assert "divide" in content, "Expected to find divide method"
    assert "memory_" in content, "Expected to find memory functions"
    assert "history" in content, "Expected to find history functionality"

    # Import and test basic calculator functionality
    import sys

    sys.path.append(temp_dir)
    # Use a try-except block for import in case the generated code is invalid
    try:
        from calculator import Calculator  # type: ignore
    except ImportError as e:
        pytest.fail(f"Failed to import Calculator class from generated code: {e}\nGenerated content:\n{content}")
    except Exception as e:
         pytest.fail(f"Unexpected error importing Calculator class: {e}\nGenerated content:\n{content}")


    # Test the calculator
    try:
        calc = Calculator()
    except Exception as e:
        pytest.fail(f"Failed to instantiate Calculator class: {e}\nGenerated content:\n{content}")


    # Test basic operations
    assert calc.add(2, 3) == 5, "Expected add(2, 3) to return 5"
    assert calc.subtract(5, 3) == 2, "Expected subtract(5, 3) to return 2"
    assert calc.multiply(2, 3) == 6, "Expected multiply(2, 3) to return 6"
    assert calc.divide(6, 3) == 2, "Expected divide(6, 3) to return 2"

    # Test division by zero error handling
    # Use a more specific check for the expected behavior (returning None or raising specific error)
    try:
        result = calc.divide(5, 0)
        assert result is None or isinstance(result, (str, type(None))), (
            "Expected divide by zero to return None or an error message string"
        )
    except ZeroDivisionError:
        # If it raises ZeroDivisionError, that's also acceptable error handling
        pass
    except Exception as e:
        pytest.fail(f"Unexpected exception during divide by zero test: {e}")


    # Test memory functions if implemented as expected
    # Use try-except for attribute/type errors in case methods weren't generated correctly
    try:
        calc.memory_store(10)
        assert calc.memory_recall() == 10, (
            "Expected memory_recall() to return stored value"
        )
        calc.memory_clear()
        assert calc.memory_recall() == 0 or calc.memory_recall() is None, (
            "Expected memory_recall() to return 0 or None after clearing"
        )
    except (AttributeError, TypeError) as e:
        pytest.fail(f"Memory function test failed: {e}\nGenerated content:\n{content}")
    except Exception as e:
        pytest.fail(f"Unexpected error during memory function test: {e}")

