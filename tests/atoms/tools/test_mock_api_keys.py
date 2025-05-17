"""
Mocks for aider library to allow tests to run without real API keys.
"""

import os
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest


class MockModel:
    """Mock implementation of aider.models.Model class."""

    def __init__(self, model_name: str, editor_model: Optional[str] = None):
        self.model_name = model_name
        self.editor_model = editor_model

    def commit_message_models(self):
        """Return mock model list for commit messages."""
        return []


class MockInputOutput:
    """Mock implementation of aider.io.InputOutput class."""

    def __init__(self, **kwargs):
        self.pretty = kwargs.get("pretty", False)
        self.yes = kwargs.get("yes", True)
        self.fancy_input = kwargs.get("fancy_input", False)
        self.chat_history_file = kwargs.get("chat_history_file", None)
        self.yes_to_all = True


class MockGitRepo:
    """Mock implementation of aider.repo.GitRepo class."""

    def __init__(self, **kwargs):
        # Accept any arguments to make it flexible
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensure required attributes are set based on actual GitRepo parameters
        if not hasattr(self, "root") and "git_dname" in kwargs:
            self.root = kwargs["git_dname"]
        elif not hasattr(self, "root") and "working_dir" in kwargs:
            # Handle old parameter name for backward compatibility
            self.root = kwargs["working_dir"]


class MockCoder:
    """Mock implementation of aider.coders.Coder class."""

    @classmethod
    def create(cls, **kwargs):
        mock_coder = MagicMock()
        # Store kwargs for inspection
        mock_coder.kwargs = kwargs

        # Define run method that simulates creating/modifying files
        def run_fn(prompt):
            # Modify files when run is called
            for file_path in kwargs.get("fnames", []):
                if os.path.exists(file_path):
                    # For tests requiring file creation/modification
                    with open(file_path, "r") as f:
                        content = f.read()

                    # Add mock implementation based on prompt
                    if "add" in prompt.lower() and "math_add.py" in file_path:
                        new_content = content + "\n\ndef add(a, b):\n    return a + b\n"
                    elif "subtract" in prompt.lower() and "math_subtract.py" in file_path:
                        new_content = content + "\n\ndef subtract(a, b):\n    return a - b\n"
                    elif "multiply" in prompt.lower() and "math_multiply.py" in file_path:
                        new_content = content + "\n\ndef multiply(a, b):\n    return a * b\n"
                    elif "divide" in prompt.lower() and "math_divide.py" in file_path:
                        new_content = (
                            content + "\n\ndef divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n"
                        )
                    elif "calculator" in prompt.lower() and "calculator.py" in file_path:
                        new_content = (
                            content
                            + "\n\nclass Calculator:\n    def __init__(self):\n        self.memory = 0\n        self.history = []\n\n"
                            + "    def add(self, a, b):\n        result = a + b\n        self.history.append(f'{a} + {b} = {result}')\n        return result\n\n"
                            + "    def subtract(self, a, b):\n        result = a - b\n        self.history.append(f'{a} - {b} = {result}')\n        return result\n\n"
                            + "    def multiply(self, a, b):\n        result = a * b\n        self.history.append(f'{a} * {b} = {result}')\n        return result\n\n"
                            + "    def divide(self, a, b):\n        if b == 0:\n            return None\n        result = a / b\n        self.history.append(f'{a} / {b} = {result}')\n        return result\n\n"
                            + "    def memory_store(self, value):\n        self.memory = value\n\n"
                            + "    def memory_recall(self):\n        return self.memory\n\n"
                            + "    def memory_clear(self):\n        self.memory = 0\n\n"
                            + "    def show_history(self):\n        return self.history\n"
                        )
                    else:
                        # Default implementation for other cases
                        new_content = content + "\n\n# Mock implementation\n"

                    with open(file_path, "w") as f:
                        f.write(new_content)

            # For test_failure_case, if using non-existent model, don't modify files
            if "non_existent_model_123456789" in str(kwargs.get("main_model")):
                # Don't modify the files at all and return error message
                return "Error: The model non_existent_model_123456789 does not exist."

            return "Mock Aider run completed."

        mock_coder.run = run_fn
        return mock_coder


def patch_api_keys():
    """Set up mock API keys in the environment."""
    env_keys = {
        "OPENAI_API_KEY": "sk-mock-openai-key",
        "ANTHROPIC_API_KEY": "sk-mock-anthropic-key",
        "GOOGLE_API_KEY": "mock-google-key",
        "GEMINI_API_KEY": "mock-gemini-key",
    }

    for key, value in env_keys.items():
        os.environ[key] = value

    return env_keys


def mock_api_keys():
    """Legacy function for setting up mock API keys, kept for compatibility."""
    return patch_api_keys()


def is_valid_mock_model(model_str: str) -> bool:
    """Check if a model string is considered valid in the mock implementation."""
    # Allow all models except our specific test case for invalid model
    return "non_existent_model_123456789" not in model_str


def patch_aider_modules() -> Tuple[patch, patch]:
    """
    Create patches for Aider modules.

    Returns:
        Tuple of (coder_patch, model_patch) that can be used for patching.
    """
    coder_patch = patch("aider_mcp_server.atoms.tools.aider_ai_code.Coder", MockCoder)
    model_patch = patch("aider_mcp_server.atoms.tools.aider_ai_code.Model", MockModel)
    return coder_patch, model_patch


@pytest.fixture(autouse=True)
def setup_mock_aider(monkeypatch):
    """
    Set up mock for aider library components.

    This fixture runs automatically for any test in the same module.
    """
    # Set up mock API keys
    mock_api_keys()

    # Mock aider.models.Model
    monkeypatch.setattr("aider_mcp_server.atoms.tools.aider_ai_code.Model", MockModel)

    # Mock aider.io.InputOutput
    monkeypatch.setattr("aider_mcp_server.atoms.tools.aider_ai_code.InputOutput", MockInputOutput)

    # Mock aider.coders.Coder
    monkeypatch.setattr("aider_mcp_server.atoms.tools.aider_ai_code.Coder", MockCoder)

    # Patch the try/except block for importing GitRepo in the implementation
    # Use a modified __import__ to return our mock when it tries to import GitRepo from aider.repo
    def mock_import_from(name, globals_dict=None, locals_dict=None, fromlist=(), level=0):
        if name == "aider.repo" and "GitRepo" in fromlist:
            # Return a module-like object with a GitRepo attribute
            class MockModule:
                GitRepo = MockGitRepo

            return MockModule()
        # For other imports, use the real __import__
        return original_import(name, globals_dict, locals_dict, fromlist, level)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", mock_import_from)
