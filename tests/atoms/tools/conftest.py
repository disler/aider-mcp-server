"""
Pytest fixtures for Aider tool tests.
"""

import os

import pytest

from .test_mock_api_keys import patch_aider_modules, patch_api_keys


@pytest.fixture(autouse=True)
def setup_api_keys():
    """Automatically set up mock API keys for all tests in this directory."""
    # Save original environment
    original_env = os.environ.copy()

    # Apply mock API keys
    patch_api_keys()

    yield

    # Restore original environment
    for key in list(os.environ.keys()):
        if key not in original_env:
            del os.environ[key]
        else:
            os.environ[key] = original_env[key]


@pytest.fixture
def mock_aider():
    """Set up mocks for Aider modules."""
    # Get the patches
    coder_patch, model_patch = patch_aider_modules()

    # Start the patches
    coder_mock = coder_patch.start()
    model_mock = model_patch.start()

    yield coder_mock, model_mock

    # Stop the patches
    coder_patch.stop()
    model_patch.stop()
