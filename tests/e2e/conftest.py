from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_popen():
    """
    Mocks subprocess.Popen to return a mock process object.
    The mock process object's communicate method returns dummy stdout/stderr.
    """
    # Create a mock object for the process returned by Popen
    mock_process = MagicMock()

    # Configure the communicate method to return the desired tuple of bytes
    mock_process.communicate.return_value = (b"stdout data", b"stderr data")

    # Set the returncode attribute
    mock_process.returncode = 0

    # Patch subprocess.Popen to return our mock process object
    with patch("subprocess.Popen", return_value=mock_process) as _mock_popen:
        yield _mock_popen  # Yield the patch object itself if needed, or just the mock_process

    # Note: The patch context manager handles restoring the original Popen


# Example usage in a test function:
# def test_something_using_subprocess(mock_popen):
#     # Your code that calls subprocess.Popen
#     # The call will now return the mock_process configured above
#     pass
