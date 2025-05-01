import sys
import logging
import os
import signal # Import signal
from pathlib import Path # Import Path
from typing import Generator, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _pytest.tmpdir import TempPathFactory

# Use absolute imports from the package root
from aider_mcp_server.__main__ import main
from aider_mcp_server.atoms.atoms_utils import (
    DEFAULT_EDITOR_MODEL,
    DEFAULT_WS_HOST,
    DEFAULT_WS_PORT,
)
from aider_mcp_server.atoms.logging import Logger # Import Logger for spec


@pytest.fixture
def mock_serve() -> Generator[AsyncMock, None, None]:
    """Mock the serve function (for stdio mode)."""
    with patch("aider_mcp_server.__main__.serve", new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture
def mock_serve_sse() -> Generator[AsyncMock, None, None]:
    """Mock the serve_sse function."""
    with patch("aider_mcp_server.__main__.serve_sse", new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture
def mock_serve_multi() -> Generator[AsyncMock, None, None]:
    """Mock the serve_multi_transport function."""
    with patch("aider_mcp_server.__main__.serve_multi_transport", new_callable=AsyncMock) as mock:
        yield mock

# Remove mock_logger_warning, use mock_get_logger instead

@pytest.fixture
def mock_is_git_repository() -> Generator[MagicMock, None, None]:
    """Mock the is_git_repository check to always return True."""
    # Patch where it's used in __main__
    with patch("aider_mcp_server.__main__.is_git_repository", return_value=(True, None)) as mock:
        yield mock

@pytest.fixture
def mock_os_makedirs() -> Generator[MagicMock, None, None]:
    """Mock os.makedirs to avoid filesystem side effects."""
    with patch("aider_mcp_server.__main__.os.makedirs") as mock: # Patch where it's used
        yield mock

@pytest.fixture
def mock_get_logger() -> Generator[MagicMock, None, None]:
    """Mock get_logger to return a mock logger with mocked methods."""
    with patch("aider_mcp_server.__main__.get_logger") as mock_get_logger_factory:
        # Create a mock logger instance that conforms to the Logger protocol/class
        mock_logger_instance = MagicMock(spec=Logger)
        # Ensure all methods used in __main__ are mocked
        mock_logger_instance.warning = MagicMock()
        mock_logger_instance.info = MagicMock()
        mock_logger_instance.critical = MagicMock()
        mock_logger_instance.error = MagicMock()
        mock_logger_instance.exception = MagicMock()
        mock_logger_instance.debug = MagicMock() # Add debug if used
        # Set the return value of the factory mock
        mock_get_logger_factory.return_value = mock_logger_instance
        yield mock_get_logger_factory # Yield the factory mock


def run_main_with_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: TempPathFactory, # Use factory to create temp paths per call
    args: List[str],
    mock_is_git_repo: MagicMock,
    mock_os_mkdirs: MagicMock,
    mock_get_log: MagicMock,
) -> str:
    """Helper function to run main with specific arguments."""
    # Create a unique temp dir for each run using the factory
    test_dir_path = tmp_path_factory.mktemp("test_run")
    abs_test_dir = str(test_dir_path.resolve()) # __main__ resolves path

    # Ensure --current-working-dir is always present
    if "--current-working-dir" not in args:
        # Find placeholder or add it
        try:
            idx = args.index("temp_dir_placeholder")
            args[idx] = abs_test_dir
        except ValueError:
             args = ["--current-working-dir", abs_test_dir] + args
    else:
        # Replace placeholder if it exists
        args = [abs_test_dir if arg == "temp_dir_placeholder" else arg for arg in args]
        # Ensure the path provided to --current-working-dir is absolute for consistency
        try:
            cwd_idx = args.index("--current-working-dir")
            if cwd_idx + 1 < len(args):
                 # Only replace if it's the placeholder, otherwise assume it's intentional
                 if args[cwd_idx + 1] == "temp_dir_placeholder":
                      args[cwd_idx + 1] = abs_test_dir
                 # If not placeholder, ensure it's absolute (though __main__ handles this now)
                 # else:
                 #     args[cwd_idx + 1] = str(Path(args[cwd_idx + 1]).resolve())
        except ValueError: # Should not happen if check above passed
            pass


    full_args = ["prog"] + args
    monkeypatch.setattr(sys, "argv", full_args)

    # Mock signal handling setup
    mock_loop = MagicMock()
    mock_loop.add_signal_handler = MagicMock()
    # Patch where get_event_loop is called (__main__)
    with patch("aider_mcp_server.__main__.asyncio.get_event_loop", return_value=mock_loop):
        # Patch Path.resolve and Path.is_dir as __main__ uses them
        with patch.object(Path, "resolve") as mock_resolve, \
             patch.object(Path, "is_dir") as mock_is_dir:

            # Configure mocks for validation steps
            # Make resolve return a Path object that has the is_dir method mocked
            mock_resolved_path = MagicMock(spec=Path)
            mock_resolved_path.is_dir.return_value = True # Assume dir exists for successful runs
            mock_resolved_path.__str__.return_value = abs_test_dir # Return string path when needed
            mock_resolve.return_value = mock_resolved_path
            mock_is_dir.return_value = True # Also mock the direct call if used

            # Mock is_git_repository (already passed in)
            mock_is_git_repo.return_value = (True, None)

            # Run main
            main()

    return abs_test_dir


# --- Test Cases ---

def test_stdio_mode_default(
    mock_serve: AsyncMock, mock_serve_sse: AsyncMock, mock_serve_multi: AsyncMock,
    mock_is_git_repository: MagicMock, mock_os_makedirs: MagicMock, mock_get_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: TempPathFactory,
) -> None:
    """Test that stdio mode is selected by default."""
    args = ["--current-working-dir", "temp_dir_placeholder"]
    abs_test_dir = run_main_with_args(monkeypatch, tmp_path_factory, args, mock_is_git_repository, mock_os_makedirs, mock_get_logger)

    mock_serve.assert_called_once()
    call_args = mock_serve.call_args[1]
    assert call_args["current_working_dir"] == abs_test_dir # __main__ passes resolved string path
    assert call_args["editor_model"] == DEFAULT_EDITOR_MODEL
    mock_serve_sse.assert_not_called()
    mock_serve_multi.assert_not_called()
    # Check is_git_repository was called with a Path object
    mock_is_git_repository.assert_called_once()
    assert isinstance(mock_is_git_repository.call_args[0][0], Path)
    assert str(mock_is_git_repository.call_args[0][0]) == abs_test_dir


def test_stdio_mode_explicit(
    mock_serve: AsyncMock, mock_serve_sse: AsyncMock, mock_serve_multi: AsyncMock,
    mock_is_git_repository: MagicMock, mock_os_makedirs: MagicMock, mock_get_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: TempPathFactory,
) -> None:
    """Test explicit selection of stdio mode."""
    args = ["--server-mode", "stdio", "--current-working-dir", "temp_dir_placeholder"]
    abs_test_dir = run_main_with_args(monkeypatch, tmp_path_factory, args, mock_is_git_repository, mock_os_makedirs, mock_get_logger)

    mock_serve.assert_called_once()
    call_args = mock_serve.call_args[1]
    assert call_args["current_working_dir"] == abs_test_dir
    assert call_args["editor_model"] == DEFAULT_EDITOR_MODEL
    mock_serve_sse.assert_not_called()
    mock_serve_multi.assert_not_called()
    mock_is_git_repository.assert_called_once()
    assert isinstance(mock_is_git_repository.call_args[0][0], Path)


def test_sse_mode_default_host_port(
    mock_serve: AsyncMock, mock_serve_sse: AsyncMock, mock_serve_multi: AsyncMock,
    mock_is_git_repository: MagicMock, mock_os_makedirs: MagicMock, mock_get_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: TempPathFactory,
) -> None:
    """Test sse mode selection with default host and port."""
    args = ["--server-mode", "sse", "--current-working-dir", "temp_dir_placeholder"]
    abs_test_dir = run_main_with_args(monkeypatch, tmp_path_factory, args, mock_is_git_repository, mock_os_makedirs, mock_get_logger)

    mock_serve_sse.assert_called_once()
    call_args = mock_serve_sse.call_args[1]
    assert call_args["host"] == DEFAULT_WS_HOST
    assert call_args["port"] == DEFAULT_WS_PORT
    assert call_args["current_working_dir"] == abs_test_dir
    assert call_args["editor_model"] == DEFAULT_EDITOR_MODEL
    mock_serve.assert_not_called()
    mock_serve_multi.assert_not_called()
    mock_is_git_repository.assert_called_once()
    assert isinstance(mock_is_git_repository.call_args[0][0], Path)


def test_sse_mode_custom_host_port(
    mock_serve: AsyncMock, mock_serve_sse: AsyncMock, mock_serve_multi: AsyncMock,
    mock_is_git_repository: MagicMock, mock_os_makedirs: MagicMock, mock_get_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: TempPathFactory,
) -> None:
    """Test sse mode selection with custom host and port."""
    custom_host = "0.0.0.0"
    custom_port = 9999
    args = ["--server-mode", "sse", "--host", custom_host, "--port", str(custom_port), "--current-working-dir", "temp_dir_placeholder"]
    abs_test_dir = run_main_with_args(monkeypatch, tmp_path_factory, args, mock_is_git_repository, mock_os_makedirs, mock_get_logger)

    mock_serve_sse.assert_called_once()
    call_args = mock_serve_sse.call_args[1]
    assert call_args["host"] == custom_host
    assert call_args["port"] == custom_port
    assert call_args["current_working_dir"] == abs_test_dir
    assert call_args["editor_model"] == DEFAULT_EDITOR_MODEL
    mock_serve.assert_not_called()
    mock_serve_multi.assert_not_called()
    mock_is_git_repository.assert_called_once()
    assert isinstance(mock_is_git_repository.call_args[0][0], Path)


def test_multi_mode_default_host_port(
    mock_serve: AsyncMock, mock_serve_sse: AsyncMock, mock_serve_multi: AsyncMock,
    mock_is_git_repository: MagicMock, mock_os_makedirs: MagicMock, mock_get_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: TempPathFactory,
) -> None:
    """Test multi mode selection with default host and port."""
    args = ["--server-mode", "multi", "--current-working-dir", "temp_dir_placeholder"]
    abs_test_dir = run_main_with_args(monkeypatch, tmp_path_factory, args, mock_is_git_repository, mock_os_makedirs, mock_get_logger)

    mock_serve_multi.assert_called_once()
    call_args = mock_serve_multi.call_args[1]
    assert call_args["host"] == DEFAULT_WS_HOST
    assert call_args["port"] == DEFAULT_WS_PORT
    assert call_args["current_working_dir"] == abs_test_dir
    assert call_args["editor_model"] == DEFAULT_EDITOR_MODEL
    mock_serve.assert_not_called()
    mock_serve_sse.assert_not_called()
    mock_is_git_repository.assert_called_once()
    assert isinstance(mock_is_git_repository.call_args[0][0], Path)


def test_multi_mode_custom_host_port(
    mock_serve: AsyncMock, mock_serve_sse: AsyncMock, mock_serve_multi: AsyncMock,
    mock_is_git_repository: MagicMock, mock_os_makedirs: MagicMock, mock_get_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: TempPathFactory,
) -> None:
    """Test multi mode selection with custom host and port."""
    custom_host = "192.168.1.100"
    custom_port = 8080
    args = ["--server-mode", "multi", "--host", custom_host, "--port", str(custom_port), "--current-working-dir", "temp_dir_placeholder"]
    abs_test_dir = run_main_with_args(monkeypatch, tmp_path_factory, args, mock_is_git_repository, mock_os_makedirs, mock_get_logger)

    mock_serve_multi.assert_called_once()
    call_args = mock_serve_multi.call_args[1]
    assert call_args["host"] == custom_host
    assert call_args["port"] == custom_port
    assert call_args["current_working_dir"] == abs_test_dir
    assert call_args["editor_model"] == DEFAULT_EDITOR_MODEL
    mock_serve.assert_not_called()
    mock_serve_sse.assert_not_called()
    mock_is_git_repository.assert_called_once()
    assert isinstance(mock_is_git_repository.call_args[0][0], Path)


def test_stdio_mode_host_port_warning(
    mock_serve: AsyncMock, # Mock for the function called
    mock_is_git_repository: MagicMock, mock_os_makedirs: MagicMock, mock_get_logger: MagicMock, # Mocks for setup
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: TempPathFactory,
) -> None:
    """Test warning when --host and/or --port are provided with stdio mode."""
    custom_host = "1.2.3.4"
    custom_port = 1234
    logger_instance = mock_get_logger.return_value # Get the mocked logger instance

    # Test case 1: Only --host provided
    args_host = ["--server-mode", "stdio", "--host", custom_host, "--current-working-dir", "temp_dir_placeholder"]
    run_main_with_args(monkeypatch, tmp_path_factory, args_host, mock_is_git_repository, mock_os_makedirs, mock_get_logger)
    mock_serve.assert_called_once()
    logger_instance.warning.assert_called_once_with("Warning: --host and --port arguments are ignored in 'stdio' mode.")
    mock_serve.reset_mock()
    logger_instance.warning.reset_mock()
    mock_is_git_repository.reset_mock()

    # Test case 2: Only --port provided
    args_port = ["--server-mode", "stdio", "--port", str(custom_port), "--current-working-dir", "temp_dir_placeholder"]
    run_main_with_args(monkeypatch, tmp_path_factory, args_port, mock_is_git_repository, mock_os_makedirs, mock_get_logger)
    mock_serve.assert_called_once()
    logger_instance.warning.assert_called_once_with("Warning: --host and --port arguments are ignored in 'stdio' mode.")
    mock_serve.reset_mock()
    logger_instance.warning.reset_mock()
    mock_is_git_repository.reset_mock()

    # Test case 3: Both --host and --port provided
    args_both = ["--server-mode", "stdio", "--host", custom_host, "--port", str(custom_port), "--current-working-dir", "temp_dir_placeholder"]
    run_main_with_args(monkeypatch, tmp_path_factory, args_both, mock_is_git_repository, mock_os_makedirs, mock_get_logger)
    mock_serve.assert_called_once()
    logger_instance.warning.assert_called_once_with("Warning: --host and --port arguments are ignored in 'stdio' mode.")


def test_invalid_mode(
    mock_serve: AsyncMock, mock_serve_sse: AsyncMock, mock_serve_multi: AsyncMock,
    mock_is_git_repository: MagicMock, mock_os_makedirs: MagicMock, mock_get_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: TempPathFactory,
) -> None:
    """Test that providing an invalid mode exits the program."""
    # No need for CWD here as argparse fails first
    args = ["--server-mode", "invalid_mode", "--current-working-dir", "dummy"]

    with pytest.raises(SystemExit) as excinfo:
        # Use run_main_with_args, but expect SystemExit from argparse
        run_main_with_args(monkeypatch, tmp_path_factory, args, mock_is_git_repository, mock_os_makedirs, mock_get_logger)

    assert excinfo.value.code == 2 # Argparse exits with 2 for invalid choice
    mock_serve.assert_not_called()
    mock_serve_sse.assert_not_called()
    mock_serve_multi.assert_not_called()


def test_custom_editor_model_across_modes(
    mock_serve: AsyncMock, mock_serve_sse: AsyncMock, mock_serve_multi: AsyncMock,
    mock_is_git_repository: MagicMock, mock_os_makedirs: MagicMock, mock_get_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: TempPathFactory,
) -> None:
    """Test that custom editor model is passed correctly for all modes."""
    custom_model = "custom-ai-model-123"
    modes = ["stdio", "sse", "multi"]
    mocks = [mock_serve, mock_serve_sse, mock_serve_multi]

    for mode, mock_func in zip(modes, mocks):
        mock_serve.reset_mock()
        mock_serve_sse.reset_mock()
        mock_serve_multi.reset_mock()
        mock_is_git_repository.reset_mock()
        mock_os_makedirs.reset_mock()
        mock_get_logger.reset_mock()
        mock_get_logger.return_value.reset_mock() # Reset logger instance mocks too

        args = ["--server-mode", mode, "--editor-model", custom_model, "--current-working-dir", "temp_dir_placeholder"]
        if mode != "stdio": args.extend(["--host", "1.2.3.4", "--port", "5678"])

        abs_test_dir = run_main_with_args(monkeypatch, tmp_path_factory, args, mock_is_git_repository, mock_os_makedirs, mock_get_logger)

        mock_func.assert_called_once()
        call_args = mock_func.call_args[1]
        assert call_args["editor_model"] == custom_model
        assert call_args["current_working_dir"] == abs_test_dir
        mock_is_git_repository.assert_called_once()
        assert isinstance(mock_is_git_repository.call_args[0][0], Path)

        for other_mock in mocks:
            if other_mock is not mock_func: other_mock.assert_not_called()


def test_missing_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: TempPathFactory,
    mock_is_git_repository: MagicMock, mock_os_makedirs: MagicMock, mock_get_logger: MagicMock,
) -> None:
    """Test that missing --current-working-dir exits the program."""
    args = ["--server-mode", "stdio"] # Missing --current-working-dir

    with pytest.raises(SystemExit) as excinfo:
        # Use run_main_with_args, it handles setup, expect SystemExit from argparse
        run_main_with_args(monkeypatch, tmp_path_factory, args, mock_is_git_repository, mock_os_makedirs, mock_get_logger)

    assert excinfo.value.code == 2 # Argparse exits with 2 for missing required arg


def test_invalid_cwd_not_git_repo(
    mock_os_makedirs: MagicMock, mock_get_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: TempPathFactory,
) -> None:
    """Test that non-git repo CWD causes exit."""
    test_dir_path = tmp_path_factory.mktemp("not_a_repo")
    abs_test_dir = str(test_dir_path.resolve())
    args = ["--server-mode", "stdio", "--current-working-dir", abs_test_dir]
    git_error_msg = "Not a git repo"

    # Mock is_git_repository to return False
    with patch("aider_mcp_server.__main__.is_git_repository", return_value=(False, git_error_msg)) as mock_git_check:
        with pytest.raises(SystemExit) as excinfo:
            # Use run_main_with_args, it handles setup
            run_main_with_args(monkeypatch, tmp_path_factory, args, mock_git_check, mock_os_makedirs, mock_get_logger)

    assert excinfo.value.code == 1
    logger_instance = mock_get_logger.return_value
    logger_instance.critical.assert_called_once()
    assert "not a valid git repository" in logger_instance.critical.call_args[0][0]
    assert git_error_msg in logger_instance.critical.call_args[0][0]
    # Check is_git_repository was called with a Path object
    mock_git_check.assert_called_once()
    assert isinstance(mock_git_check.call_args[0][0], Path)
    assert str(mock_git_check.call_args[0][0]) == abs_test_dir


def test_invalid_cwd_not_a_directory(
    mock_os_makedirs: MagicMock, mock_get_logger: MagicMock,
    mock_is_git_repository: MagicMock, # Still need this for run_main_with_args signature
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: TempPathFactory,
) -> None:
    """Test that non-directory CWD causes exit."""
    test_file_path = tmp_path_factory.mktemp("test_run") / "not_a_dir.txt"
    test_file_path.touch()
    abs_test_path = str(test_file_path.resolve())
    args = ["--server-mode", "stdio", "--current-working-dir", abs_test_path]

    # Mock Path.resolve to return a path object that returns False for is_dir
    mock_resolved_path = MagicMock(spec=Path)
    mock_resolved_path.is_dir.return_value = False
    mock_resolved_path.__str__.return_value = abs_test_path # Ensure string representation is correct

    with patch.object(Path, "resolve", return_value=mock_resolved_path):
         # Also mock the direct Path.is_dir call if resolve isn't used consistently
         with patch.object(Path, "is_dir", return_value=False):
            with pytest.raises(SystemExit) as excinfo:
                # Use run_main_with_args, it handles setup
                run_main_with_args(monkeypatch, tmp_path_factory, args, mock_is_git_repository, mock_os_makedirs, mock_get_logger)

    assert excinfo.value.code == 1
    logger_instance = mock_get_logger.return_value
    logger_instance.critical.assert_called_once()
    # The error message comes from the resolve(strict=True) check now
    assert "Specified working directory does not exist" in logger_instance.critical.call_args[0][0]
    assert abs_test_path in logger_instance.critical.call_args[0][0]
    mock_is_git_repository.assert_not_called() # Should fail before git check
