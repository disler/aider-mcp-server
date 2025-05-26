import asyncio  # Import asyncio
import os
import sys
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import pytest

# Use absolute imports from the package root
# Import the cli module for testing
import aider_mcp_server.templates.initialization.cli as cli_module

# Import Logger for mocking
from aider_mcp_server.atoms.logging.logger import Logger
from aider_mcp_server.atoms.utils.config_constants import (
    DEFAULT_EDITOR_MODEL,
    DEFAULT_WS_HOST,
    DEFAULT_WS_PORT,
)


# Helper function to run the main function with specific args
def run_main(
    monkeypatch: pytest.MonkeyPatch, args: List[str]
) -> Tuple[
    mock.MagicMock,
    mock.MagicMock,
    mock.MagicMock,
    mock.MagicMock,
    mock.MagicMock,
    mock.MagicMock,
    mock.MagicMock,
    mock.MagicMock,
]:
    """Runs the CLI main function with mocked dependencies."""
    mock_serve = mock.MagicMock()  # Use MagicMock since this is just passing it to setattr
    mock_serve_sse = mock.MagicMock()
    mock_serve_multi = mock.MagicMock()
    mock_logger_instance = mock.MagicMock(spec=Logger)  # Mock the logger instance
    # Mock the get_logger function *where it's used* in __main__
    mock_get_logger = mock.MagicMock(return_value=mock_logger_instance)
    # Mock Path.is_dir *where it's used* in __main__ (via Path instance)
    mock_path_is_dir = mock.MagicMock(return_value=True)
    # Mock is_git_repository *where it's used* in __main__
    mock_is_git_repo = mock.MagicMock(return_value=(True, None))  # Default to True
    mock_exit = mock.MagicMock(side_effect=SystemExit)  # Raise SystemExit
    # Mock asyncio.run to just execute the coroutine directly for testing
    mock_asyncio_run = mock.MagicMock()

    def run_sync(coro):
        # A simple way to run the async function in tests if needed,
        # or just check if it was called with the right coroutine.
        # For these tests, we mainly care *that* the serve functions are called.
        pass  # We check the mock_serve* calls instead

    mock_asyncio_run.side_effect = run_sync

    # Patch the functions and classes used within __main__.py's scope
    monkeypatch.setattr(cli_module, "serve", mock_serve)
    monkeypatch.setattr(cli_module, "serve_sse", mock_serve_sse)
    monkeypatch.setattr(cli_module, "serve_multi_transport", mock_serve_multi)
    monkeypatch.setattr(cli_module, "get_logger", mock_get_logger)
    # Patch Path.is_dir globally as __main__ uses Path objects now
    monkeypatch.setattr(Path, "is_dir", mock_path_is_dir)

    # Patch Path.resolve globally - assume it works unless specifically testing failure
    # Let's mock resolve to return a predictable object based on input
    def simple_resolve(self, strict=False):
        # Return a new Path object based on the input string for consistency
        # In a real scenario, this might need more sophisticated mocking if paths matter
        resolved_path = Path(os.path.abspath(str(self)))
        if strict and not resolved_path.exists():  # Basic check
            raise FileNotFoundError(f"Mock FileNotFoundError for {self}")
        return resolved_path

    monkeypatch.setattr(Path, "resolve", simple_resolve)

    monkeypatch.setattr(cli_module, "is_git_repository", mock_is_git_repo)
    monkeypatch.setattr(sys, "exit", mock_exit)
    monkeypatch.setattr(asyncio, "run", mock_asyncio_run)  # Mock asyncio.run

    # Mock sys.argv for this specific run
    monkeypatch.setattr(sys, "argv", ["script_name"] + args)

    try:
        cli_module.main()
    except SystemExit:
        pass  # Capture SystemExit raised by mock_exit or argparse error

    return (
        mock_serve,
        mock_serve_sse,
        mock_serve_multi,
        mock_get_logger,
        mock_logger_instance,
        mock_path_is_dir,
        mock_is_git_repo,
        mock_exit,
    )


# --- Test Cases ---


def test_default_stdio_mode(monkeypatch: pytest.MonkeyPatch):
    """Test default mode is stdio."""
    # Provide the required CWD argument
    args = ["--current-working-dir", "."]
    (
        mock_serve,
        mock_serve_sse,
        mock_serve_multi,
        _,
        _,
        mock_path_is_dir,
        mock_is_git_repo,
        mock_exit,
    ) = run_main(monkeypatch, args)

    cwd_path = Path(".").resolve()  # __main__ resolves the path
    # Check that Path.is_dir was called
    mock_path_is_dir.assert_called_once()
    # REMOVED: assert mock_path_is_dir.call_args[0][0] == cwd_path # This was incorrect
    # Check that is_git_repository was called with the resolved Path object
    mock_is_git_repo.assert_called_once_with(cwd_path)
    mock_serve.assert_called_once()
    call_args, call_kwargs = mock_serve.call_args
    assert call_kwargs.get("editor_model") == DEFAULT_EDITOR_MODEL
    assert call_kwargs.get("current_working_dir") == str(cwd_path)  # __main__ passes string path
    mock_serve_sse.assert_not_called()
    mock_serve_multi.assert_not_called()
    mock_exit.assert_not_called()


def test_explicit_stdio_mode(monkeypatch: pytest.MonkeyPatch):
    """Test explicit stdio mode selection."""
    args = ["--server-mode", "stdio", "--current-working-dir", "."]
    (
        mock_serve,
        mock_serve_sse,
        mock_serve_multi,
        _,
        _,
        mock_path_is_dir,
        mock_is_git_repo,
        mock_exit,
    ) = run_main(monkeypatch, args)

    cwd_path = Path(".").resolve()
    mock_path_is_dir.assert_called_once()
    # REMOVED: assert mock_path_is_dir.call_args[0][0] == cwd_path
    mock_is_git_repo.assert_called_once_with(cwd_path)
    mock_serve.assert_called_once()
    mock_serve_sse.assert_not_called()
    mock_serve_multi.assert_not_called()
    mock_exit.assert_not_called()


def test_sse_mode_default_host_port(monkeypatch: pytest.MonkeyPatch):
    """Test SSE mode with default host and port."""
    args = ["--server-mode", "sse", "--current-working-dir", "."]
    (
        mock_serve,
        mock_serve_sse,
        mock_serve_multi,
        _,
        _,
        mock_path_is_dir,
        mock_is_git_repo,
        mock_exit,
    ) = run_main(monkeypatch, args)

    cwd_path = Path(".").resolve()
    mock_path_is_dir.assert_called_once()
    # REMOVED: assert mock_path_is_dir.call_args[0][0] == cwd_path
    mock_is_git_repo.assert_called_once_with(cwd_path)
    mock_serve_sse.assert_called_once()
    call_args, call_kwargs = mock_serve_sse.call_args
    assert call_kwargs.get("host") == DEFAULT_WS_HOST
    assert call_kwargs.get("port") == DEFAULT_WS_PORT
    assert call_kwargs.get("editor_model") == DEFAULT_EDITOR_MODEL
    assert call_kwargs.get("current_working_dir") == str(cwd_path)  # Passes string path
    mock_serve.assert_not_called()
    mock_serve_multi.assert_not_called()
    mock_exit.assert_not_called()


def test_sse_mode_custom_host_port(monkeypatch: pytest.MonkeyPatch):
    """Test SSE mode with custom host and port."""
    custom_host = "192.168.1.100"
    custom_port = 9999
    args = [
        "--server-mode",
        "sse",
        "--host",
        custom_host,
        "--port",
        str(custom_port),
        "--current-working-dir",
        ".",
    ]
    (
        mock_serve,
        mock_serve_sse,
        mock_serve_multi,
        _,
        _,
        mock_path_is_dir,
        mock_is_git_repo,
        mock_exit,
    ) = run_main(monkeypatch, args)

    cwd_path = Path(".").resolve()
    mock_path_is_dir.assert_called_once()
    # REMOVED: assert mock_path_is_dir.call_args[0][0] == cwd_path
    mock_is_git_repo.assert_called_once_with(cwd_path)
    mock_serve_sse.assert_called_once()
    call_args, call_kwargs = mock_serve_sse.call_args
    assert call_kwargs.get("host") == custom_host
    assert call_kwargs.get("port") == custom_port
    mock_serve.assert_not_called()
    mock_serve_multi.assert_not_called()
    mock_exit.assert_not_called()


def test_multi_mode_default_host_port(monkeypatch: pytest.MonkeyPatch):
    """Test multi mode with default host and port."""
    args = ["--server-mode", "multi", "--current-working-dir", "."]
    (
        mock_serve,
        mock_serve_sse,
        mock_serve_multi,
        _,
        _,
        mock_path_is_dir,
        mock_is_git_repo,
        mock_exit,
    ) = run_main(monkeypatch, args)

    cwd_path = Path(".").resolve()
    mock_path_is_dir.assert_called_once()
    # REMOVED: assert mock_path_is_dir.call_args[0][0] == cwd_path
    mock_is_git_repo.assert_called_once_with(cwd_path)
    mock_serve_multi.assert_called_once()
    call_args, call_kwargs = mock_serve_multi.call_args
    assert call_kwargs.get("host") == DEFAULT_WS_HOST
    assert call_kwargs.get("port") == DEFAULT_WS_PORT
    assert call_kwargs.get("editor_model") == DEFAULT_EDITOR_MODEL
    assert call_kwargs.get("current_working_dir") == str(cwd_path)  # Passes string path
    mock_serve.assert_not_called()
    mock_serve_sse.assert_not_called()
    mock_exit.assert_not_called()


def test_multi_mode_custom_host_port(monkeypatch: pytest.MonkeyPatch):
    """Test multi mode with custom host and port."""
    custom_host = "127.0.0.1"  # Use loopback instead of all interfaces
    custom_port = 8080
    args = [
        "--server-mode",
        "multi",
        "--host",
        custom_host,
        "--port",
        str(custom_port),
        "--current-working-dir",
        ".",
    ]
    (
        mock_serve,
        mock_serve_sse,
        mock_serve_multi,
        _,
        _,
        mock_path_is_dir,
        mock_is_git_repo,
        mock_exit,
    ) = run_main(monkeypatch, args)

    cwd_path = Path(".").resolve()
    mock_path_is_dir.assert_called_once()
    # REMOVED: assert mock_path_is_dir.call_args[0][0] == cwd_path
    mock_is_git_repo.assert_called_once_with(cwd_path)
    mock_serve_multi.assert_called_once()
    call_args, call_kwargs = mock_serve_multi.call_args
    assert call_kwargs.get("host") == custom_host
    assert call_kwargs.get("port") == custom_port
    mock_serve.assert_not_called()
    mock_serve_sse.assert_not_called()
    mock_exit.assert_not_called()


@pytest.mark.parametrize(
    "mode_args", [["--server-mode", "stdio"], ["--server-mode", "stdio"]]
)  # Default is stdio, test explicit too
@pytest.mark.parametrize("extra_args", [["--host", "1.1.1.1"], ["--port", "1234"]])
def test_stdio_mode_host_port_warning(monkeypatch: pytest.MonkeyPatch, mode_args: List[str], extra_args: List[str]):
    """Test warning when host/port are used with stdio mode."""
    args = mode_args + extra_args + ["--current-working-dir", "."]
    (
        mock_serve,
        _,
        _,
        _,
        mock_logger_instance,
        mock_path_is_dir,
        mock_is_git_repo,
        mock_exit,
    ) = run_main(monkeypatch, args)

    cwd_path = Path(".").resolve()
    mock_path_is_dir.assert_called_once()
    # REMOVED: assert mock_path_is_dir.call_args[0][0] == cwd_path
    mock_is_git_repo.assert_called_once_with(cwd_path)
    mock_serve.assert_called_once()
    mock_logger_instance.warning.assert_called_once_with(
        "Warning: --host and --port arguments are ignored in 'stdio' mode."
    )
    mock_exit.assert_not_called()


def test_custom_editor_model_stdio(monkeypatch: pytest.MonkeyPatch):
    """Test custom editor model parameter with stdio mode."""
    model = "test-editor-model"
    args = ["--editor-model", model, "--current-working-dir", "."]
    mock_serve, _, _, _, _, mock_path_is_dir, mock_is_git_repo, mock_exit = run_main(monkeypatch, args)

    cwd_path = Path(".").resolve()
    mock_path_is_dir.assert_called_once()
    # REMOVED: assert mock_path_is_dir.call_args[0][0] == cwd_path
    mock_is_git_repo.assert_called_once_with(cwd_path)
    mock_serve.assert_called_once()
    call_args, call_kwargs = mock_serve.call_args
    assert call_kwargs.get("editor_model") == model
    mock_exit.assert_not_called()


def test_custom_editor_model_sse(monkeypatch: pytest.MonkeyPatch):
    """Test custom editor model parameter with sse mode."""
    model = "test-editor-model-sse"
    args = [
        "--server-mode",
        "sse",
        "--editor-model",
        model,
        "--current-working-dir",
        ".",
    ]
    _, mock_serve_sse, _, _, _, mock_path_is_dir, mock_is_git_repo, mock_exit = run_main(monkeypatch, args)

    cwd_path = Path(".").resolve()
    mock_path_is_dir.assert_called_once()
    # REMOVED: assert mock_path_is_dir.call_args[0][0] == cwd_path
    mock_is_git_repo.assert_called_once_with(cwd_path)
    mock_serve_sse.assert_called_once()
    call_args, call_kwargs = mock_serve_sse.call_args
    assert call_kwargs.get("editor_model") == model
    mock_exit.assert_not_called()


def test_custom_editor_model_multi(monkeypatch: pytest.MonkeyPatch):
    """Test custom editor model parameter with multi mode."""
    model = "test-editor-model-multi"
    args = [
        "--server-mode",
        "multi",
        "--editor-model",
        model,
        "--current-working-dir",
        ".",
    ]
    _, _, mock_serve_multi, _, _, mock_path_is_dir, mock_is_git_repo, mock_exit = run_main(monkeypatch, args)

    cwd_path = Path(".").resolve()
    mock_path_is_dir.assert_called_once()
    # REMOVED: assert mock_path_is_dir.call_args[0][0] == cwd_path
    mock_is_git_repo.assert_called_once_with(cwd_path)
    mock_serve_multi.assert_called_once()
    call_args, call_kwargs = mock_serve_multi.call_args
    assert call_kwargs.get("editor_model") == model
    mock_exit.assert_not_called()


def test_working_dir_not_exists(monkeypatch: pytest.MonkeyPatch):
    """Test validation failure if working directory does not exist."""
    non_existent_dir_str = "/path/to/non/existent/dir"
    args = ["--current-working-dir", non_existent_dir_str]

    # Mock Path.resolve to raise FileNotFoundError for the specific path
    def mock_resolve(self, strict=False):
        # Use os.path.abspath for basic resolution simulation
        abs_path_str = os.path.abspath(str(self))
        if abs_path_str == os.path.abspath(non_existent_dir_str) and strict:
            raise FileNotFoundError(f"[Errno 2] No such file or directory: '{abs_path_str}'")
        # For other paths or non-strict calls, return a Path object
        return Path(abs_path_str)

    monkeypatch.setattr(Path, "resolve", mock_resolve)
    # Mock Path.exists needed by the simple_resolve mock above
    original_exists = Path.exists

    def mock_exists(self):
        if str(self) == os.path.abspath(non_existent_dir_str):
            return False
        return original_exists(self)  # Allow checking real paths like '.'

    monkeypatch.setattr(Path, "exists", mock_exists)

    # No need to mock is_dir or is_git_repo as resolve(strict=True) fails first

    # Set up other mocks needed by run_main
    mock_serve = mock.MagicMock()
    mock_serve_sse = mock.MagicMock()
    mock_serve_multi = mock.MagicMock()
    mock_logger_instance = mock.MagicMock(spec=Logger)
    mock_get_logger = mock.MagicMock(return_value=mock_logger_instance)
    mock_is_git_repo = mock.MagicMock()  # Won't be called
    mock_exit = mock.MagicMock(side_effect=SystemExit)
    mock_asyncio_run = mock.MagicMock()  # Mock asyncio.run

    monkeypatch.setattr(cli_module, "serve", mock_serve)
    monkeypatch.setattr(cli_module, "serve_sse", mock_serve_sse)
    monkeypatch.setattr(cli_module, "serve_multi_transport", mock_serve_multi)
    monkeypatch.setattr(cli_module, "get_logger", mock_get_logger)
    monkeypatch.setattr(cli_module, "is_git_repository", mock_is_git_repo)
    monkeypatch.setattr(sys, "exit", mock_exit)
    monkeypatch.setattr(asyncio, "run", mock_asyncio_run)  # Mock asyncio.run
    monkeypatch.setattr(sys, "argv", ["script_name"] + args)

    # Run cli_module, expecting it to exit
    with pytest.raises(SystemExit):
        cli_module.main()

    # Assertions
    mock_logger_instance.critical.assert_called_once()
    error_message = mock_logger_instance.critical.call_args[0][0]
    # Check that the original path provided by the user is in the error message
    assert f"Specified working directory does not exist: {non_existent_dir_str}" in error_message
    mock_exit.assert_called_once_with(1)
    mock_serve.assert_not_called()
    mock_serve_sse.assert_not_called()
    mock_serve_multi.assert_not_called()


def test_working_dir_not_git_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test validation failure if working directory is not a git repository."""
    not_git_dir = tmp_path / "not_a_repo"
    not_git_dir.mkdir()
    not_git_dir_resolved = not_git_dir.resolve()  # Resolve it for assertion comparison
    git_error_message = "fatal: not a git repository"
    args = ["--current-working-dir", str(not_git_dir)]

    # Set up mocks *before* running main
    mock_serve = mock.AsyncMock()
    mock_serve_sse = mock.AsyncMock()
    mock_serve_multi = mock.AsyncMock()
    mock_logger_instance = mock.MagicMock(spec=Logger)
    mock_get_logger = mock.MagicMock(return_value=mock_logger_instance)
    # Mock Path.is_dir to return True for this path
    # We need to ensure our mock_resolve (if active from run_main) handles this path
    monkeypatch.setattr(Path, "is_dir", mock.MagicMock(return_value=True))
    # Mock is_git_repository to return False
    mock_is_git_repo = mock.MagicMock(return_value=(False, git_error_message))
    mock_exit = mock.MagicMock(side_effect=SystemExit)
    mock_asyncio_run = mock.MagicMock()  # Mock asyncio.run

    # Patch necessary components
    monkeypatch.setattr(cli_module, "serve", mock_serve)
    monkeypatch.setattr(cli_module, "serve_sse", mock_serve_sse)
    monkeypatch.setattr(cli_module, "serve_multi_transport", mock_serve_multi)
    monkeypatch.setattr(cli_module, "get_logger", mock_get_logger)
    # Patch Path.resolve to handle the tmp_path correctly
    original_resolve = Path.resolve

    def specific_resolve(self, strict=False):
        # Let the original resolve work for tmp_path
        return original_resolve(self, strict)

    monkeypatch.setattr(Path, "resolve", specific_resolve)
    monkeypatch.setattr(cli_module, "is_git_repository", mock_is_git_repo)
    monkeypatch.setattr(sys, "exit", mock_exit)
    monkeypatch.setattr(asyncio, "run", mock_asyncio_run)  # Mock asyncio.run
    monkeypatch.setattr(sys, "argv", ["script_name"] + args)

    # Run cli_module, expecting it to exit
    with pytest.raises(SystemExit):
        cli_module.main()

    # Assertions
    # Check that is_git_repository was called with the *resolved* Path object
    mock_is_git_repo.assert_called_once_with(not_git_dir_resolved)
    mock_logger_instance.critical.assert_called_once()
    error_message = mock_logger_instance.critical.call_args[0][0]
    assert f"Specified working directory is not a valid git repository: {not_git_dir_resolved}" in error_message
    assert git_error_message in error_message
    mock_exit.assert_called_once_with(1)
    mock_serve.assert_not_called()
    mock_serve_sse.assert_not_called()
    mock_serve_multi.assert_not_called()


def test_working_dir_valid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test successful validation with a valid working directory."""
    valid_dir = tmp_path / "valid_repo"
    valid_dir.mkdir()
    # Simulate it being a git repo by mocking is_git_repository to return True
    # The actual directory doesn't need git init for this test because of mocking.
    valid_dir_resolved = valid_dir.resolve()
    args = ["--current-working-dir", str(valid_dir)]

    # Use run_main which sets default mocks for success
    # Need to override the default Path.resolve mock in run_main to handle tmp_path
    original_resolve = Path.resolve

    def specific_resolve(self, strict=False):
        return original_resolve(self, strict)  # Use real resolve for tmp_path

    monkeypatch.setattr(Path, "resolve", specific_resolve)

    (
        mock_serve,
        _,
        _,
        _,
        mock_logger_instance,
        mock_path_is_dir,
        mock_is_git_repo,
        mock_exit,
    ) = run_main(monkeypatch, args)

    # Check validation calls happened correctly
    mock_path_is_dir.assert_called_once()
    # REMOVED: assert mock_path_is_dir.call_args[0][0] == valid_dir_resolved # Incorrect assertion
    mock_is_git_repo.assert_called_once_with(valid_dir_resolved)  # Check Path object passed

    # Check that the correct serve function (default stdio) was called
    mock_serve.assert_called_once()
    call_args, call_kwargs = mock_serve.call_args
    assert call_kwargs.get("current_working_dir") == str(valid_dir_resolved)  # Passes string path
    assert call_kwargs.get("editor_model") == DEFAULT_EDITOR_MODEL
    mock_logger_instance.critical.assert_not_called()  # Check critical log not called
    mock_exit.assert_not_called()
