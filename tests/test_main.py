import sys
from pathlib import Path  # Import Path
from typing import Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _pytest.tmpdir import TempPathFactory

from aider_mcp_server.atoms.logging.logger import Logger  # Import Logger for spec
from aider_mcp_server.atoms.utils.config_constants import (
    DEFAULT_EDITOR_MODEL,
)

# Use absolute imports from the package root
# Import main from cli instead of __main__ to prevent RuntimeWarning
from aider_mcp_server.templates.initialization.cli import main


@pytest.fixture
def mock_serve() -> Generator[AsyncMock, None, None]:
    """Mock the serve function (for stdio mode)."""
    with patch("aider_mcp_server.templates.initialization.cli.serve", new_callable=AsyncMock) as mock:
        yield mock


@pytest.fixture
def mock_serve_sse() -> Generator[AsyncMock, None, None]:
    """Mock the serve_sse function."""
    with patch("aider_mcp_server.templates.initialization.cli.serve_sse", new_callable=AsyncMock) as mock:
        yield mock


# Remove mock_logger_warning, use mock_get_logger instead


@pytest.fixture
def mock_is_git_repository() -> Generator[MagicMock, None, None]:
    """Mock the is_git_repository check to always return True."""
    # Patch where it's used in cli
    with patch("aider_mcp_server.templates.initialization.cli.is_git_repository", return_value=(True, None)) as mock:
        yield mock


@pytest.fixture
def mock_os_makedirs() -> Generator[MagicMock, None, None]:
    """Mock os.makedirs to avoid filesystem side effects."""
    with patch("aider_mcp_server.templates.initialization.cli.os.makedirs") as mock:  # Patch where it's used
        yield mock


@pytest.fixture
def mock_get_logger() -> Generator[MagicMock, None, None]:
    """Mock get_logger to return a mock logger with mocked methods."""
    with patch("aider_mcp_server.templates.initialization.cli.get_logger", autospec=True) as mock_get_logger_factory:
        # Create a mock logger instance that conforms to the Logger protocol/class
        mock_logger_instance = MagicMock(spec=Logger)
        # Ensure all methods used in cli are mocked
        # Ensure all methods used in cli are mocked and will be tracked
        for method in ["debug", "info", "warning", "error", "critical", "exception"]:
            setattr(mock_logger_instance, method, MagicMock())
        # Set the return value of the factory mock
        mock_get_logger_factory.return_value = mock_logger_instance
        yield mock_get_logger_factory  # Yield the factory mock


def run_main_with_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: TempPathFactory,  # Use factory to create temp paths per call
    args: List[str],
    mock_is_git_repo: MagicMock,
    mock_os_mkdirs: MagicMock,
    mock_get_logger: MagicMock,
) -> str:
    """Helper function to run main with specific arguments."""
    # Create a unique temp dir for each run using the factory with a unique name
    import uuid

    unique_name = f"test_run_{uuid.uuid4().hex[:8]}"  # Use first 8 chars of a UUID
    test_dir_path = tmp_path_factory.mktemp(unique_name)
    abs_test_dir = str(test_dir_path.resolve())  # __main__ resolves path

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
        except ValueError:  # Should not happen if check above passed
            pass

    full_args = ["prog"] + args
    monkeypatch.setattr(sys, "argv", full_args)

    # Mock signal handling setup
    mock_loop = MagicMock()
    mock_loop.add_signal_handler = MagicMock()
    # Patch where get_event_loop is called (cli)
    with patch("aider_mcp_server.templates.initialization.cli.asyncio.get_event_loop", return_value=mock_loop):
        # Patch Path.resolve and Path.is_dir as cli uses them
        with (
            patch.object(Path, "resolve") as mock_resolve,
            patch.object(Path, "is_dir") as mock_is_dir,
        ):
            # Configure mocks for validation steps
            # Make resolve return a Path object that has the is_dir method mocked
            mock_resolved_path = MagicMock(spec=Path)
            mock_resolved_path.is_dir.return_value = True  # Assume dir exists for successful runs
            mock_resolved_path.__str__.return_value = abs_test_dir  # Return string path when needed
            mock_resolve.return_value = mock_resolved_path
            mock_is_dir.return_value = True  # Also mock the direct call if used

            # Mock is_git_repository (already passed in)
            mock_is_git_repo.return_value = (True, None)

            # Pass the mocked logger factory to main
            main(logger_factory=mock_get_logger)

    return abs_test_dir


# --- Test Cases ---


def test_stdio_mode_default(
    mock_serve: AsyncMock,
    mock_serve_sse: AsyncMock,
    mock_is_git_repository: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_get_logger: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: TempPathFactory,
) -> None:
    """Test that stdio mode is selected by default."""
    args = ["--current-working-dir", "temp_dir_placeholder"]
    abs_test_dir = run_main_with_args(
        monkeypatch,
        tmp_path_factory,
        args,
        mock_is_git_repository,
        mock_os_makedirs,
        mock_get_logger,
    )

    mock_serve.assert_called_once()
    call_args = mock_serve.call_args[1]
    assert call_args["current_working_dir"] == abs_test_dir  # __main__ passes resolved string path
    assert call_args["editor_model"] == DEFAULT_EDITOR_MODEL
    mock_serve_sse.assert_not_called()
    # Check is_git_repository was called with a Path object
    mock_is_git_repository.assert_called_once()
    assert isinstance(mock_is_git_repository.call_args[0][0], Path)
    assert str(mock_is_git_repository.call_args[0][0]) == abs_test_dir


def test_invalid_cwd_not_git_repo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: TempPathFactory,
    mock_is_git_repository: MagicMock,
    mock_os_makedirs: MagicMock,
    mock_get_logger: MagicMock,
) -> None:
    """Test that non-git repo CWD causes exit."""
    abs_test_dir = "/fake/path/to/not_a_repo"
    git_error_msg = "Not a git repo"

    # Mock Path.resolve to return a mock Path object
    mock_resolved_path = MagicMock(spec=Path)
    mock_resolved_path.is_dir.return_value = True
    mock_resolved_path.__str__.return_value = abs_test_dir
    with patch.object(Path, "resolve", return_value=mock_resolved_path):
        # Mock is_git_repository to return False
        with patch(
            "aider_mcp_server.templates.initialization.cli.is_git_repository",
            return_value=(False, git_error_msg),
        ) as mock_git_check:
            with patch("aider_mcp_server.templates.initialization.cli.get_logger") as mock_get_logger:
                mock_logger_instance = mock_get_logger.return_value
                mock_logger_instance.critical = MagicMock()

                # Set sys.argv directly
                with patch.object(
                    sys,
                    "argv",
                    [
                        "prog",
                        "--server-mode",
                        "stdio",
                        "--current-working-dir",
                        abs_test_dir,
                    ],
                ):
                    with pytest.raises(SystemExit) as excinfo:
                        main()

    assert excinfo.value.code == 1
    mock_logger_instance.critical.assert_called_once()
    assert "not a valid git repository" in mock_logger_instance.critical.call_args[0][0]
    assert git_error_msg in mock_logger_instance.critical.call_args[0][0]
    # Check is_git_repository was called with a Path object
    mock_git_check.assert_called_once()
    assert isinstance(mock_git_check.call_args[0][0], Path)
    assert str(mock_git_check.call_args[0][0]) == abs_test_dir
