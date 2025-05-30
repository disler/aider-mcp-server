"""
Tests for WorkspaceManager class.

This module contains comprehensive unit tests for the WorkspaceManager
that handles workspace creation, git initialization, status reporting, and cleanup.
"""

import asyncio
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from aider_mcp_server.managers.workspace_manager import WorkspaceManager


# Helper to create mock asyncio.subprocess.Process objects
def create_mock_process(returncode=0, stdout="", stderr=""):
    """Creates a mock asyncio.subprocess.Process."""
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.returncode = returncode

    stdout_bytes = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
    stderr_bytes = stderr.encode("utf-8") if isinstance(stderr, str) else stderr

    # communicate() is an async method returning a future
    async def mock_communicate():
        return stdout_bytes, stderr_bytes

    process.communicate = MagicMock(side_effect=mock_communicate)
    # For cases where communicate might be awaited, ensure it's an AsyncMock or returns a coroutine
    # If MagicMock(side_effect=...) is used with an async def, it should work.
    # Alternatively, explicitly use AsyncMock for communicate:
    process.communicate = AsyncMock(return_value=(stdout_bytes, stderr_bytes))

    return process


@pytest_asyncio.fixture
async def manager(tmp_path: Path):
    """Create a fresh WorkspaceManager instance for each test, using a temporary base directory."""
    with patch("aider_mcp_server.managers.workspace_manager.WORKSPACE_BASE_DIR", str(tmp_path)):
        m = WorkspaceManager()
        yield m
    # Pytest's tmp_path fixture handles cleanup of the directory itself.


@pytest.fixture
def mock_async_subprocess_exec():
    """Fixture to mock asyncio.create_subprocess_exec."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        # Default to successful execution
        mock_exec.return_value = create_mock_process()
        yield mock_exec


class TestWorkspaceManager:
    """Test suite for WorkspaceManager class."""

    CLIENT_ID = "test_client_1"  # Renamed from WORKSPACE_ID

    @pytest.mark.asyncio
    async def test_create_client_workspace_success(self, manager: WorkspaceManager, tmp_path: Path):
        """Test successful client workspace directory creation."""
        workspace_path = tmp_path / self.CLIENT_ID

        created_path = await manager.create_client_workspace(self.CLIENT_ID)

        assert created_path == workspace_path
        assert workspace_path.exists()
        assert workspace_path.is_dir()
        # create_client_workspace only creates the directory, no git operations.

    @pytest.mark.asyncio
    async def test_initialize_git_repo_success(
        self, manager: WorkspaceManager, mock_async_subprocess_exec: MagicMock, tmp_path: Path
    ):
        """Test successful git initialization in an existing workspace."""
        workspace_path = tmp_path / self.CLIENT_ID
        workspace_path.mkdir(parents=True, exist_ok=True)  # Ensure dir exists

        mock_async_subprocess_exec.return_value = create_mock_process(stdout="Initialized empty Git repository")

        result = await manager.initialize_git_repo(workspace_path)
        assert result is True
        # Note: In real implementation, git init would create .git directory,
        # but in tests with mocked subprocess, we don't create actual .git dirs

        mock_async_subprocess_exec.assert_called_once_with(
            "git", "init", cwd=str(workspace_path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

    @pytest.mark.asyncio
    async def test_create_client_workspace_already_exists(self, manager: WorkspaceManager, tmp_path: Path):
        """Test creating workspace if directory already exists."""
        workspace_path = tmp_path / self.CLIENT_ID
        workspace_path.mkdir(parents=True, exist_ok=True)  # Pre-create directory

        # Should not raise error, should return the path
        created_path = await manager.create_client_workspace(self.CLIENT_ID)
        assert created_path == workspace_path
        assert workspace_path.exists()

    @pytest.mark.asyncio
    async def test_initialize_git_repo_already_exists(
        self, manager: WorkspaceManager, mock_async_subprocess_exec: MagicMock, tmp_path: Path
    ):
        """Test initializing git if .git directory already exists."""
        workspace_path = tmp_path / self.CLIENT_ID
        workspace_path.mkdir(parents=True, exist_ok=True)
        (workspace_path / ".git").mkdir()  # Simulate existing git repo

        result = await manager.initialize_git_repo(workspace_path)
        assert result is True  # Should report success
        mock_async_subprocess_exec.assert_not_called()  # Git init should not be called

    @pytest.mark.asyncio
    async def test_create_client_workspace_os_error_on_mkdir(self, manager: WorkspaceManager, tmp_path: Path):
        """Test OSError during workspace directory creation."""

        # os.makedirs is called twice: once for base dir, once for client dir
        # We'll make the second call (client dir) fail
        def makedirs_side_effect(path, exist_ok=True):
            if str(path).endswith(self.CLIENT_ID):
                raise OSError("Disk full")

        with patch("os.makedirs", side_effect=makedirs_side_effect) as mock_makedirs:
            with pytest.raises(RuntimeError, match=f"Failed to create workspace for client {self.CLIENT_ID}"):
                await manager.create_client_workspace(self.CLIENT_ID)
            # Should be called at least once for the client directory
            assert mock_makedirs.call_count >= 1

    @pytest.mark.asyncio
    async def test_initialize_git_repo_git_init_fails(
        self, manager: WorkspaceManager, mock_async_subprocess_exec: MagicMock, tmp_path: Path
    ):
        """Test failure during 'git init'."""
        workspace_path = tmp_path / self.CLIENT_ID
        workspace_path.mkdir(parents=True, exist_ok=True)

        mock_async_subprocess_exec.return_value = create_mock_process(returncode=1, stderr="git init failed")

        result = await manager.initialize_git_repo(workspace_path)
        assert result is False
        mock_async_subprocess_exec.assert_called_once()
        # Error is logged, method returns False, does not raise RuntimeError itself for this specific case

    @pytest.mark.asyncio
    async def test_initialize_git_repo_git_command_not_found(
        self, manager: WorkspaceManager, mock_async_subprocess_exec: MagicMock, tmp_path: Path
    ):
        """Test FileNotFoundError for git command."""
        workspace_path = tmp_path / self.CLIENT_ID
        workspace_path.mkdir(parents=True, exist_ok=True)
        mock_async_subprocess_exec.side_effect = FileNotFoundError("git not found")

        result = await manager.initialize_git_repo(workspace_path)
        assert result is False
        mock_async_subprocess_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_workspace_status_clean(
        self, manager: WorkspaceManager, mock_async_subprocess_exec: MagicMock, tmp_path: Path
    ):
        """Test git status for a clean repository."""
        workspace_path = await manager.create_client_workspace(self.CLIENT_ID)
        # Create .git directory manually to simulate git repo
        (workspace_path / ".git").mkdir()

        # Mock the git status call
        mock_async_subprocess_exec.return_value = create_mock_process(stdout="")  # Clean status

        status_dict = await manager.get_workspace_status(workspace_path)

        assert status_dict["path"] == str(workspace_path)
        assert status_dict["exists"] is True
        assert status_dict["is_directory"] is True
        assert status_dict["is_git_repo"] is True
        assert status_dict["git_status_output"] == ""
        assert status_dict["error"] is None

        mock_async_subprocess_exec.assert_called_once_with(
            "git",
            "status",
            "--porcelain",
            cwd=str(workspace_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    @pytest.mark.asyncio
    async def test_get_workspace_status_dirty(
        self, manager: WorkspaceManager, mock_async_subprocess_exec: MagicMock, tmp_path: Path
    ):
        """Test git status for a repository with changes."""
        workspace_path = await manager.create_client_workspace(self.CLIENT_ID)
        # Create .git directory manually to simulate git repo
        (workspace_path / ".git").mkdir()

        mock_async_subprocess_exec.return_value = create_mock_process(stdout=" M modified_file.txt\n?? new_file.txt")

        status_dict = await manager.get_workspace_status(workspace_path)
        assert status_dict["git_status_output"] == "M modified_file.txt\n?? new_file.txt"

    @pytest.mark.asyncio
    async def test_get_workspace_status_not_a_repo(self, manager: WorkspaceManager, tmp_path: Path):
        """Test git status for a directory that is not a git repository."""
        workspace_path = await manager.create_client_workspace(self.CLIENT_ID)
        # Do not initialize git repo

        status_dict = await manager.get_workspace_status(workspace_path)
        assert status_dict["path"] == str(workspace_path)
        assert status_dict["exists"] is True
        assert status_dict["is_directory"] is True
        assert status_dict["is_git_repo"] is False
        assert status_dict["git_status_output"] is None
        assert status_dict["error"] is None

    @pytest.mark.asyncio
    async def test_get_workspace_status_git_command_fails(
        self, manager: WorkspaceManager, mock_async_subprocess_exec: MagicMock, tmp_path: Path
    ):
        """Test git status when the git command itself fails."""
        workspace_path = await manager.create_client_workspace(self.CLIENT_ID)
        # Create .git directory manually to simulate git repo
        (workspace_path / ".git").mkdir()

        mock_async_subprocess_exec.return_value = create_mock_process(
            returncode=128, stderr="fatal: not a git repository"
        )

        status_dict = await manager.get_workspace_status(workspace_path)
        assert status_dict["is_git_repo"] is True  # .git dir exists
        assert status_dict["error"] == "Git status command failed: fatal: not a git repository"
        assert status_dict["git_status_output"] is None

    @pytest.mark.asyncio
    async def test_cleanup_workspace_success(self, manager: WorkspaceManager, tmp_path: Path):
        """Test successful workspace cleanup."""
        # First, create the workspace so it's tracked (if manager uses tracking)
        # The provided WorkspaceManager does not track, it derives path from client_id.
        # So, we just need to ensure the directory exists to test removal.
        workspace_path = await manager.create_client_workspace(self.CLIENT_ID)
        assert workspace_path.exists()

        # For cleanup_workspace, the WorkspaceManager uses _active_workspaces.
        # We need to ensure the client_id is in _active_workspaces for it to proceed.
        # This seems to be a mismatch with the provided WorkspaceManager source,
        # which does NOT use _active_workspaces.
        # Assuming the provided source is the target:

        await manager.cleanup_workspace(self.CLIENT_ID)
        assert not workspace_path.exists()

    @pytest.mark.asyncio
    async def test_cleanup_workspace_non_existent_dir(self, manager: WorkspaceManager, tmp_path: Path):
        """Test cleaning up when client is not tracked (should raise ValueError)."""
        client_id_non_existent_dir = "non_existent_dir_ws"

        # WorkspaceManager tracks clients in _active_workspaces
        # cleanup_workspace will raise ValueError if client_id not found
        with pytest.raises(ValueError, match=f"No active workspace found for client {client_id_non_existent_dir}"):
            await manager.cleanup_workspace(client_id_non_existent_dir)

    @pytest.mark.asyncio
    async def test_cleanup_workspace_os_error(self, manager: WorkspaceManager, tmp_path: Path):
        """Test OSError during workspace cleanup."""
        workspace_path = await manager.create_client_workspace(self.CLIENT_ID)

        # shutil.rmtree is called via asyncio.to_thread
        with patch("asyncio.to_thread", side_effect=OSError("Permission denied")) as mock_to_thread:
            with pytest.raises(RuntimeError, match=f"Failed to remove workspace directory for client {self.CLIENT_ID}"):
                await manager.cleanup_workspace(self.CLIENT_ID)
            # Check that asyncio.to_thread was called with shutil.rmtree and workspace_path
            mock_to_thread.assert_called_once_with(shutil.rmtree, workspace_path)

    @pytest.mark.asyncio
    async def test_concurrent_create_client_workspace(self, manager: WorkspaceManager, tmp_path: Path):
        """Test concurrent client workspace creation."""
        num_workspaces = 3
        client_ids = [f"concurrent_client_{i}" for i in range(num_workspaces)]

        tasks = [manager.create_client_workspace(cid) for cid in client_ids]
        results = await asyncio.gather(*tasks)

        assert len(results) == num_workspaces
        for i in range(num_workspaces):
            ws_path = tmp_path / client_ids[i]
            assert ws_path.exists()
            assert ws_path in results

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_workspace(self, manager: WorkspaceManager, tmp_path: Path):
        """Test concurrent workspace cleanup."""
        num_workspaces = 3
        client_ids = [f"concurrent_cleanup_client_{i}" for i in range(num_workspaces)]

        # Create workspaces first
        for cid in client_ids:
            await manager.create_client_workspace(cid)
            assert (tmp_path / cid).exists()

        tasks = [manager.cleanup_workspace(cid) for cid in client_ids]
        await asyncio.gather(*tasks)

        for cid in client_ids:
            assert not (tmp_path / cid).exists()

    @pytest.mark.asyncio
    async def test_get_workspace_status_various(
        self, manager: WorkspaceManager, mock_async_subprocess_exec: MagicMock, tmp_path: Path
    ):
        """Test workspace status reporting under various conditions."""
        client_id_git = "status_client_git"
        client_id_no_git = "status_client_no_git"
        client_id_not_exists = "status_client_not_exists"  # Path won't exist
        client_id_is_file = "status_client_is_file"

        # 1. Exists and is a Git repo
        path_git = await manager.create_client_workspace(client_id_git)
        # Create .git directory manually to simulate git repo
        (path_git / ".git").mkdir()
        mock_async_subprocess_exec.return_value = create_mock_process(stdout=" M file.txt")  # Git status output

        status_git = await manager.get_workspace_status(path_git)
        assert status_git == {
            "path": str(path_git),
            "exists": True,
            "is_directory": True,
            "is_git_repo": True,
            "git_status_output": "M file.txt",
            "error": None,
        }

        # 2. Exists but not a Git repo
        path_no_git = await manager.create_client_workspace(client_id_no_git)
        status_no_git = await manager.get_workspace_status(path_no_git)
        assert status_no_git == {
            "path": str(path_no_git),
            "exists": True,
            "is_directory": True,
            "is_git_repo": False,
            "git_status_output": None,
            "error": None,
        }

        # 3. Path does not exist
        path_not_exists = tmp_path / client_id_not_exists  # Construct path, but don't create
        status_not_exists = await manager.get_workspace_status(path_not_exists)
        assert status_not_exists == {
            "path": str(path_not_exists),
            "exists": False,
            "is_directory": False,  # Path.is_dir() is False if not exists
            "is_git_repo": False,
            "git_status_output": None,
            "error": "Workspace path does not exist.",
        }

        # 4. Path exists but is a file
        path_is_file = tmp_path / client_id_is_file
        path_is_file.touch()  # Create as a file
        status_is_file = await manager.get_workspace_status(path_is_file)
        assert status_is_file == {
            "path": str(path_is_file),
            "exists": True,
            "is_directory": False,
            "is_git_repo": False,
            "git_status_output": None,
            "error": "Workspace path is not a directory.",
        }

    @pytest.mark.asyncio
    async def test_create_client_workspace_permission_denied_mkdir(self, manager: WorkspaceManager, tmp_path: Path):
        """Test PermissionError during os.makedirs."""

        # Same fix as OSError test - makedirs is called twice
        def makedirs_side_effect(path, exist_ok=True):
            if str(path).endswith(self.CLIENT_ID):
                raise PermissionError("Permission denied for makedirs")

        with patch("os.makedirs", side_effect=makedirs_side_effect) as mock_makedirs:
            with pytest.raises(RuntimeError, match=f"Failed to create workspace for client {self.CLIENT_ID}"):
                await manager.create_client_workspace(self.CLIENT_ID)
            # Should be called at least once for the client directory
            assert mock_makedirs.call_count >= 1

    @pytest.mark.asyncio
    async def test_workspace_isolation(
        self, manager: WorkspaceManager, mock_async_subprocess_exec: MagicMock, tmp_path: Path
    ):
        """Test workspace isolation between different client IDs."""
        client_id1 = "isolated_client_1"
        client_id2 = "isolated_client_2"

        path1 = await manager.create_client_workspace(client_id1)
        path2 = await manager.create_client_workspace(client_id2)

        # Create .git directory manually for path1 to simulate git repo
        (path1 / ".git").mkdir()

        assert path1 == tmp_path / client_id1
        assert path2 == tmp_path / client_id2
        assert path1 != path2
        assert path1.exists() and (path1 / ".git").exists()
        assert path2.exists() and not (path2 / ".git").exists()  # path2 should not be a git repo

        await manager.cleanup_workspace(client_id1)
        assert not path1.exists()
        assert path2.exists()  # client_id2 workspace should remain untouched

    @pytest.mark.asyncio
    async def test_validate_workspace_success(self, manager: WorkspaceManager, tmp_path: Path):
        """Test validation of a healthy workspace directory."""
        workspace_path = await manager.create_client_workspace(self.CLIENT_ID)
        assert await manager.validate_workspace(workspace_path) is True

    @pytest.mark.asyncio
    async def test_validate_workspace_dir_missing(self, manager: WorkspaceManager, tmp_path: Path):
        """Test validation when workspace directory is missing."""
        non_existent_path = tmp_path / "non_existent_validation_ws"
        assert await manager.validate_workspace(non_existent_path) is False

    @pytest.mark.asyncio
    async def test_validate_workspace_not_a_directory(self, manager: WorkspaceManager, tmp_path: Path):
        """Test validation when workspace path is a file."""
        ws_path_file = tmp_path / "file_ws_validate"
        ws_path_file.touch()  # Create as file

        assert await manager.validate_workspace(ws_path_file) is False


# Removed tests:
# - test_get_workspace_path (method doesn't exist)
# - test_initialize_git_repo_git_config_fails (config not done by manager)
# - test_initialize_git_repo_git_commit_fails (commit not done by manager)
# - test_is_git_repo (protected method, not present)
# - test_validate_workspace_not_writable (validate_workspace doesn't check this)
# - test_validate_workspace_not_a_git_repo (validate_workspace doesn't check this)
# Original test_get_workspace_status was split/renamed to test_get_workspace_status_various
# Original test_create_workspace_success was split into test_create_client_workspace_success and test_initialize_git_repo_success
# Original test_create_workspace_already_exists_no_git -> test_create_client_workspace_already_exists (as create doesn't do git)
# Original test_create_workspace_already_exists_with_git -> test_initialize_git_repo_already_exists (for git part)
