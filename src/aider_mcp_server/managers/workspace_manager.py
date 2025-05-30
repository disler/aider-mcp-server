"""
Workspace Manager for multi-client orchestration.

This module provides the WorkspaceManager class that handles creation,
validation, and cleanup of client-specific workspaces, including Git operations.
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Dict

# Assuming this file will be placed in a location like src/aider_mcp_server/managers/
# for these imports to work correctly.
from aider_mcp_server.atoms.logging.logger import get_logger
from aider_mcp_server.atoms.utils.config_constants import WORKSPACE_BASE_DIR


class WorkspaceManager:
    """
    Manages client-specific workspaces, including their creation,
    validation, git initialization, and cleanup.
    """

    def __init__(self) -> None:
        """Initialize the WorkspaceManager."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._lock = asyncio.Lock()
        self._active_workspaces: Dict[str, Path] = {}  # client_id -> workspace_path
        self.logger.info("WorkspaceManager initialized")

    async def create_client_workspace(self, client_id: str) -> Path:
        """
        Create a dedicated workspace for a client.

        Args:
            client_id: The unique identifier for the client.

        Returns:
            Path: The path to the created client's workspace.

        Raises:
            ValueError: If the client_id is already actively managed.
            RuntimeError: If workspace creation fails at the filesystem level.
        """
        async with self._lock:
            if client_id in self._active_workspaces:
                self.logger.warning(
                    f"Client {client_id} already has a managed workspace: {self._active_workspaces[client_id]}"
                )
                raise ValueError(f"Client {client_id} already has an active workspace entry.")

            base_workspace_dir = Path(WORKSPACE_BASE_DIR).expanduser().resolve()
            workspace_path = base_workspace_dir / client_id

            try:
                # Ensure the base directory for all workspaces exists
                os.makedirs(base_workspace_dir, exist_ok=True)
                # Create the client-specific workspace
                os.makedirs(workspace_path, exist_ok=True)
                self.logger.info(f"Ensured workspace directory exists: {workspace_path}")
            except OSError as e:
                self.logger.error(f"Failed to create workspace directory {workspace_path}: {e}")
                raise RuntimeError(f"Failed to create workspace for client {client_id}") from e

            self._active_workspaces[client_id] = workspace_path
            self.logger.info(f"Registered workspace for client {client_id} at {workspace_path}")
            return workspace_path

    async def validate_workspace(self, workspace_path: Path) -> bool:
        """
        Validate if the given path is a valid workspace directory.

        Args:
            workspace_path: The path to validate.

        Returns:
            True if the path is an existing directory, False otherwise.
        """
        is_valid = workspace_path.exists() and workspace_path.is_dir()
        if is_valid:
            self.logger.debug(f"Workspace path {workspace_path} is valid.")
        else:
            self.logger.warning(
                f"Workspace path {workspace_path} is not valid (exists: {workspace_path.exists()}, is_dir: {workspace_path.is_dir()})."
            )
        return is_valid

    async def cleanup_workspace(self, client_id: str) -> None:
        """
        Remove the workspace directory for a given client and untrack it.

        Args:
            client_id: The unique identifier for the client whose workspace is to be cleaned.

        Raises:
            ValueError: If no active workspace is found for the client.
            RuntimeError: If workspace cleanup (directory removal) fails.
        """
        async with self._lock:
            if client_id not in self._active_workspaces:
                self.logger.warning(f"No active workspace found for client {client_id} during cleanup attempt.")
                raise ValueError(f"No active workspace found for client {client_id} to clean up.")

            workspace_path = self._active_workspaces.pop(client_id)
            self.logger.info(f"Attempting to clean up workspace {workspace_path} for client {client_id}.")

            try:
                if workspace_path.exists() and workspace_path.is_dir():
                    await asyncio.to_thread(shutil.rmtree, workspace_path)
                    self.logger.info(f"Successfully removed workspace directory {workspace_path}.")
                elif not workspace_path.exists():
                    self.logger.info(f"Workspace directory {workspace_path} does not exist. No removal needed.")
                else:
                    self.logger.warning(
                        f"Workspace path {workspace_path} is not a directory. Cannot remove."
                    )
            except OSError as e:
                self.logger.error(f"Failed to remove workspace directory {workspace_path}: {e}")
                # Re-add to _active_workspaces if removal failed and we want to retry?
                # For now, it remains removed from tracking, and we raise an error.
                raise RuntimeError(f"Failed to remove workspace directory for client {client_id}") from e
            
            self.logger.info(f"Cleaned up and untracked workspace for client {client_id}.")

    async def initialize_git_repo(self, workspace_path: Path) -> bool:
        """
        Initialize a Git repository in the specified workspace path.

        Args:
            workspace_path: The path to the workspace.

        Returns:
            True if initialization was successful or already a repo, False otherwise.
        """
        if not await self.validate_workspace(workspace_path):
            self.logger.error(f"Cannot initialize Git repo: Workspace {workspace_path} is not valid.")
            return False

        git_dir = workspace_path / ".git"
        if git_dir.exists() and git_dir.is_dir():
            self.logger.info(f"Git repository already exists in {workspace_path}.")
            return True

        try:
            self.logger.info(f"Initializing Git repository in {workspace_path}...")
            process = await asyncio.create_subprocess_exec(
                'git', 'init',
                cwd=str(workspace_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                decoded_stdout = stdout.decode().strip() if stdout else ""
                self.logger.info(f"Successfully initialized Git repository in {workspace_path}. Output: {decoded_stdout}")
                return True
            else:
                decoded_stderr = stderr.decode().strip() if stderr else ""
                self.logger.error(f"Failed to initialize Git repository in {workspace_path}. Error: {decoded_stderr}")
                return False
        except FileNotFoundError:
            self.logger.error(f"Git command not found. Cannot initialize repository in {workspace_path}.")
            return False
        except Exception as e:
            self.logger.error(f"Exception during Git initialization in {workspace_path}: {e}")
            return False

    async def get_workspace_status(self, workspace_path: Path) -> Dict[str, Any]:
        """
        Get the status of the workspace, including Git status if applicable.

        Args:
            workspace_path: The path to the workspace.

        Returns:
            A dictionary containing workspace status information.
        """
        status: Dict[str, Any] = {
            "path": str(workspace_path),
            "exists": False,
            "is_directory": False,
            "is_git_repo": False,
            "git_status_output": None,
            "error": None,
        }

        if not workspace_path.exists():
            status["error"] = "Workspace path does not exist."
            self.logger.warning(f"Workspace path {workspace_path} does not exist for status check.")
            return status
        
        status["exists"] = True

        if not workspace_path.is_dir():
            status["error"] = "Workspace path is not a directory."
            self.logger.warning(f"Workspace path {workspace_path} is not a directory for status check.")
            return status
        
        status["is_directory"] = True

        git_dir = workspace_path / ".git"
        if git_dir.exists() and git_dir.is_dir():
            status["is_git_repo"] = True
            try:
                self.logger.debug(f"Fetching Git status for {workspace_path}...")
                process = await asyncio.create_subprocess_exec(
                    'git', 'status', '--porcelain',
                    cwd=str(workspace_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()

                if process.returncode == 0:
                    git_output = stdout.decode().strip() if stdout else ""
                    status["git_status_output"] = git_output
                    self.logger.debug(f"Git status for {workspace_path}: {git_output if git_output else 'clean'}")
                else:
                    error_msg = stderr.decode().strip() if stderr else "Unknown Git error"
                    status["error"] = f"Git status command failed: {error_msg}"
                    self.logger.error(f"Failed to get Git status for {workspace_path}. Error: {error_msg}")
            except FileNotFoundError:
                error_msg = f"Git command not found. Cannot get Git status for workspace: {workspace_path}"
                status["error"] = error_msg
                self.logger.error(error_msg)
            except Exception as e:
                error_msg = f"Exception during Git status check for {workspace_path}: {e}"
                status["error"] = error_msg
                self.logger.error(error_msg)
        else:
            self.logger.info(f"Workspace {workspace_path} is not a Git repository.")

        return status
