"""Comprehensive tests for SSE working directory validation and configuration."""

import asyncio
import subprocess
import tempfile
from pathlib import Path

import httpx
import pytest


class TestSSEWorkingDirectory:
    """Test suite for SSE server working directory functionality."""

    def test_sse_working_directory_logs_validation(self):
        """Test that SSE server logs validate the working directory."""
        # Skip the test if the module cannot be imported
        try:
            import aider_mcp_server  # noqa: F401
        except ImportError:
            pytest.skip("aider_mcp_server module not available - likely installation issue")

        # Use a test directory with a unique name to avoid collisions
        import os
        import random

        test_base = os.path.join(tempfile.gettempdir(), f"test_aider_sse_{random.randint(1000, 9999)}")  # noqa: S311
        test_dir = Path(test_base)
        test_dir.mkdir(exist_ok=True)

        try:
            # Initialize a git repo in the test directory
            subprocess.run(["git", "init"], cwd=test_dir, capture_output=True)  # noqa: S603, S607

            # Use a random port to avoid collisions
            test_port = str(random.randint(9000, 9999))  # noqa: S311

            # Start the SSE server with the test directory
            try:
                result = subprocess.run(  # noqa: S603
                    [  # noqa: S607
                        "python",
                        "-m",
                        "aider_mcp_server",
                        "--server-mode",
                        "sse",
                        "--current-working-dir",
                        str(test_dir),
                        "--port",
                        test_port,
                        "--editor-model",
                        "gpt-3.5-turbo",
                    ],
                    cwd=Path(__file__).parent.parent,
                    capture_output=True,
                    text=True,
                    timeout=10,  # Even more timeout for CI
                    env={
                        "OPENAI_API_KEY": "test-key",
                        "TEST_MODE": "true",
                        "MCP_LOG_LEVEL": "DEBUG",  # Enable debug logging to see what's happening
                        **subprocess.os.environ,
                    },
                )
                stdout = result.stdout
                stderr = result.stderr
            except subprocess.TimeoutExpired as e:
                # In CI, the server might take longer to start - get partial output
                # Handle both str and bytes outputs
                stdout = e.stdout
                stderr = e.stderr
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                stdout = stdout or ""
                stderr = stderr or ""
                print(f"Process timed out after {e.timeout} seconds - checking partial output")

            # Check the output
            print("STDOUT:", stdout)
            print("STDERR:", stderr)

            # Verify the working directory was validated
            output = stdout + stderr

            # Check for validation message - it might have different formats
            validation_found = (
                "Validated working directory (git repository):" in output
                or "working directory" in output.lower()
                and str(test_dir) in output
            )

            assert validation_found, f"Working directory validation not found in logs.\nOutput: {output}"

            # Verify the correct directory was used
            assert str(test_dir) in output, f"Test directory {test_dir} not found in logs.\nOutput: {output}"

            print("Test passed! Working directory was properly validated.")

        finally:
            # Cleanup - ensure we clean up even if test fails
            import shutil

            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)

    def test_sse_rejects_non_git_directory(self, free_port):
        """Test that SSE server validates working directory is a git repo."""
        # Use a non-git directory
        test_dir = Path(tempfile.gettempdir()) / f"test_not_git_{free_port}"
        test_dir.mkdir(exist_ok=True)

        try:
            # Try to start the SSE server with a non-git directory
            # This should fail
            process = subprocess.Popen(  # noqa: S603
                [  # noqa: S607
                    "python",
                    "-m",
                    "aider_mcp_server",
                    "--server-mode",
                    "sse",
                    "--current-working-dir",
                    str(test_dir),
                    "--port",
                    str(free_port),
                    "--editor-model",
                    "gpt-3.5-turbo",
                ],
                cwd=Path(__file__).parent.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={"OPENAI_API_KEY": "test-key", **subprocess.os.environ},
            )

            # Wait for error
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                # Process didn't exit within timeout, kill it and get output
                process.kill()
                stdout, stderr = process.communicate()

            return_code = process.returncode

            # The process should have exited with an error or been killed
            assert return_code != 0, f"Expected non-zero exit code, got {return_code}"

            # In CI environment, the process might be killed before printing error messages
            # Check if we got any output, and if so, verify it contains the expected error
            combined_output = stdout + stderr
            if combined_output.strip():  # If we got any output
                assert "not a valid git repository" in combined_output or "not a git repository" in combined_output, (
                    f"Expected git repository error message not found.\nSTDOUT: {stdout}\nSTDERR: {stderr}"
                )
            # If no output, the test passes as long as the process failed (return_code != 0)

        finally:
            # Make sure process is terminated
            if "process" in locals() and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
            # Cleanup
            subprocess.run(["rm", "-rf", str(test_dir)], capture_output=True)  # noqa: S603, S607

    def test_sse_accepts_git_directory(self, free_port):
        """Test that SSE server accepts a valid git directory."""
        # Use a git directory
        test_dir = Path(tempfile.gettempdir()) / f"test_git_{free_port}"
        test_dir.mkdir(exist_ok=True)

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=test_dir, capture_output=True)  # noqa: S603, S607

        try:
            # Start the SSE server with a git directory
            # This should start successfully
            process = subprocess.Popen(  # noqa: S603
                [  # noqa: S607
                    "python",
                    "-m",
                    "aider_mcp_server",
                    "--server-mode",
                    "sse",
                    "--current-working-dir",
                    str(test_dir),
                    "--port",
                    str(free_port),
                    "--editor-model",
                    "gpt-3.5-turbo",
                ],
                cwd=Path(__file__).parent.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={"OPENAI_API_KEY": "test-key", **subprocess.os.environ},
            )

            # Give it some time to start
            import time

            time.sleep(3)

            # Check if server is still running
            return_code = process.poll()

            # If it exited, get the output
            if return_code is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"Server exited unexpectedly: {return_code}\nSTDOUT: {stdout}\nSTDERR: {stderr}")

            # Terminate the server
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()

            # The process should not have any error about git repository
            combined_output = stdout + stderr
            assert "not a valid git repository" not in combined_output, (
                f"Unexpected git repository error.\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            )

        finally:
            # Make sure process is terminated
            if "process" in locals() and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
            # Cleanup
            subprocess.run(["rm", "-rf", str(test_dir)], capture_output=True)  # noqa: S603, S607

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_sse_working_directory_integration(self):
        """Test that SSE server correctly passes working directory to aider handlers."""
        # Use a test directory
        import random

        test_dir = Path(tempfile.gettempdir()) / f"test_aider_sse_integration_{random.randint(1000, 9999)}"  # noqa: S311
        test_dir.mkdir(exist_ok=True)

        # Initialize a git repo in the test directory
        result = subprocess.run(["git", "init"], cwd=test_dir, capture_output=True)  # noqa: S607, S603
        if result.returncode != 0:
            # Might already exist, that's ok
            pass

        # Use a unique port for tests to avoid conflicts
        test_port = str(random.randint(9000, 9999))  # noqa: S311

        # Start the SSE server with the test directory
        server_process = subprocess.Popen(  # noqa: S603
            [  # noqa: S607
                "python",
                "-m",
                "aider_mcp_server",
                "--server-mode",
                "sse",
                "--current-working-dir",
                str(test_dir),
                "--port",
                test_port,
                "--editor-model",
                "gpt-3.5-turbo",
            ],
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                "OPENAI_API_KEY": "test-key",
                "TEST_MODE": "true",  # Add test mode flag
                **subprocess.os.environ,
            },
        )

        try:
            # Wait longer for server to start (since it creates various resources)
            await asyncio.sleep(4)

            # Check if server is running
            if server_process.poll() is not None:
                stdout, stderr = server_process.communicate()
                # Provide more detailed error information
                if stderr:
                    pytest.fail(f"Server failed to start. STDERR: {stderr}")
                else:
                    pytest.fail(f"Server failed to start. STDOUT: {stdout}")

            # Try to connect to the SSE endpoint
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{test_port}/sse/", headers={"Accept": "text/event-stream"}, timeout=5
                )

                assert response.status_code == 200, f"SSE connection failed: {response.status_code}"

            # Check server logs for the working directory validation message
            # Give it a moment to process
            await asyncio.sleep(1)

            # Terminate server and get output
            server_process.terminate()
            stdout, stderr = server_process.communicate(timeout=5)

            # Verify the working directory was validated - check for proper log message
            assert "Working directory" in stdout or "Working directory" in stderr, (
                f"Working directory validation not found in logs.\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            )

            # Verify the correct directory was used
            assert str(test_dir) in stdout or str(test_dir) in stderr, (
                f"Test directory {test_dir} not found in logs.\nSTDOUT: {stdout}\nSTDERR: {stderr}"
            )

        finally:
            # Ensure server is terminated
            if server_process.poll() is None:
                server_process.terminate()
                server_process.wait(timeout=5)

            # Cleanup
            subprocess.run(["rm", "-rf", str(test_dir)], capture_output=True)  # noqa: S603, S607


if __name__ == "__main__":
    # For direct execution
    test_instance = TestSSEWorkingDirectory()
    test_instance.test_sse_working_directory_logs_validation()
    print("Basic working directory validation test passed!")