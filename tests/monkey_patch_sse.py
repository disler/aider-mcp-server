# Import the module containing the function to be patched
import aider_mcp_server.templates.servers.server


# Define the patched version of is_git_repository
def patched_is_git_repository(directory):
    """
    Patched version that always returns True for testing.
    This allows tests to run without needing a real git repository.
    """
    return True, ""


# Apply the patch: Override the original function with our patched version
aider_mcp_server.server.is_git_repository = patched_is_git_repository
