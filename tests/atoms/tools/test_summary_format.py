import json
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aider_mcp_server.atoms.tools.aider_ai_code import code_with_aider
from aider_mcp_server.atoms.tools.changes_summarizer import summarize_changes


@pytest.mark.asyncio
@patch("aider_mcp_server.atoms.tools.aider_ai_code._run_aider_session")
@patch("aider_mcp_server.atoms.tools.aider_ai_code._setup_aider_coder")
@patch("aider_mcp_server.atoms.tools.aider_ai_code._configure_model")
async def test_format_includes_summary(mock_configure_model, mock_setup_coder, mock_run_aider):
    """Test that the response format includes changes summary."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup mocks
        mock_model = MagicMock()
        mock_configure_model.return_value = mock_model

        mock_coder = MagicMock()
        mock_setup_coder.return_value = mock_coder

        # Create a git diff sample
        git_diff = """diff --git a/test_file.py b/test_file.py
index 1234567..abcdef0 100644
--- a/test_file.py
+++ b/test_file.py
@@ -1,5 +1,6 @@
 def hello():
     print("Hello")
+    print("World")
 
 def goodbye():
     print("Goodbye")
"""
        # Create summary and status
        mock_summary = summarize_changes(git_diff)
        mock_status = {
            "has_changes": True, 
            "status_summary": "Changes detected: 1 files modified", 
            "files": [{"name": "test_file.py", "operation": "modified"}], 
            "files_modified": 1
        }
        
        # Setup mock result
        mock_run_aider.return_value = {
            "success": True,
            "diff": git_diff,
            "is_cached_diff": False,
            "changes_summary": mock_summary,
            "file_status": mock_status,
        }

        # Call the code_with_aider function
        result_json = await code_with_aider(
            ai_coding_prompt="Add a line to the hello function",
            relative_editable_files=["test_file.py"],
            working_dir=temp_dir,
            model="test-model",
        )
        
        # Parse the JSON result
        result = json.loads(result_json)
        
        # Print the result to console for verification
        print("\nRESPONSE FORMAT:", file=sys.stderr)
        print(json.dumps(result, indent=2), file=sys.stderr)
        
        # Verify the response includes essential information
        # Note: diff might be removed in some cases but changes_summary should always be present
        assert "changes_summary" in result
        assert "file_status" in result
        
        # Verify the summary content
        assert "summary" in result["changes_summary"]
        assert "files" in result["changes_summary"]
        assert "stats" in result["changes_summary"]
        
        # Verify useful information is included
        assert result["changes_summary"]["stats"]["lines_added"] > 0
        
        # Find the test_file.py in the files array
        found_file = False
        for file_entry in result["changes_summary"]["files"]:
            if file_entry["name"] == "test_file.py":
                found_file = True
                break
        assert found_file, "test_file.py should be in the files array"
        
        assert "print(\"World\")" in git_diff