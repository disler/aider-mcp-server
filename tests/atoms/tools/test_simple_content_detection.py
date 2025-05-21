"""
Test for verifying the simple content detection works properly.
"""

import os
import tempfile

from aider_mcp_server.atoms.tools.aider_ai_code import _check_for_meaningful_changes


def test_check_for_simple_content_detection():
    """Test that simple content is correctly detected as meaningful."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple file with minimal content
        simple_file = os.path.join(temp_dir, "simple.py")
        with open(simple_file, "w") as f:
            f.write('print("Hello, World!")')

        # Test with our modified function
        result = _check_for_meaningful_changes(["simple.py"], temp_dir)

        # Assert that the simple content is detected as meaningful
        assert result is True, "Simple program content should be considered meaningful"

        # For comparison, create an empty file
        empty_file = os.path.join(temp_dir, "empty.py")
        with open(empty_file, "w") as f:
            f.write("")

        # Test with just the empty file
        result = _check_for_meaningful_changes(["empty.py"], temp_dir)

        # Assert that empty file is not meaningful
        assert result is False, "Empty file should not be considered meaningful"
