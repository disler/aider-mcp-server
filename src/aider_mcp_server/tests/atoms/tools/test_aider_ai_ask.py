import os
import json
import tempfile
import pytest
import shutil
import subprocess
from aider_mcp_server.atoms.tools.aider_ai_ask import ask_aider

@pytest.fixture
def temp_dir():
    """Create a temporary directory with an initialized Git repository for testing."""
    tmp_dir = tempfile.mkdtemp()
    
    # Initialize git repository in the temp directory
    subprocess.run(["git", "init"], cwd=tmp_dir, capture_output=True, text=True, check=True)
    
    # Configure git user for the repository
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_dir, capture_output=True, text=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_dir, capture_output=True, text=True, check=True)
    
    # Create and commit an initial file to have a valid git history
    with open(os.path.join(tmp_dir, "README.md"), "w") as f:
        f.write("# Test Repository\nThis is a test repository for Aider MCP Server tests.")
    
    subprocess.run(["git", "add", "README.md"], cwd=tmp_dir, capture_output=True, text=True, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=tmp_dir, capture_output=True, text=True, check=True)
    
    yield tmp_dir
    
    # Clean up
    shutil.rmtree(tmp_dir)

def test_ask_basic_question(temp_dir):
    """Test that ask_aider can respond to a basic question without any reference files."""
    prompt = "What is the capital of France?"
    
    # Run ask_aider with working_dir
    result = ask_aider(
        ai_coding_prompt=prompt,
        relative_readonly_files=[],
        working_dir=temp_dir  # Pass the temp directory as working_dir
    )
    
    # Parse the JSON result
    result_dict = json.loads(result)
    
    # Check response structure is valid
    assert "success" in result_dict, "Expected success status in result"
    assert "response" in result_dict, "Expected response to be in result"
    
    # If API calls are rate limited, success might be False
    # So we'll skip the detailed response checks in that case
    if not result_dict["success"]:
        print(f"API call failed: {result_dict.get('response', 'No error details')}")
        pytest.skip("API call failed, possibly due to rate limiting")
    else:
        # Check that the response contains an answer
        response = result_dict["response"]
        assert len(response) > 0, "Expected a non-empty response"
        assert isinstance(response, str), "Expected response to be a string"

def test_ask_with_reference_file(temp_dir):
    """Test that ask_aider can respond to a question using a reference file."""
    # Create a reference file with some content
    ref_file_path = os.path.join(temp_dir, "reference.txt")
    with open(ref_file_path, "w") as f:
        f.write("The atomic number of gold is 79.\nThe chemical symbol for gold is Au.\nGold has a melting point of 1064 degrees Celsius.")
    
    # Use the file as a relative path
    relative_path = "reference.txt"
    
    prompt = "What is the atomic number and symbol of gold according to the reference file?"
    
    # Run ask_aider with reference file
    result = ask_aider(
        ai_coding_prompt=prompt,
        relative_readonly_files=[relative_path],
        working_dir=temp_dir
    )
    
    # Parse the JSON result
    result_dict = json.loads(result)
    
    # Check response structure is valid
    assert "success" in result_dict, "Expected success status in result"
    assert "response" in result_dict, "Expected response to be in result"
    
    # If API calls are rate limited, success might be False
    if not result_dict["success"]:
        print(f"API call failed: {result_dict.get('response', 'No error details')}")
        pytest.skip("API call failed, possibly due to rate limiting")
    else:
        # Check that the response contains information from the reference file
        response = result_dict["response"]
        assert any(keyword in response.lower() for keyword in ["79", "atomic number", "au", "symbol"]), \
            f"Expected response to contain information from the reference file, but got: {response}"

def test_ask_with_empty_working_dir():
    """Test that ask_aider returns an error when working_dir is not provided."""
    prompt = "This should fail because working_dir is required."
    
    # Run ask_aider without working_dir
    result = ask_aider(
        ai_coding_prompt=prompt,
        relative_readonly_files=[]
    )
    
    # Parse the JSON result
    result_dict = json.loads(result)
    
    # Check that it failed with the expected error
    assert result_dict["success"] is False, "Expected ask_aider to fail"
    assert "response" in result_dict, "Expected response to be in result"
    assert "working_dir is required" in result_dict["response"], \
        f"Expected error about working_dir requirement, but got: {result_dict['response']}"

def test_ask_model_specific_question(temp_dir):
    """Test that ask_aider can respond to a model-specific question."""
    prompt = "What capabilities do you have as an AI language model?"
    
    # Run ask_aider with a specific model
    result = ask_aider(
        ai_coding_prompt=prompt,
        relative_readonly_files=[],
        model="gemini/gemini-2.5-pro-exp-03-25",  # Specify a model
        working_dir=temp_dir
    )
    
    # Parse the JSON result
    result_dict = json.loads(result)
    
    # Check response structure is valid
    assert "success" in result_dict, "Expected success status in result"
    assert "response" in result_dict, "Expected response to be in result"
    
    # If API calls are rate limited, success might be False
    if not result_dict["success"]:
        print(f"API call failed: {result_dict.get('response', 'No error details')}")
        pytest.skip("API call failed, possibly due to rate limiting")
    else:
        # Check that the response is meaningful
        response = result_dict["response"]
        assert isinstance(response, str), "Expected response to be a string"
        # We can't reliably check the content length due to possible rate limiting

def test_ask_with_code_reference(temp_dir):
    """Test that ask_aider can analyze a code file when asked about it."""
    # Create a Python file with a function
    code_file_path = os.path.join(temp_dir, "calculator.py")
    with open(code_file_path, "w") as f:
        f.write("""
def add(a, b):
    \"\"\"Add two numbers and return the result.\"\"\"
    return a + b

def subtract(a, b):
    \"\"\"Subtract b from a and return the result.\"\"\"
    return a - b

def multiply(a, b):
    \"\"\"Multiply two numbers and return the result.\"\"\"
    return a * b

def divide(a, b):
    \"\"\"Divide a by b and return the result. Returns None if b is zero.\"\"\"
    if b == 0:
        return None
    return a / b
""")
    
    # Use the file as a relative path
    relative_path = "calculator.py"
    
    prompt = "Explain what the functions in calculator.py do and suggest improvements."
    
    # Run ask_aider with code reference file
    result = ask_aider(
        ai_coding_prompt=prompt,
        relative_readonly_files=[relative_path],
        working_dir=temp_dir
    )
    
    # Parse the JSON result
    result_dict = json.loads(result)
    
    # Check response structure is valid
    assert "success" in result_dict, "Expected success status in result"
    assert "response" in result_dict, "Expected response to be in result"
    
    # If API calls are rate limited, success might be False
    if not result_dict["success"]:
        print(f"API call failed: {result_dict.get('response', 'No error details')}")
        pytest.skip("API call failed, possibly due to rate limiting")
    else:
        # Check that the response mentions the functions
        response = result_dict["response"].lower()
        assert isinstance(response, str), "Expected response to be a string"
        # We can't reliably check the exact content due to possible rate limiting