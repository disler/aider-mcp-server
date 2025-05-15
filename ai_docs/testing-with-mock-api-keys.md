# Testing Aider Integrations with Mock API Keys

This document explains how to use the mock implementation for testing code that depends on the Aider library and external API keys.

## Problem

Testing code that depends on the Aider library can be challenging because it:

1. Requires valid API keys for services like OpenAI, Anthropic, or Gemini/Google
2. Makes real API calls which can be slow, expensive, and unreliable
3. May be rate-limited or fail if API quotas are exceeded
4. Makes tests dependent on external services

## Solution: Mock Implementation

To address these challenges, we've created a mock implementation that:

1. Provides fake API keys for testing
2. Mocks the Aider library components to avoid real API calls
3. Simulates the behavior of Aider without external dependencies
4. Implements predictable file modification based on prompts

## How to Use the Mock Implementation

### Option 1: Use the Dedicated Mock Test File

The simplest approach is to use the dedicated mock test file:

- `tests/atoms/tools/test_aider_ai_code_mock.py`

This file imports and uses the mock implementation automatically. All tests in this file use mocks and will work without real API keys.

### Option 2: Import the Mock Implementation in Your Tests

If you need to use mocks in your own test files:

```python
# Import the mock setup
from tests.atoms.tools.test_mock_api_keys import setup_mock_aider

# The setup_mock_aider fixture will automatically run before your tests
# It patches all the necessary components of the Aider library

def test_your_function():
    # Your test code here
    # No need to explicitly use the fixture
    ...
```

### Option 3: Use the Global API Key Fixture

For tests that don't need to mock the entire Aider library but still need mock API keys:

```python
import pytest

def test_your_function(mock_api_keys):
    # The mock_api_keys fixture will set up fake API keys
    # and restore the original ones after the test
    ...
```

## How the Mock Implementation Works

### Components

The mock implementation consists of:

1. `MockModel`: A mock version of `aider.models.Model`
2. `MockInputOutput`: A mock version of `aider.io.InputOutput`
3. `MockGitRepo`: A mock version of `aider.repo.GitRepo`
4. `MockCoder`: A mock version of `aider.coders.Coder`

### Functionality

The mock implementation simulates Aider's behavior:

- It provides mock API keys in the environment
- It creates predictable code implementations based on prompts
- It supports basic testing of the error handling and fallback mechanisms
- It tracks calls and arguments for verification in tests

### File Modification

The mock implementation will modify files based on prompt keywords:

- `add` → Creates an addition function
- `subtract` → Creates a subtraction function
- `multiply` → Creates a multiplication function
- `divide` → Creates a division function with zero-handling
- `calculator` → Creates a Calculator class with all required methods

## Extending the Mock Implementation

If you need to extend the mock implementation for new test cases:

1. Add new patterns in the `run_fn` method of `MockCoder` in `test_mock_api_keys.py`
2. Add more mock classes as needed for other Aider components
3. Update the `setup_mock_aider` fixture to patch additional imports

## Best Practices

- Use the mock implementation for unit and integration tests
- Use real API calls only for end-to-end tests
- Keep the mock implementation in sync with the real Aider API
- Use descriptive names for test functions to clarify they use mocks
- Add the `_mock` suffix to test functions that use the mock implementation

## Troubleshooting

If your tests fail with the mock implementation:

1. Check that all necessary Aider components are mocked
2. Verify that the mock implementation matches your expectations
3. Update the mock implementation if the Aider API has changed
4. Use print statements or logging to debug the behavior

## Example Test

```python
@pytest.mark.asyncio
async def test_addition_mock(temp_dir: str) -> None:
    """Test that code_with_aider can create a file that adds two numbers using mocks."""
    # Create the test file
    test_file = os.path.join(temp_dir, "math_add.py")
    with open(test_file, "w") as f:
        f.write("# This file should implement addition\n")

    prompt = "Implement a function add(a, b) that returns the sum of a and b in the math_add.py file."

    try:
        # Initialize diff_cache
        await init_diff_cache()

        # Run code_with_aider with mocked components
        result = await code_with_aider(
            ai_coding_prompt=prompt,
            relative_editable_files=[test_file],
            working_dir=temp_dir,
        )

        # Verify the results
        result_dict = json.loads(result)
        assert result_dict["success"] is True

        # Check the file contents
        with open(test_file, "r") as f:
            content = f.read()
        assert "def add(a, b):" in content
        assert "return a + b" in content

        # Try to use the function
        import sys
        sys.path.append(temp_dir)
        from math_add import add
        assert add(2, 3) == 5
    finally:
        # Clean up
        await shutdown_diff_cache()
```
