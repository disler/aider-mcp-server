#!/usr/bin/env python
"""
Test script for the rate limit fallback mechanism.
"""

import json
import sys
from typing import Any, Dict, List

# Using proper package import now that we've installed in development mode
from aider_mcp_server.atoms.utils.fallback_config import (
    detect_rate_limit_error,
    get_fallback_model,
    get_retry_delay,
    handle_rate_limit,
)


def test_detect_rate_limit_error() -> None:
    """Test the detect_rate_limit_error function."""
    print("Testing detect_rate_limit_error function...")

    # Print the DEFAULT_FALLBACK_CONFIG to see what's loaded
    from aider_mcp_server.atoms.utils.fallback_config import DEFAULT_FALLBACK_CONFIG

    print(f"DEFAULT_FALLBACK_CONFIG['gemini']: {DEFAULT_FALLBACK_CONFIG.get('gemini', {})}")

    # Test one provider at a time
    test_cases: List[Dict[str, Any]] = [
        {
            "provider": "openai",
            "test_errors": [
                {"error": "rate_limit_exceeded: error", "should_detect": True},
                {"error": "insufficient_quota: error", "should_detect": True},
                {"error": "other error", "should_detect": False},
            ],
        },
        {
            "provider": "anthropic",
            "test_errors": [
                {"error": "rate_limit_exceeded: error", "should_detect": True},
                {"error": "other error", "should_detect": False},
            ],
        },
        {
            "provider": "gemini",
            "test_errors": [
                {"error": "quota_exceeded: error", "should_detect": True},
                {"error": "rate_limit_exceeded: error", "should_detect": True},
                {"error": "other error", "should_detect": False},
            ],
        },
    ]

    for provider_case in test_cases:
        provider = provider_case["provider"]
        print(f"\nTesting provider: {provider}")

        for test_case in provider_case["test_errors"]:
            error_message = test_case["error"]
            should_detect = test_case["should_detect"]
            error = Exception(error_message)

            result = detect_rate_limit_error(error, provider)
            print(f"Error: '{error_message}', Should detect: {should_detect}, Result: {result}")

            if result != should_detect:
                print(f"❌ Test failed! Expected {should_detect}, got {result}")
                print(f"Provider: {provider}, Error: '{error_message}'")
                if provider in DEFAULT_FALLBACK_CONFIG:
                    config = DEFAULT_FALLBACK_CONFIG[provider]
                    if "rate_limit_errors" in config:
                        error_patterns = config["rate_limit_errors"]
                        if isinstance(error_patterns, (list, tuple)):
                            print(f"Error patterns: {error_patterns}")
                            for pattern in error_patterns:
                                if isinstance(pattern, str):
                                    print(
                                        f"Pattern '{pattern}' in '{error_message.lower()}': {pattern.lower() in error_message.lower()}"
                                    )
                raise AssertionError(f"Test failed for provider {provider}, error: {error_message}")

    print("✅ detect_rate_limit_error tests passed")


def test_get_fallback_model() -> None:
    """Test the get_fallback_model function."""
    # OpenAI fallback model
    assert get_fallback_model("gpt-4", "openai") == "gpt-3.5-turbo"
    assert get_fallback_model("gpt-3.5-turbo", "openai") == "gpt-3.5-turbo-16k"

    # Anthropic fallback model
    assert get_fallback_model("claude-3-opus-20240229", "anthropic") == "claude-3-sonnet-20240229"
    assert get_fallback_model("claude-3-sonnet-20240229", "anthropic") == "claude-3-haiku-20240307"

    # Gemini fallback model
    assert get_fallback_model("gemini-2.5-pro-preview-03-25", "gemini") == "gemini-1.5-pro"
    assert get_fallback_model("gemini-pro", "gemini") == "gemini-1.5-pro"

    # Test cycling back to the beginning after last model
    assert get_fallback_model("gpt-3.5-turbo-16k", "openai") == "gpt-4"

    print("✅ get_fallback_model tests passed")


def test_retry_logic() -> None:
    """Test the retry logic functions."""
    # Test get_retry_delay
    assert get_retry_delay(0, "openai") == 1
    assert get_retry_delay(1, "openai") == 2
    assert get_retry_delay(2, "openai") == 4

    # Test handle_rate_limit
    should_retry, delay = handle_rate_limit("openai", 0)
    assert should_retry is True
    assert delay == 1

    should_retry, delay = handle_rate_limit("openai", 10)  # Exceed max retries
    assert should_retry is False

    print("✅ retry logic tests passed")


def main() -> int:
    """Run all tests."""
    print("🧪 Testing rate limit fallback mechanism...")

    test_detect_rate_limit_error()
    test_get_fallback_model()
    test_retry_logic()

    print("✅ All tests passed!")

    # Load the .rate-limit-fallback.json file to verify its contents
    try:
        with open("../.rate-limit-fallback.json", "r") as f:
            config = json.load(f)
            print("\n📄 .rate-limit-fallback.json configuration:")
            print(json.dumps(config, indent=2))
    except Exception as e:
        print(f"\n❌ Error loading .rate-limit-fallback.json: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
