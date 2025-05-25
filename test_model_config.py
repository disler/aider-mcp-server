#!/usr/bin/env python3
"""Test Model Configuration and Normalization"""

import sys
sys.path.insert(0, 'src')

from src.aider_mcp_server.molecules.tools.aider_ai_code import (
    _normalize_model_name,
    _determine_provider,
    _configure_model,
    check_api_keys
)

def test_model_normalization():
    print("=== TESTING MODEL NORMALIZATION ===")
    
    test_cases = [
        # Gemini models
        ("gemini-pro", "gemini/gemini-pro"),
        ("gemini-2.5-flash-preview-04-17", "gemini/gemini-2.5-flash-preview-04-17"),
        ("gemini/gemini-pro", "gemini/gemini-pro"),  # Already normalized
        
        # OpenAI models
        ("gpt-4", "openai/gpt-4"),
        ("gpt-4o", "openai/gpt-4o"),
        ("openai/gpt-4", "openai/gpt-4"),  # Already normalized
        
        # Anthropic models
        ("claude-3-5-sonnet-20241022", "anthropic/claude-3-5-sonnet-20241022"),
        ("anthropic/claude-3-5-sonnet-20241022", "anthropic/claude-3-5-sonnet-20241022"),  # Already normalized
        
        # Unknown models (default to gemini)
        ("unknown-model", "gemini/unknown-model"),
    ]
    
    for input_model, expected in test_cases:
        result = _normalize_model_name(input_model)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {input_model} -> {result} (expected: {expected})")

def test_provider_determination():
    print("\n=== TESTING PROVIDER DETERMINATION ===")
    
    test_cases = [
        ("gemini/gemini-pro", "gemini"),
        ("openai/gpt-4", "openai"),
        ("anthropic/claude-3-5-sonnet-20241022", "anthropic"),
        ("gemini-pro", "gemini"),  # Unnormalized
        ("gpt-4", "openai"),       # Unnormalized
        ("claude-3-5-sonnet-20241022", "anthropic"),  # Unnormalized
    ]
    
    for input_model, expected in test_cases:
        result = _determine_provider(input_model)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {input_model} -> {result} (expected: {expected})")

def test_model_configuration():
    print("\n=== TESTING MODEL CONFIGURATION ===")
    
    # Test basic model configuration
    try:
        model = _configure_model("gemini/gemini-2.5-flash-preview-04-17")
        print(f"  ✅ Basic model configuration successful")
    except Exception as e:
        print(f"  ❌ Basic model configuration failed: {e}")
    
    # Test architect mode configuration
    try:
        model = _configure_model(
            "gemini/gemini-2.5-pro-preview-05-06",
            editor_model="gemini/gemini-2.5-flash-preview-04-17",
            architect_mode=True
        )
        print(f"  ✅ Architect mode configuration successful")
    except Exception as e:
        print(f"  ❌ Architect mode configuration failed: {e}")
    
    # Test same model for architect and editor
    try:
        model = _configure_model(
            "gemini/gemini-2.5-pro-preview-05-06",
            architect_mode=True
        )
        print(f"  ✅ Same model architect/editor configuration successful")
    except Exception as e:
        print(f"  ❌ Same model architect/editor configuration failed: {e}")

def test_api_key_management():
    print("\n=== TESTING API KEY MANAGEMENT ===")
    
    # Test API key checking
    try:
        key_status = check_api_keys()
        print(f"  ✅ API key check successful")
        print(f"    Any keys found: {key_status.get('any_keys_found', False)}")
        print(f"    Available providers: {key_status.get('available_providers', [])}")
        print(f"    Missing providers: {key_status.get('missing_providers', [])}")
    except Exception as e:
        print(f"  ❌ API key check failed: {e}")

if __name__ == "__main__":
    test_model_normalization()
    test_provider_determination()
    test_model_configuration()
    test_api_key_management()