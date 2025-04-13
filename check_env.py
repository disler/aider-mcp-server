#!/usr/bin/env python3
"""
Simple script to check if API keys are properly loaded from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def main():
    # Try to load from .env in the current directory
    env_path = Path('.env')
    if env_path.exists():
        print(f"Loading environment variables from {env_path}")
        load_dotenv(dotenv_path=env_path)
    else:
        print(f"Warning: .env file not found at {env_path}")
    
    # Check for API keys
    keys_to_check = {
        "OPENAI_API_KEY": "OpenAI",
        "GOOGLE_API_KEY": "Google/Gemini",
        "GEMINI_API_KEY": "Google/Gemini (alternative)",
        "ANTHROPIC_API_KEY": "Anthropic/Claude",
        "AZURE_OPENAI_API_KEY": "Azure OpenAI",
        "VERTEX_AI_API_KEY": "Vertex AI"
    }
    
    print("\nAPI Keys in environment:")
    for key, provider in keys_to_check.items():
        value = os.environ.get(key)
        if value:
            # Show first few and last few characters of the key
            masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
            print(f"✓ {provider} API key found: {masked_value}")
        else:
            print(f"✗ {provider} API key missing")
    
    print("\nAll environment variables:")
    for key, value in os.environ.items():
        if any(api_key in key.upper() for api_key in ["API_KEY", "TOKEN", "SECRET"]):
            # Mask sensitive values
            masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
            print(f"{key}={masked_value}")

if __name__ == "__main__":
    main()
