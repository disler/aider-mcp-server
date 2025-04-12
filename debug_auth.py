#!/usr/bin/env python3
"""
Debug script to test API authentication for different LLM providers.
Run this script to check if your API keys are working correctly.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(os.path.dirname(__file__)) / '.env'
if env_path.exists():
    print(f"Loading environment variables from {env_path}")
    load_dotenv(dotenv_path=env_path)
else:
    print(f"Warning: .env file not found at {env_path}")
    # Try loading from the current directory
    if Path('.env').exists():
        print("Loading environment variables from ./.env")
        load_dotenv()
    else:
        print("Warning: No .env file found in current directory either")

# Ensure the src directory is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from aider.models import Model
    from src.aider_mcp_server.atoms.logging import get_logger
except ImportError:
    print("Error: Required modules not found. Make sure you have installed the package.")
    print("Try running: pip install -e .")
    sys.exit(1)

# Configure logging
logger = get_logger("debug_auth")

def check_environment_variables():
    """Check if necessary API keys are set in the environment."""
    keys_to_check = {
        "OPENAI_API_KEY": "OpenAI",
        "GOOGLE_API_KEY": "Google/Gemini",
        "ANTHROPIC_API_KEY": "Anthropic/Claude",
        "AZURE_OPENAI_API_KEY": "Azure OpenAI"
    }
    
    print("Checking API keys in environment:")
    missing_keys = []
    for key, provider in keys_to_check.items():
        if os.environ.get(key):
            print(f"✓ {provider} API key found ({key})")
        else:
            print(f"✗ {provider} API key missing ({key})")
            missing_keys.append(key)
    
    return missing_keys

def test_model_connection(model_name):
    """Test connection to a specific model."""
    print(f"\nTesting connection to {model_name}...")
    try:
        model = Model(model_name)
        # Make a simple API call
        response = model.complete("Say 'Authentication successful!'")
        print(f"Response: {response}")
        print(f"✓ Successfully connected to {model_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to connect to {model_name}: {str(e)}")
        return False

def main():
    """Main function to run authentication tests."""
    print("=== LLM Authentication Debugging Tool ===")
    
    # Check environment variables
    missing_keys = check_environment_variables()
    
    # Test models based on available keys
    if "OPENAI_API_KEY" not in missing_keys:
        test_model_connection("openai/gpt-4")
    
    if "GOOGLE_API_KEY" not in missing_keys:
        test_model_connection("gemini/gemini-2.5-pro-exp-03-25")
    
    if "ANTHROPIC_API_KEY" not in missing_keys:
        test_model_connection("anthropic/claude-3-opus-20240229")
    
    print("\n=== Debugging Complete ===")
    print("If you're experiencing authentication issues:")
    print("1. Verify your API keys are correct")
    print("2. Check if your API keys have proper permissions")
    print("3. Ensure you have billing set up for the respective services")
    print("4. Check for any IP restrictions or VPN issues")
    print("5. Verify your system time is correct (important for API signatures)")
    print("6. Check if your API keys have the correct scope/permissions")
    print("7. Verify network connectivity to the API endpoints")

if __name__ == "__main__":
    main()
