#!/usr/bin/env python3
"""
Setup script for aider-mcp integration with Claude Code.
This script creates a .mcp.json configuration file in the target directory.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default models for different providers
DEFAULT_MODELS = {
    "gemini": "gemini/gemini-2.5-pro-exp-03-25",
    "openai": "openai/gpt-4o",
    "anthropic": "anthropic/claude-3-opus-20240229",
    "openrouter": "openrouter/openrouter/quasar-alpha",
    "fireworks": "fireworks_ai/accounts/fireworks/models/llama4-maverick-instruct-basic",
}

# API key environment variables for each provider
PROVIDER_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def is_git_repo(directory: str) -> bool:
    """Check if the specified directory is a git repository."""
    try:
        result = subprocess.run(
            ["git", "-C", directory, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def find_aider_mcp_server(specified_path: Optional[str] = None) -> Optional[str]:
    """Find the aider-mcp-server installation directory."""
    if specified_path:
        path = Path(specified_path).expanduser().resolve()
        if path.exists() and (path / "src" / "aider_mcp_server").exists():
            return str(path)
        print(f"Error: Could not find aider-mcp-server at {path}")
        return None

    # If no path specified, try these locations in order:

    # 1. Try current directory first
    current_dir = Path.cwd()
    if (current_dir / "src" / "aider_mcp_server").exists():
        return str(current_dir)

    # 2. Try parent directory (common when running from within the project)
    parent_dir = current_dir.parent
    if (parent_dir / "src" / "aider_mcp_server").exists():
        return str(parent_dir)

    # 3. Try to find the package in the Python path if installed
    try:
        import aider_mcp_server
        pkg_path = Path(aider_mcp_server.__file__).parent.parent.parent
        if (pkg_path / "src" / "aider_mcp_server").exists():
            return str(pkg_path)
    except ImportError:
        pass

    # 4. Try locating it via the script path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent  # src/aider_mcp_server/setup_aider_mcp.py -> root
    if (script_dir / "src" / "aider_mcp_server").exists():
        return str(script_dir)

    print("Error: Could not find aider-mcp-server installation.")
    print("Please specify the path using --aider-dir.")
    return None


def read_env_file(aider_dir: str) -> Dict[str, str]:
    """Read API keys from the .env file in the aider-mcp-server directory."""
    env_path = Path(aider_dir) / ".env"
    sample_env_path = Path(aider_dir) / ".env.sample"
    
    env_vars = {}
    
    # If .env doesn't exist but .env.sample does, copy it
    if not env_path.exists() and sample_env_path.exists():
        print(f"Warning: .env file not found at {env_path}")
        print(f"Creating from {sample_env_path}")
        with open(sample_env_path, "r") as f:
            sample_content = f.read()
        with open(env_path, "w") as f:
            f.write(sample_content)
    
    # Read the .env file if it exists
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    
    return env_vars


def get_available_models(env_vars: Dict[str, str]) -> Dict[str, List[str]]:
    """Get available models based on configured API keys."""
    available_models = {}
    
    # Check which providers have API keys
    for provider, env_var in PROVIDER_ENV_VARS.items():
        if env_var in env_vars and env_vars[env_var] and env_vars[env_var] != f"your_{provider}_api_key_here":
            if provider in DEFAULT_MODELS:
                available_models[provider] = [DEFAULT_MODELS[provider]]
            else:
                available_models[provider] = []
    
    return available_models


def select_model(available_models: Dict[str, List[str]]) -> Optional[str]:
    """Allow user to select a model from the available options."""
    if not available_models:
        print("Error: No API keys configured. Please edit the .env file and add your API keys.")
        return None
    
    print("\nAvailable models:")
    choices = []
    for i, (provider, models) in enumerate(available_models.items(), 1):
        for model in models:
            choices.append(model)
            print(f"{len(choices)}. {model}")
    
    if not choices:
        print("Error: No models available. Please configure API keys in the .env file.")
        return None
    
    while True:
        try:
            choice = input("\nSelect a model (number): ")
            idx = int(choice) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            print(f"Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nOperation cancelled")
            return None


def create_mcp_config(target_dir: str, aider_dir: str, model: str) -> bool:
    """Create the .mcp.json configuration file in the target directory."""
    config_path = Path(target_dir) / ".mcp.json"
    
    # Check if config already exists
    if config_path.exists():
        overwrite = input(f".mcp.json already exists at {config_path}. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Operation cancelled")
            return False
    
    # Create the configuration
    config = {
        "mcpServers": {
            "aider-mcp-server": {
                "type": "stdio",
                "command": "uv",
                "args": [
                    "--directory",
                    str(aider_dir),
                    "run",
                    "aider-mcp-server",
                    "--editor-model",
                    model,
                    "--current-working-dir",
                    str(target_dir)
                ]
            }
        }
    }
    
    # Write the configuration file
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created .mcp.json at {config_path}")
    return True


def generate_claude_command(aider_dir: str, model: str, target_dir: str) -> str:
    """Generate the claude mcp add command for reference."""
    return f"""claude mcp add aider-mcp-server -s local \\
  -- \\
  uv --directory "{aider_dir}" \\
  run aider-mcp-server \\
  --editor-model "{model}" \\
  --current-working-dir "{target_dir}"
"""


def install_global_symlink():
    """Install a symlink to this script in a directory in the user's PATH."""
    try:
        # Find the path to the current script
        script_path = Path(__file__).resolve()

        # Target installation directories in order of preference
        user_bin_dirs = [
            Path.home() / ".local" / "bin",
            Path.home() / "bin",
            Path("/usr/local/bin") if Path("/usr/local/bin").exists() and os.access("/usr/local/bin", os.W_OK) else None
        ]

        # Find the first valid directory
        target_dir = next((d for d in user_bin_dirs if d and d.exists()), None)

        if not target_dir:
            # Try to create ~/.local/bin if it doesn't exist
            target_dir = Path.home() / ".local" / "bin"
            target_dir.mkdir(parents=True, exist_ok=True)

        # Create the target symlink
        target_link = target_dir / "setup-aider-mcp"

        # Check if the link already exists
        if target_link.exists() or target_link.is_symlink():
            print(f"Link already exists at {target_link}")
            return True

        # Create the symlink
        os.symlink(script_path, target_link)
        os.chmod(target_link, 0o755)  # Make executable

        print(f"Installed symlink at {target_link}")

        # Check if the directory is in PATH
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        if str(target_dir) not in path_dirs:
            print(f"\nWARNING: {target_dir} is not in your PATH.")
            print(f"Add it to your PATH by adding this line to your shell profile:")
            print(f"export PATH=\"{target_dir}:$PATH\"")

        return True
    except Exception as e:
        print(f"Error installing symlink: {e}")
        return False


def main():
    """Main function to set up aider-mcp integration."""
    parser = argparse.ArgumentParser(description="Set up aider-mcp for Claude Code")
    parser.add_argument("--current-dir", help="Target project directory (defaults to current directory)")
    parser.add_argument("--aider-dir", help="Path to aider-mcp-server installation")
    parser.add_argument("--model", help="Model to use (will list available options if not specified)")
    parser.add_argument("--install-global", action="store_true", help="Install a global symlink to this script")

    args = parser.parse_args()

    # Handle global installation if requested
    if args.install_global:
        return 0 if install_global_symlink() else 1

    # Set the target directory
    target_dir = Path(args.current_dir).expanduser().resolve() if args.current_dir else Path.cwd()
    print(f"Target directory: {target_dir}")

    # Check if the target directory is a git repository
    if not is_git_repo(str(target_dir)):
        print(f"Error: {target_dir} is not a git repository. Aider requires a git repository to function.")
        return 1

    # Find the aider-mcp-server installation
    aider_dir = find_aider_mcp_server(args.aider_dir)
    if not aider_dir:
        return 1
    print(f"Found aider-mcp-server at: {aider_dir}")

    # Read API keys from .env
    env_vars = read_env_file(aider_dir)

    # Get available models
    available_models = get_available_models(env_vars)

    # Select model
    model = args.model
    if not model:
        model = select_model(available_models)
        if not model:
            return 1

    # Create MCP configuration
    success = create_mcp_config(str(target_dir), aider_dir, model)
    if not success:
        return 1

    # Show instructions
    print("\nSetup complete! You can now use Claude Code with aider-mcp-server.")
    print("\nTo add the server to Claude Code, run:")
    print(generate_claude_command(aider_dir, model, str(target_dir)))

    return 0


if __name__ == "__main__":
    sys.exit(main())