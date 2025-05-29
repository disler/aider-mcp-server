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
    "gemini-flash": "gemini/gemini-2.5-flash-preview-04-17",
    "openai": "openai/gpt-4o",
    "anthropic": "anthropic/claude-3-opus-20240229",
    "openrouter": "openrouter/openrouter/quasar-alpha",
    "fireworks": "fireworks_ai/accounts/fireworks/models/llama4-maverick-instruct-basic",
    "groq": "groq/llama-3.1-70b-versatile",
    "deepseek": "deepseek/deepseek-coder",
}

# Just-prompt default models
JUST_PROMPT_DEFAULT_MODELS = {
    "openai": "o:gpt-4o-mini",
    "anthropic": "a:claude-3-5-haiku",
    "gemini": "g:gemini-2.0-flash",
    "gemini-flash": "g:gemini-2.5-flash-preview-04-17",
    "groq": "q:llama-3.1-70b-versatile",
    "deepseek": "d:deepseek-coder",
    "ollama": "l:llama3.1",
}

# API key environment variables for each provider
PROVIDER_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "gemini-flash": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# Just-prompt API key environment variables
JUST_PROMPT_PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "gemini-flash": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def is_git_repo(directory: str) -> bool:
    """Check if the given directory is a git repository."""
    try:
        subprocess.run(["git", "rev-parse", "--git-dir"], 
                      cwd=directory, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def read_local_env_file() -> Dict[str, str]:
    """Read environment variables from .env file in the current directory."""
    local_env_path = Path.cwd() / ".env"
    env_vars = {}
    
    if local_env_path.exists():
        with open(local_env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        key, value = line.split("=", 1)
                        env_vars[key.strip()] = value.strip().strip('"').strip("'")
                    except ValueError:
                        # Skip lines that don't have a key=value format
                        pass
    
    return env_vars


def find_aider_mcp_server(specified_path: Optional[str] = None) -> Optional[str]:
    """Find the aider-mcp-server installation directory."""
    if specified_path:
        path = Path(specified_path).expanduser().resolve()
        if path.exists() and (path / "src" / "aider_mcp_server").exists():
            return str(path)
        print(f"Error: Could not find aider-mcp-server at {path}")
        return None
    
    # Check environment variable first
    env_path = os.environ.get("AIDER_MCP_SERVER_PATH")
    
    # If not found in environment, check local .env file
    if not env_path:
        local_env = read_local_env_file()
        env_path = local_env.get("AIDER_MCP_SERVER_PATH")
    
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists() and (path / "src" / "aider_mcp_server").exists():
            return str(path)
        print(f"Warning: AIDER_MCP_SERVER_PATH is set to {env_path}, but aider-mcp-server not found there")

    # Check relative to the script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    if (project_root / "src" / "aider_mcp_server").exists():
        return str(project_root)

    # If this is running from within the cloned git repo
    git_root = Path.cwd()
    for _ in range(3):  # Check up to 3 levels up
        if (git_root / "src" / "aider_mcp_server").exists():
            return str(git_root)
        git_root = git_root.parent

    print("Error: Could not find aider-mcp-server installation.")
    print("Please specify the path using --aider-dir or set AIDER_MCP_SERVER_PATH environment variable")
    return None


def read_env_file(directory: str) -> Dict[str, str]:
    """Read environment variables from .env file in the specified directory."""
    env_path = Path(directory) / ".env"
    env_vars = {}
    
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    
    # Merge with current environment, giving precedence to .env file
    full_env = os.environ.copy()
    full_env.update(env_vars)
    return full_env


def get_available_models(env_vars: Dict[str, str]) -> Dict[str, str]:
    """Get available models based on API keys present in the environment."""
    available = {}
    
    for provider, key_name in PROVIDER_ENV_VARS.items():
        if key_name in env_vars and env_vars[key_name]:
            if provider in DEFAULT_MODELS:
                available[provider] = DEFAULT_MODELS[provider]
            else:
                print(f"Warning: Provider '{provider}' has an API key but no default model configured. Skipping.")
                continue
    
    return available


def select_model(available_models: Dict[str, str]) -> Optional[str]:
    """Let the user select a model from available options."""
    if not available_models:
        print("No API keys found in environment.")
        print("Please set one of the following environment variables:")
        for provider, key_name in PROVIDER_ENV_VARS.items():
            print(f"  {key_name} - for {provider}")
        return None
    
    print("\nAvailable models:")
    models_list = list(available_models.items())
    for i, (provider, model) in enumerate(models_list, 1):
        print(f"{i}. {provider}: {model}")
    
    while True:
        try:
            choice = input("\nSelect a model (1-{}): ".format(len(models_list)))
            index = int(choice) - 1
            if 0 <= index < len(models_list):
                return models_list[index][1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None


def create_mcp_config(config_path: Path, aider_dir: str, model: str, target_dir: str, just_prompt_config: Optional[Dict] = None) -> bool:
    """Create the .mcp.json configuration file."""
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
    
    # Merge just-prompt configuration if provided
    if just_prompt_config:
        config["mcpServers"].update(just_prompt_config)
    
    # Write the configuration file
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created .mcp.json at {config_path}")
    return True


def generate_claude_command(aider_dir: str, model: str, target_dir: str, has_mcp_file: bool = False, just_prompt_config: Optional[Dict] = None) -> str:
    """Generate a bash script to properly add MCP servers to Claude Code."""
    config_path = Path(target_dir) / ".mcp.json"
    
    if has_mcp_file and config_path.exists():
        # Read the .mcp.json file to generate proper add-json commands
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        commands = []
        commands.append("#!/bin/bash")
        commands.append("# Script to add MCP servers to Claude Code")
        commands.append("# This script will automatically remove existing servers before adding new ones")
        commands.append("# from all possible scopes (project, local, user) before adding the new configuration")
        commands.append("")
        
        for server_name, server_config in config.get("mcpServers", {}).items():
            commands.append(f"# Configure {server_name}")
            commands.append(f'echo "Removing any existing {server_name} server..."')
            # Try removing from all scopes - ignore errors if not found
            commands.append(f'# Remove from project scope')
            commands.append(f'claude mcp remove -s project {server_name} 2>/dev/null || true')
            commands.append(f'# Remove from local scope')
            commands.append(f'claude mcp remove -s local {server_name} 2>/dev/null || true')
            commands.append(f'# Remove from user/global scope')
            commands.append(f'claude mcp remove -s user {server_name} 2>/dev/null || true')
            commands.append("")
            
            # Escape the JSON properly for bash
            server_json = json.dumps(server_config).replace("'", "'\"'\"'")
            
            commands.append(f'echo "Adding {server_name} server..."')
            commands.append(f"claude mcp add-json {server_name} -s local '{server_json}'")
            commands.append(f'if [ $? -eq 0 ]; then')
            commands.append(f'    echo "{server_name} server added successfully!"')
            commands.append(f'else')
            commands.append(f'    echo "Failed to add {server_name} server. Please check the error message above."')
            commands.append(f'    exit 1')
            commands.append(f'fi')
            commands.append("")
        
        commands.append('echo ""')
        commands.append('echo "All servers have been configured successfully!"')
        commands.append('echo "You can now use Claude Code with your MCP servers."')
        
        return '\n'.join(commands)
    else:
        return f"""claude mcp add aider-mcp-server -s local \\
  -- \\
  uv --directory "{aider_dir}" \\
  run aider-mcp-server \\
  --editor-model "{model}" \\
  --current-working-dir "{target_dir}"
"""


def find_just_prompt_server(specified_path: Optional[str] = None) -> Optional[str]:
    """Find the just-prompt installation directory."""
    if specified_path:
        path = Path(specified_path).expanduser().resolve()
        if path.exists() and (path / "src" / "just_prompt").exists():
            return str(path)
        print(f"Error: Could not find just-prompt at {path}")
        return None

    # Check environment variable first
    env_path = os.environ.get("JUST_PROMPT_PATH")
    
    # If not found in environment, check local .env file
    if not env_path:
        local_env = read_local_env_file()
        env_path = local_env.get("JUST_PROMPT_PATH")
    
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists() and (path / "src" / "just_prompt").exists():
            return str(path)
        print(f"Warning: JUST_PROMPT_PATH is set to {env_path}, but just-prompt not found there")

    # Look for just-prompt in common locations
    # Try in sibling directory to aider-mcp-server
    if find_aider_mcp_server():
        aider_dir = Path(find_aider_mcp_server())
        sibling_path = aider_dir.parent / "just-prompt"
        if sibling_path.exists() and (sibling_path / "src" / "just_prompt").exists():
            return str(sibling_path)

    print("Error: Could not find just-prompt installation.")
    print("Please specify the path using --just-prompt-dir or set JUST_PROMPT_PATH environment variable")
    return None


def read_just_prompt_env_file(directory: str) -> Dict[str, str]:
    """Read just-prompt's environment variables from .env file."""
    return read_env_file(directory)


def get_just_prompt_providers(env_vars: Dict[str, str]) -> List[str]:
    """Get available just-prompt providers based on API keys."""
    available = []
    
    for provider, key_name in JUST_PROMPT_PROVIDER_ENV_VARS.items():
        if key_name in env_vars and env_vars[key_name]:
            available.append(provider)
    
    # Also check for Ollama availability (doesn't require API key)
    if "OLLAMA_API_BASE" in env_vars or Path.home().joinpath(".ollama").exists():
        available.append("ollama")
    
    return available


def select_just_prompt_models(available_providers: List[str], preselected: Optional[str] = None) -> List[str]:
    """Select models for just-prompt from available providers."""
    if preselected:
        # Parse pre-selected models
        return [m.strip() for m in preselected.split(",")]
    
    if not available_providers:
        print("No just-prompt providers found with API keys.")
        return []
    
    print("\nAvailable just-prompt providers:")
    valid_providers = []
    for i, provider in enumerate(available_providers, 1):
        default_model = JUST_PROMPT_DEFAULT_MODELS.get(provider)
        if default_model is None:
            print(f"  {i}. {provider} - No default model configured, skipping.")
            continue
        valid_providers.append(provider)
        print(f"  {i}. {provider} (default: {default_model})")
    
    print("\nYou can:")
    print("1. Use all available providers (recommended)")
    print("2. Select specific providers")
    
    try:
        choice = input("\nChoice (1-2): ").strip()
        if choice == "1":
            return [JUST_PROMPT_DEFAULT_MODELS[p] for p in valid_providers]
        elif choice == "2":
            selected = []
            print("\nEnter provider numbers (comma-separated, e.g., 1,3,5):")
            choices = input().strip().split(",")
            for c in choices:
                try:
                    idx = int(c.strip()) - 1
                    if 0 <= idx < len(valid_providers):
                        provider = valid_providers[idx]
                        selected.append(JUST_PROMPT_DEFAULT_MODELS[provider])
                except ValueError:
                    continue
            return selected
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    
    return []


def create_just_prompt_config(just_prompt_dir: str, models: List[str]) -> Dict[str, Dict]:
    """Create just-prompt MCP configuration."""
    return {
        "just-prompt": {
            "type": "stdio",
            "command": "uv",
            "args": [
                "--directory",
                just_prompt_dir,
                "run",
                "just-prompt",
                "--default-models",
                ",".join(models)
            ],
            "env": {}
        }
    }


def install_global_symlink():
    """Install a global symlink to this script."""
    script_path = Path(__file__).resolve()
    
    # Check if the script is in the expected location
    if not str(script_path).endswith("src/aider_mcp_server/setup_aider_mcp.py"):
        print("Error: Script is not in the expected location. Cannot install global symlink.")
        return False
    
    try:
        # Find a suitable location for the symlink
        user_bin_dirs = [
            Path.home() / ".local" / "bin",
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


def update_model_config(target_dir: str, new_model: str) -> bool:
    """Update the model in the existing .mcp.json configuration."""
    config_path = Path(target_dir) / ".mcp.json"
    
    if not config_path.exists():
        print(f"Error: No .mcp.json found at {config_path}")
        print("Please run 'setup-aider-mcp setup' first to create the configuration.")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update the model in the aider configuration
        if "mcpServers" in config:
            # Look for either "aider" or "aider-mcp-server" in mcpServers
            server_name = None
            if "aider" in config["mcpServers"]:
                server_name = "aider"
            elif "aider-mcp-server" in config["mcpServers"]:
                server_name = "aider-mcp-server"
                
            if server_name:
                args = config["mcpServers"][server_name]["args"]
                # Find the --editor-model argument and update it
                for i, arg in enumerate(args):
                    if arg == "--editor-model" and i + 1 < len(args):
                        args[i + 1] = new_model
                        break
                
                # Write the updated configuration
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"Updated model to: {new_model}")
                print(f"Configuration saved to: {config_path}")
                
                # Show the command to apply the change
                print("\nTo apply this change, run the following commands:")
                print("\n" + generate_claude_command("", new_model, str(target_dir), True))
                
                return True
            else:
                print("Error: Couldn't find aider or aider-mcp-server in the configuration")
                return False
        else:
            print("Error: Invalid .mcp.json configuration format")
            return False
            
    except Exception as e:
        print(f"Error updating configuration: {e}")
        return False


def list_available_models(env_vars: Dict[str, str]) -> None:
    """List all available models based on API keys."""
    available = get_available_models(env_vars)
    
    if not available:
        print("No API keys found in environment.")
        print("\nTo use models, set one of the following environment variables:")
        for provider, key_name in PROVIDER_ENV_VARS.items():
            print(f"  {key_name} - for {provider}")
        print("\nDefault models for each provider:")
        for provider, model in DEFAULT_MODELS.items():
            print(f"  {provider}: {model}")
    else:
        print("Available models based on your API keys:")
        for provider, model in available.items():
            print(f"  {provider}: {model}")
        
        print("\nAdditional providers can be enabled by setting their API keys:")
        for provider, key_name in PROVIDER_ENV_VARS.items():
            if provider not in available:
                print(f"  {key_name} - for {provider}")


def setup_command(args):
    """Handle the setup subcommand."""
    # Set the target directory
    target_dir = Path(args.current_dir).expanduser().resolve() if args.current_dir else Path.cwd()
    print(f"Target directory: {target_dir}")

    # Check if the target directory is a git repository
    if not is_git_repo(str(target_dir)):
        print(f"""Error: {target_dir} is not a git repository.

Aider requires a git repository to function properly. To fix this:

1. Initialize a git repository in this directory:
   $ git init

2. Then run this command again:
   $ setup-aider-mcp setup --also-just-prompt

Alternatively, run this command from within an existing git repository.""")
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

    # Handle just-prompt setup if requested
    just_prompt_config = None
    if args.also_just_prompt:
        # Find just-prompt installation
        just_prompt_dir = find_just_prompt_server(args.just_prompt_dir)
        if just_prompt_dir:
            # Read just-prompt env file
            just_prompt_env = read_just_prompt_env_file(just_prompt_dir)
            
            # Get available providers
            providers = get_just_prompt_providers(just_prompt_env)
            
            # Select models
            models = select_just_prompt_models(providers, args.just_prompt_models)
            
            if models:
                just_prompt_config = create_just_prompt_config(just_prompt_dir, models)
                print(f"\nConfigured just-prompt with models: {', '.join(models)}")
            else:
                print("\nSkipping just-prompt configuration (no models selected)")

    # Create the .mcp.json configuration
    config_path = Path(target_dir) / ".mcp.json"
    if create_mcp_config(config_path, aider_dir, model, str(target_dir), just_prompt_config):
        print(f"\nUsing model: {model}")
        
        # Show the command to run
        print("\nTo install these MCP servers with Claude Code, save and run this bash script:")
        print("\n" + generate_claude_command(aider_dir, model, str(target_dir), True, just_prompt_config))
        print("\nAlternatively, save the above script as 'install_mcp.sh' and run 'bash install_mcp.sh'")
        return 0
    else:
        return 1


def change_model_command(args):
    """Handle the change-model subcommand."""
    # Set the target directory
    target_dir = Path(args.current_dir).expanduser().resolve() if args.current_dir else Path.cwd()
    
    # Find aider directory for env vars
    aider_dir = find_aider_mcp_server(args.aider_dir)
    if not aider_dir:
        return 1
    
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
    
    # Update the configuration
    if update_model_config(str(target_dir), model):
        return 0
    else:
        return 1


def list_models_command(args):
    """Handle the list-models subcommand."""
    # Find aider directory for env vars
    aider_dir = find_aider_mcp_server(args.aider_dir)
    if not aider_dir:
        return 1
    
    # Read API keys from .env
    env_vars = read_env_file(aider_dir)
    
    # List available models
    list_available_models(env_vars)
    return 0


def main():
    """Main function to set up aider-mcp integration."""
    parser = argparse.ArgumentParser(description="Set up aider-mcp for Claude Code")
    parser.add_argument("--install-global", action="store_true", help="Install a global symlink to this script")
    
    # Create subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command (default behavior)
    setup_parser = subparsers.add_parser('setup', help='Set up aider-mcp configuration')
    setup_parser.add_argument("--current-dir", help="Target project directory (defaults to current directory)")
    setup_parser.add_argument("--aider-dir", help="Path to aider-mcp-server installation (or set AIDER_MCP_SERVER_PATH env var)")
    setup_parser.add_argument("--model", help="Model to use (will list available options if not specified)")
    
    # Just-prompt integration options for setup
    setup_parser.add_argument("--also-just-prompt", action="store_true", help="Also set up the just-prompt MCP plugin")
    setup_parser.add_argument("--just-prompt-dir", help="Path to just-prompt installation (or set JUST_PROMPT_PATH env var)")
    setup_parser.add_argument("--just-prompt-models", help="Comma-separated list of models")
    setup_parser.set_defaults(func=setup_command)
    
    # Change model command
    change_model_parser = subparsers.add_parser('change-model', help='Change the model in existing configuration')
    change_model_parser.add_argument("--current-dir", help="Target project directory (defaults to current directory)")
    change_model_parser.add_argument("--aider-dir", help="Path to aider-mcp-server installation")
    change_model_parser.add_argument("--model", help="New model to use (will list available options if not specified)")
    change_model_parser.set_defaults(func=change_model_command)
    
    # List models command
    list_models_parser = subparsers.add_parser('list-models', help='List available models')
    list_models_parser.add_argument("--aider-dir", help="Path to aider-mcp-server installation")
    list_models_parser.set_defaults(func=list_models_command)
    
    args = parser.parse_args()
    
    # Handle global installation if requested
    if args.install_global:
        return 0 if install_global_symlink() else 1
    
    # If no command specified, default to setup
    if not args.command:
        # Default behavior - run setup
        args.command = 'setup'
        args.func = setup_command
        args.current_dir = getattr(args, 'current_dir', None)
        args.aider_dir = getattr(args, 'aider_dir', None)
        args.model = getattr(args, 'model', None)
        args.also_just_prompt = False
        args.just_prompt_dir = None
        args.just_prompt_models = None
    
    # Execute the command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())