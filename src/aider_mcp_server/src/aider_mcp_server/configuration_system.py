import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from atoms.logging import get_logger

logger = get_logger(__name__)


class ConfigurationSystem:
    _instance: Optional["ConfigurationSystem"] = None
    _initialized: bool = False

    def __new__(cls) -> "ConfigurationSystem":
        if cls._instance is None:
            cls._instance = super(ConfigurationSystem, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._config: Dict[str, Any] = {}
        self._default_config: Dict[str, Any] = {
            "logging": {
                "level": "INFO",
                "directory": "logs",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "transports": {
                "sse": {"enabled": True, "host": "localhost", "port": 8000},
                "stdio": {"enabled": True},
            },
            "application": {"name": "Aider MCP Server", "version": "0.1.0"},
        }
        # Initialize with defaults
        self._config = self._deep_copy_dict(self._default_config)
        self._initialized = True
        logger.debug("ConfigurationSystem initialized.")

    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy a dictionary."""
        # A simple deep copy for this context; json load/dump is a common way
        return json.loads(json.dumps(d))

    def load_from_env(self, prefix: str = "AIDER_MCP_") -> None:
        """Load configuration from environment variables."""
        logger.debug(f"Loading configuration from environment variables with prefix '{prefix}'.")
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_path_str = key[len(prefix) :].lower()
                config_path = config_path_str.split("_")
                logger.verbose(f"Found env var: {key}, path: {config_path}, value: {value}")
                self._set_config_value_from_env(config_path, value)
        logger.debug("Finished loading from environment variables.")

    def _set_config_value_from_env(self, path: List[str], value: str) -> None:
        """Set a configuration value from an environment variable string, performing type conversion."""
        current_level = self._config
        for i, key_part in enumerate(path):
            if i == len(path) - 1:
                # Try to convert string value to appropriate type
                if value.lower() == "true":
                    current_level[key_part] = True
                elif value.lower() == "false":
                    current_level[key_part] = False
                elif value.isdigit():
                    current_level[key_part] = int(value)
                elif value.replace(".", "", 1).isdigit() and value.count(".") <= 1:
                    try:
                        current_level[key_part] = float(value)
                    except ValueError: # pragma: no cover
                        # Should not happen with the isdigit/replace checks, but as a safeguard
                        current_level[key_part] = value
                else:
                    current_level[key_part] = value
                logger.verbose(f"Set config from env: {'.'.join(path)} = {current_level[key_part]}")
            else:
                if key_part not in current_level or not isinstance(current_level[key_part], dict):
                    current_level[key_part] = {}
                current_level = current_level[key_part]


    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """Load configuration from a file (JSON or YAML)."""
        p_file_path = Path(file_path)
        logger.debug(f"Loading configuration from file: {p_file_path}")
        if not p_file_path.exists():
            logger.error(f"Configuration file not found: {p_file_path}")
            raise FileNotFoundError(f"Configuration file not found: {p_file_path}")

        try:
            with open(p_file_path, "r") as f:
                if p_file_path.suffix.lower() in [".yaml", ".yml"]:
                    loaded_config = yaml.safe_load(f)
                elif p_file_path.suffix.lower() == ".json":
                    loaded_config = json.load(f)
                else:
                    logger.error(f"Unsupported configuration file format: {p_file_path.suffix}")
                    raise ValueError(
                        f"Unsupported configuration file format: {p_file_path.suffix}"
                    )
            if loaded_config:
                self._merge_config(loaded_config, self._config)
            logger.debug(f"Finished loading from file: {p_file_path}")
        except Exception as e:
            logger.exception(f"Error loading configuration file {p_file_path}: {e}")
            raise

    def _merge_config(self, source: Dict[str, Any], destination: Dict[str, Any]) -> None:
        """Merge a source configuration dictionary into a destination dictionary."""
        for key, value in source.items():
            if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                self._merge_config(value, destination[key])
            else:
                destination[key] = value
                logger.verbose(f"Merged config from file: {key} = {value}")


    def get(self, *path: str, default: Any = None) -> Any:
        """
        Get a configuration value at the specified path.
        Path elements are case-sensitive.
        """
        current_level = self._config
        for key_part in path:
            if isinstance(current_level, dict) and key_part in current_level:
                current_level = current_level[key_part]
            else:
                # Path not found, try default config
                current_level_default = self._default_config
                for default_key_part in path:
                    if isinstance(current_level_default, dict) and default_key_part in current_level_default:
                        current_level_default = current_level_default[default_key_part]
                    else:
                        logger.verbose(f"Path {'_'.join(path)} not found in config or defaults. Returning provided default.")
                        return default
                logger.verbose(f"Path {'_'.join(path)} not found in user config, using default value.")
                return current_level_default

        return current_level

    def _set_config_value(self, path: List[str], value: Any) -> None:
        """Set a configuration value at the specified path."""
        current_level = self._config
        for i, key_part in enumerate(path):
            if i == len(path) - 1:
                current_level[key_part] = value
                logger.debug(f"Set config programmatically: {'.'.join(path)} = {value}")
            else:
                if key_part not in current_level or not isinstance(current_level[key_part], dict):
                    current_level[key_part] = {}
                current_level = current_level[key_part]

    def set(self, value: Any, *path: str) -> None:
        """
        Set a configuration value at the specified path.
        Path elements are case-sensitive.
        """
        if not path:
            logger.warning("Cannot set configuration value with an empty path.")
            return
        self._set_config_value(list(path), value)

    def get_all(self) -> Dict[str, Any]:
        """Get the complete current configuration (a deep copy)."""
        return self._deep_copy_dict(self._config)

    def get_default_config(self) -> Dict[str, Any]:
        """Get the default configuration (a deep copy)."""
        return self._deep_copy_dict(self._default_config)

    def reset_to_defaults(self) -> None:
        """Reset the current configuration to the default configuration."""
        self._config = self._deep_copy_dict(self._default_config)
        logger.info("Configuration has been reset to defaults.")

    def reset_instance_state_for_testing(self) -> None: # pragma: no cover
        """
        Resets the singleton's state. Primarily for testing purposes.
        This allows tests to re-initialize the singleton.
        """
        ConfigurationSystem._instance = None
        ConfigurationSystem._initialized = False
        logger.debug("ConfigurationSystem instance state reset for testing.")
