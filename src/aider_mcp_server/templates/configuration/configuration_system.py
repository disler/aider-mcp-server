import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aider_mcp_server.atoms.logging.logger import Logger, get_logger


class ConfigurationSystem:
    """
    A comprehensive configuration system that loads configuration from multiple sources.

    Supports loading from:
    - Environment variables (with AIDER_MCP_ prefix)
    - JSON and YAML configuration files
    - Default configuration values

    Provides type conversion and hierarchical access to configuration values.
    """

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
                "verbose": False,
            },
            "transports": {
                "sse": {"enabled": True, "host": "localhost", "port": 8000, "heartbeat_interval": 30},
                "stdio": {"enabled": True},
                "http_streamable": {"enabled": True, "host": "localhost", "port": 8001},
            },
            "application": {
                "name": "Aider MCP Server",
                "version": "0.1.0",
                "max_concurrent_requests": 100,
                "request_timeout": 30.0,
            },
            "handlers": {"max_retries": 3, "retry_delay": 1.0},
        }

        # Initialize logger using the project's logging system
        self._logger: Logger = get_logger("configuration_system")
        self._initialized = True

        # Load configuration from environment by default
        self.load_from_env()

        self._logger.info("Configuration system initialized")

    def load_from_env(self, prefix: str = "AIDER_MCP_") -> None:
        """
        Load configuration from environment variables.

        Environment variables should be in the format:
        AIDER_MCP_SECTION_SUBSECTION_KEY=value

        Example: AIDER_MCP_LOGGING_LEVEL=DEBUG
        """
        loaded_count = 0

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert AIDER_MCP_LOGGING_LEVEL to ['logging', 'level']
                config_path = key[len(prefix) :].lower().split("_")
                self._set_config_value(config_path, value)
                loaded_count += 1
                self._logger.debug(f"Loaded environment variable: {key} -> {config_path}")

        if loaded_count > 0:
            self._logger.info(f"Loaded {loaded_count} configuration values from environment variables")

    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Load configuration from a file (JSON or YAML).

        Args:
            file_path: Path to the configuration file

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the file format is unsupported
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        self._logger.info(f"Loading configuration from file: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix.lower() in [".yaml", ".yml"]:
                    try:
                        import yaml

                        config = yaml.safe_load(f)
                    except ImportError as e:
                        raise ValueError("PyYAML is required to load YAML configuration files") from e
                elif file_path.suffix.lower() == ".json":
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")

            if config:
                self._merge_config(config)
                self._logger.info(f"Successfully loaded configuration from {file_path}")
            else:
                self._logger.warning(f"Configuration file {file_path} is empty or invalid")

        except Exception as e:
            self._logger.error(f"Error loading configuration from {file_path}: {e}")
            raise

    def _merge_config(self, config: Dict[str, Any]) -> None:
        """
        Merge a configuration dictionary with the current configuration.

        Args:
            config: Configuration dictionary to merge
        """

        def merge_dicts(source: Dict[str, Any], destination: Dict[str, Any]) -> None:
            for key, value in source.items():
                if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                    merge_dicts(value, destination[key])
                else:
                    destination[key] = value

        merge_dicts(config, self._config)

    def _merge_dicts(self, source: Dict[str, Any], destination: Dict[str, Any]) -> None:
        """
        Merge source dictionary into destination dictionary.

        Args:
            source: Source dictionary to merge from
            destination: Destination dictionary to merge into
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                self._merge_dicts(value, destination[key])
            else:
                destination[key] = value

    def _set_config_value(self, path: List[str], value: str) -> None:
        """
        Set a configuration value at the specified path.

        Args:
            path: List of keys representing the path to the configuration value
            value: String value to set (will be converted to appropriate type)
        """
        current = self._config
        for i, key in enumerate(path):
            if i == len(path) - 1:
                # Convert string value to appropriate type
                current[key] = self._convert_value(value)
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]

    def _convert_value(self, value: str) -> Any:
        """
        Convert a string value to the appropriate Python type.

        Args:
            value: String value to convert

        Returns:
            Converted value (bool, int, float, or str)
        """
        # Handle boolean values
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        elif value.lower() in ("false", "no", "0", "off"):
            return False

        # Handle numeric values
        if value.isdigit():
            return int(value)

        # Handle float values
        try:
            if "." in value and value.replace(".", "", 1).isdigit():
                return float(value)
        except ValueError:
            pass

        # Return as string if no conversion applies
        return value

    def get(self, *path: str, default: Any = None) -> Any:
        """
        Get a configuration value at the specified path.

        Args:
            *path: Path components to the configuration value
            default: Default value to return if the path doesn't exist

        Returns:
            Configuration value or default
        """
        # Handle empty path by returning empty config if no path given
        if not path:
            return default

        # First try user config
        value = self._get_value(self._config, path)
        if value is not None:
            return value

        # Then try default config
        value = self._get_value(self._default_config, path)
        if value is not None:
            return value

        # Finally return the provided default
        return default

    def _get_value(self, config: Dict[str, Any], path: tuple[str, ...]) -> Optional[Any]:
        """
        Get a value from a nested dictionary using a path.

        Args:
            config: Dictionary to search
            path: Tuple of keys representing the path

        Returns:
            Value at the path or None if not found
        """
        current = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def set(self, value: Any, *path: str) -> None:
        """
        Set a configuration value at the specified path.

        Args:
            value: Value to set
            *path: Path components to the configuration value
        """
        current = self._config
        for i, key in enumerate(path):
            if i == len(path) - 1:
                current[key] = value
                self._logger.debug(f"Set configuration value: {'.'.join(path)} = {value}")
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]

    def get_all(self) -> Dict[str, Any]:
        """
        Get the complete configuration (merged defaults and user config).

        Returns:
            Complete configuration dictionary
        """
        # Create a copy of the default config first
        import copy

        result = copy.deepcopy(self._default_config)
        # Then merge user config on top using a helper function
        self._merge_dicts(self._config, result)
        return result

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Section name (e.g., 'logging', 'transports')

        Returns:
            Configuration section dictionary
        """
        result = self.get(section, default={})
        return result if isinstance(result, dict) else {}

    def has(self, *path: str) -> bool:
        """
        Check if a configuration path exists.

        Args:
            *path: Path components to check

        Returns:
            True if the path exists, False otherwise
        """
        # Handle empty path
        if not path:
            return False

        return (
            self._get_value(self._config, path) is not None or self._get_value(self._default_config, path) is not None
        )

    def reload(self) -> None:
        """
        Reload configuration from environment variables.

        This clears user configuration and reloads from environment.
        """
        self._config.clear()
        self.load_from_env()
        self._logger.info("Configuration reloaded from environment")

    def validate(self) -> List[str]:
        """
        Validate the current configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        # Validate logging configuration
        log_level = self.get("logging", "level")
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level not in valid_levels:
            errors.append(f"Invalid logging level: {log_level}. Must be one of {valid_levels}")

        # Validate transport ports
        for transport in ["sse", "http_streamable"]:
            if self.get("transports", transport, "enabled"):
                port = self.get("transports", transport, "port")
                if not isinstance(port, int) or port < 1 or port > 65535:
                    errors.append(f"Invalid port for {transport}: {port}. Must be 1-65535")

        # Validate application settings
        timeout = self.get("application", "request_timeout")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            errors.append(f"Invalid request timeout: {timeout}. Must be a positive number")

        max_requests = self.get("application", "max_concurrent_requests")
        if not isinstance(max_requests, int) or max_requests < 1:
            errors.append(f"Invalid max concurrent requests: {max_requests}. Must be positive integer")

        if errors:
            self._logger.warning(f"Configuration validation found {len(errors)} errors")
            for error in errors:
                self._logger.warning(f"Validation error: {error}")
        else:
            self._logger.info("Configuration validation passed")

        return errors

    def get_transport_config(self, transport_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific transport.

        Args:
            transport_name: Name of the transport (e.g., 'sse', 'stdio')

        Returns:
            Transport configuration dictionary
        """
        result = self.get("transports", transport_name, default={})
        return result if isinstance(result, dict) else {}

    def is_transport_enabled(self, transport_name: str) -> bool:
        """
        Check if a specific transport is enabled.

        Args:
            transport_name: Name of the transport

        Returns:
            True if enabled, False otherwise
        """
        result = self.get("transports", transport_name, "enabled", default=False)
        return bool(result)


# Global configuration instance
def get_config() -> ConfigurationSystem:
    """
    Get the global configuration instance.

    Returns:
        ConfigurationSystem singleton instance
    """
    return ConfigurationSystem()
