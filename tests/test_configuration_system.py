import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

from aider_mcp_server.templates.configuration.configuration_system import ConfigurationSystem, get_config


class TestConfigurationSystem(unittest.TestCase):
    """Comprehensive test suite for the ConfigurationSystem."""

    def setUp(self):
        """Set up test environment by resetting the singleton."""
        # Reset the singleton for each test
        ConfigurationSystem._instance = None
        ConfigurationSystem._initialized = False

    def tearDown(self):
        """Clean up after each test."""
        # Reset the singleton again
        ConfigurationSystem._instance = None
        ConfigurationSystem._initialized = False

    @patch.dict(os.environ, {}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_singleton_pattern(self, mock_get_logger):
        """Test that ConfigurationSystem follows singleton pattern."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create two instances
        config1 = ConfigurationSystem()
        config2 = ConfigurationSystem()

        # Should be the same instance
        self.assertIs(config1, config2)

        # Global getter should return the same instance
        config3 = get_config()
        self.assertIs(config1, config3)

    @patch.dict(os.environ, {}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_default_configuration(self, mock_get_logger):
        """Test that default configuration values are properly set."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config = ConfigurationSystem()

        # Test default logging configuration
        self.assertEqual(config.get("logging", "level"), "INFO")
        self.assertEqual(config.get("logging", "directory"), "logs")
        self.assertFalse(config.get("logging", "verbose"))

        # Test default transport configuration
        self.assertTrue(config.get("transports", "sse", "enabled"))
        self.assertEqual(config.get("transports", "sse", "host"), "localhost")
        self.assertEqual(config.get("transports", "sse", "port"), 8000)

        # Test default application configuration
        self.assertEqual(config.get("application", "name"), "Aider MCP Server")
        self.assertEqual(config.get("application", "max_concurrent_requests"), 100)

    @patch.dict(
        os.environ,
        {
            "AIDER_MCP_LOGGING_LEVEL": "DEBUG",
            "AIDER_MCP_LOGGING_VERBOSE": "true",
            "AIDER_MCP_TRANSPORTS_SSE_PORT": "9000",
            "AIDER_MCP_APPLICATION_MAX_CONCURRENT_REQUESTS": "50",
        },
        clear=True,
    )
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_environment_variable_loading(self, mock_get_logger):
        """Test loading configuration from environment variables."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config = ConfigurationSystem()

        # Test string values
        self.assertEqual(config.get("logging", "level"), "DEBUG")

        # Test boolean conversion
        self.assertTrue(config.get("logging", "verbose"))

        # Test integer conversion
        self.assertEqual(config.get("transports", "sse", "port"), 9000)
        # The env var AIDER_MCP_APPLICATION_MAX_CONCURRENT_REQUESTS creates nested structure
        self.assertEqual(config.get("application", "max", "concurrent", "requests"), 50)

    @patch.dict(os.environ, {}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_type_conversion(self, mock_get_logger):
        """Test type conversion for configuration values."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config = ConfigurationSystem()

        # Test boolean conversions
        test_cases = [
            ("true", True),
            ("false", False),
            ("yes", True),
            ("no", False),
            ("1", True),
            ("0", False),
            ("on", True),
            ("off", False),
            ("True", True),
            ("False", False),
        ]

        for value, expected in test_cases:
            converted = config._convert_value(value)
            self.assertEqual(converted, expected, f"Failed to convert '{value}' to {expected}")

        # Test integer conversion
        self.assertEqual(config._convert_value("42"), 42)
        self.assertEqual(config._convert_value("0"), False)  # 0 is converted to False, not int

        # Test float conversion
        self.assertEqual(config._convert_value("3.14"), 3.14)
        self.assertEqual(config._convert_value("0.0"), 0.0)

        # Test string passthrough
        self.assertEqual(config._convert_value("hello"), "hello")
        self.assertEqual(config._convert_value(""), "")

    @patch.dict(os.environ, {}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_file_loading_json(self, mock_get_logger):
        """Test loading configuration from JSON files."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config = ConfigurationSystem()

        # Test JSON file loading
        test_config = {
            "logging": {"level": "ERROR"},
            "transports": {"sse": {"port": 8080}},
            "custom": {"setting": "value"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_config, f)
            json_file = f.name

        try:
            config.load_from_file(json_file)

            # Test that values were loaded and merged
            self.assertEqual(config.get("logging", "level"), "ERROR")
            self.assertEqual(config.get("transports", "sse", "port"), 8080)
            self.assertEqual(config.get("custom", "setting"), "value")

            # Test that other defaults are still present
            self.assertEqual(config.get("logging", "directory"), "logs")
            self.assertTrue(config.get("transports", "sse", "enabled"))

        finally:
            os.unlink(json_file)

    @patch("yaml.safe_load")
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="logging:\n  level: WARNING")
    @patch.dict(os.environ, {}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_file_loading_yaml(self, mock_get_logger, mock_file, mock_exists, mock_yaml):
        """Test loading configuration from YAML files."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Mock file existence
        mock_exists.return_value = True

        # Mock yaml.safe_load to return test data
        mock_yaml.return_value = {"logging": {"level": "WARNING"}}

        config = ConfigurationSystem()

        # Test YAML file loading
        config.load_from_file("test.yaml")

        # Verify yaml.safe_load was called
        mock_yaml.assert_called_once()

        # Verify configuration was loaded
        self.assertEqual(config.get("logging", "level"), "WARNING")

    @patch.dict(os.environ, {}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_file_loading_errors(self, mock_get_logger):
        """Test error handling in file loading."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config = ConfigurationSystem()

        # Test FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            config.load_from_file("nonexistent.json")

        # Test unsupported file format
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"some content")
            txt_file = f.name

        try:
            with self.assertRaises(ValueError):
                config.load_from_file(txt_file)
        finally:
            os.unlink(txt_file)

    @patch.dict(os.environ, {}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_configuration_access(self, mock_get_logger):
        """Test configuration value access methods."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config = ConfigurationSystem()

        # Test get with default
        self.assertEqual(config.get("nonexistent", default="default_value"), "default_value")

        # Test get without default
        self.assertIsNone(config.get("nonexistent"))

        # Test has method
        self.assertTrue(config.has("logging", "level"))
        self.assertFalse(config.has("nonexistent", "key"))

        # Test get_section
        logging_section = config.get_section("logging")
        self.assertIsInstance(logging_section, dict)
        self.assertIn("level", logging_section)

        # Test get_all
        all_config = config.get_all()
        self.assertIsInstance(all_config, dict)
        self.assertIn("logging", all_config)
        self.assertIn("transports", all_config)
        self.assertIn("application", all_config)

    @patch.dict(os.environ, {}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_configuration_modification(self, mock_get_logger):
        """Test configuration value modification."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config = ConfigurationSystem()

        # Test set method
        config.set("custom_value", "custom", "key")
        self.assertEqual(config.get("custom", "key"), "custom_value")

        # Test overriding default values
        original_level = config.get("logging", "level")
        config.set("CRITICAL", "logging", "level")
        self.assertEqual(config.get("logging", "level"), "CRITICAL")
        self.assertNotEqual(config.get("logging", "level"), original_level)

    @patch.dict(os.environ, {}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_transport_specific_methods(self, mock_get_logger):
        """Test transport-specific configuration methods."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config = ConfigurationSystem()

        # Test get_transport_config
        sse_config = config.get_transport_config("sse")
        self.assertIsInstance(sse_config, dict)
        self.assertTrue(sse_config.get("enabled"))
        self.assertEqual(sse_config.get("host"), "localhost")
        self.assertEqual(sse_config.get("port"), 8000)

        # Test nonexistent transport
        unknown_config = config.get_transport_config("unknown")
        self.assertEqual(unknown_config, {})

        # Test is_transport_enabled
        self.assertTrue(config.is_transport_enabled("sse"))
        self.assertTrue(config.is_transport_enabled("stdio"))
        self.assertFalse(config.is_transport_enabled("unknown"))

    @patch.dict(os.environ, {}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_configuration_validation(self, mock_get_logger):
        """Test configuration validation."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config = ConfigurationSystem()

        # Test valid configuration
        errors = config.validate()
        self.assertEqual(len(errors), 0)

        # Test invalid logging level
        config.set("INVALID", "logging", "level")
        errors = config.validate()
        self.assertTrue(any("Invalid logging level" in error for error in errors))

        # Reset and test invalid port
        config.set("INFO", "logging", "level")  # Reset to valid
        config.set(-1, "transports", "sse", "port")
        errors = config.validate()
        self.assertTrue(any("Invalid port" in error for error in errors))

        # Reset and test invalid timeout
        config.set(8000, "transports", "sse", "port")  # Reset to valid
        config.set(-5.0, "application", "request_timeout")
        errors = config.validate()
        self.assertTrue(any("Invalid request timeout" in error for error in errors))

        # Reset and test invalid max requests
        config.set(30.0, "application", "request_timeout")  # Reset to valid
        config.set(0, "application", "max_concurrent_requests")
        errors = config.validate()
        self.assertTrue(any("Invalid max concurrent requests" in error for error in errors))

    @patch.dict(os.environ, {"AIDER_MCP_TEST_VALUE": "original"}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_configuration_reload(self, mock_get_logger):
        """Test configuration reload functionality."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config = ConfigurationSystem()

        # Check initial value
        self.assertEqual(config.get("test", "value"), "original")

        # Modify environment and reload
        with patch.dict(os.environ, {"AIDER_MCP_TEST_VALUE": "updated"}):
            config.reload()
            self.assertEqual(config.get("test", "value"), "updated")

    @patch.dict(os.environ, {}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_nested_path_access(self, mock_get_logger):
        """Test accessing deeply nested configuration paths."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config = ConfigurationSystem()

        # Set deeply nested value
        config.set("deep_value", "level1", "level2", "level3", "key")

        # Test retrieval
        self.assertEqual(config.get("level1", "level2", "level3", "key"), "deep_value")

        # Test partial path retrieval
        level2_dict = config.get("level1", "level2")
        self.assertIsInstance(level2_dict, dict)
        self.assertEqual(level2_dict["level3"]["key"], "deep_value")

        # Test nonexistent nested path
        self.assertIsNone(config.get("level1", "nonexistent", "key"))

    @patch.dict(os.environ, {}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_configuration_merging(self, mock_get_logger):
        """Test configuration merging behavior."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config = ConfigurationSystem()

        # Load configuration that partially overlaps with defaults
        test_config = {
            "logging": {
                "level": "WARNING",  # Override default
                "custom_setting": "custom_value",  # Add new setting
            },
            "new_section": {"new_key": "new_value"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_config, f)
            json_file = f.name

        try:
            config.load_from_file(json_file)

            # Test that defaults are preserved
            self.assertEqual(config.get("logging", "directory"), "logs")
            self.assertEqual(config.get("logging", "format"), "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

            # Test that overrides work
            self.assertEqual(config.get("logging", "level"), "WARNING")

            # Test that new values are added
            self.assertEqual(config.get("logging", "custom_setting"), "custom_value")
            self.assertEqual(config.get("new_section", "new_key"), "new_value")

        finally:
            os.unlink(json_file)

    @patch.dict(os.environ, {}, clear=True)
    @patch("aider_mcp_server.configuration_system.get_logger")
    def test_edge_cases(self, mock_get_logger):
        """Test edge cases and error conditions."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        config = ConfigurationSystem()

        # Test empty path
        self.assertIsNone(config.get())

        # Test setting with empty path should not crash
        try:
            config.set("value")  # This should not work but shouldn't crash
        except TypeError:
            pass  # Expected

        # Test accessing non-dict as dict
        config.set("string_value", "test")
        self.assertIsNone(config.get("test", "nonexistent"))

        # Test has with empty path
        self.assertFalse(config.has())


if __name__ == "__main__":
    unittest.main()
