import unittest
from typing import Any, Callable, Dict, Type

from aider_mcp_server.handler_registry import HandlerRegistry


# --- Mock Handlers and Classes ---
async def mock_echo_handler(request: Dict[str, Any]) -> Dict[str, Any]:
    """A simple handler that echoes back the 'data' field of the request."""
    return {"success": True, "data": request.get("data")}


async def mock_error_handler(request: Dict[str, Any]) -> Dict[str, Any]:
    """A handler that always raises an exception."""
    raise ValueError("Simulated handler error")


class MockHandlerClass:
    """A class containing methods that can be registered as handlers."""

    async def handle_ping(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handles 'ping' requests."""
        return {"success": True, "response": "pong", "params": request.get("params")}

    async def handle_status(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handles 'status' requests."""
        return {"success": True, "status": "all systems nominal"}

    def not_a_handler_method(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """This method should not be registered as it doesn't start with 'handle_'."""
        return {"error": "should not be called"}

    async def handle_(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """This method has 'handle_' prefix but no request type, should be skipped."""
        return {"success": False, "error": "empty request type"}


class EmptyHandlerClass:
    """A class with no handler methods."""

    pass


class TestHandlerRegistry(unittest.IsolatedAsyncioTestCase):
    """Test suite for the HandlerRegistry class."""

    def setUp(self) -> None:
        """Set up for each test."""
        self.registry = HandlerRegistry()

    def assertIsBoundMethod(self, method: Callable, instance_class: Type) -> None:
        """Asserts that the callable is a bound method of an instance of instance_class."""
        self.assertTrue(hasattr(method, "__self__"), "Method has no __self__ attribute (not bound).")
        self.assertIsInstance(method.__self__, instance_class, "Method is not bound to the correct class instance.")

    async def test_register_and_get_handler(self) -> None:
        """Test registration and retrieval of a single handler."""
        self.registry.register_handler("echo", mock_echo_handler)
        handler = self.registry.get_handler("echo")
        self.assertIsNotNone(handler)
        self.assertEqual(handler, mock_echo_handler)
        self.assertIsNone(self.registry.get_handler("nonexistent"))

    async def test_register_handler_overwrite(self) -> None:
        """Test that registering a handler for an existing type overwrites the old one."""
        self.registry.register_handler("test_op", mock_echo_handler)

        # Intentionally register a different handler for the same type
        async def new_mock_handler(request: Dict[str, Any]) -> Dict[str, Any]:
            return {"new_handler": True}

        self.registry.register_handler("test_op", new_mock_handler)

        handler = self.registry.get_handler("test_op")
        self.assertEqual(handler, new_mock_handler)

        # Test via handle_request
        response = await self.registry.handle_request({"type": "test_op"})
        self.assertEqual(response, {"new_handler": True})

    async def test_unregister_handler(self) -> None:
        """Test unregistering a handler."""
        self.registry.register_handler("echo", mock_echo_handler)
        self.assertIsNotNone(self.registry.get_handler("echo"))

        self.registry.unregister_handler("echo")
        self.assertIsNone(self.registry.get_handler("echo"))

        # Test unregistering a non-existent handler (should not raise error)
        self.registry.unregister_handler("nonexistent")

    async def test_register_handler_class(self) -> None:
        """Test registration of handlers from a class."""
        self.registry.register_handler_class(MockHandlerClass)

        ping_handler = self.registry.get_handler("ping")
        self.assertIsNotNone(ping_handler)
        # Check if it's a bound method of MockHandlerClass
        self.assertIsBoundMethod(ping_handler, MockHandlerClass)

        status_handler = self.registry.get_handler("status")
        self.assertIsNotNone(status_handler)
        self.assertIsBoundMethod(status_handler, MockHandlerClass)

        # Method not starting with 'handle_' should not be registered
        self.assertIsNone(self.registry.get_handler("not_a_handler_method"))
        # Method with 'handle_' but no type should not be registered
        self.assertIsNone(self.registry.get_handler(""))

    async def test_register_empty_handler_class(self) -> None:
        """Test registering a class with no handler methods."""
        self.registry.register_handler_class(EmptyHandlerClass)
        self.assertEqual(self.registry.get_supported_request_types(), [])

    async def test_get_supported_request_types(self) -> None:
        """Test retrieving the list of supported request types."""
        self.assertEqual(self.registry.get_supported_request_types(), [])

        self.registry.register_handler("echo", mock_echo_handler)
        self.registry.register_handler("status", mock_error_handler)  # Any handler will do

        supported_types = self.registry.get_supported_request_types()
        self.assertCountEqual(supported_types, ["echo", "status"])  # Order doesn't matter

        self.registry.register_handler_class(MockHandlerClass)
        supported_types_after_class = self.registry.get_supported_request_types()
        self.assertCountEqual(supported_types_after_class, ["echo", "status", "ping"])

    async def test_handle_request_success(self) -> None:
        """Test successful request handling."""
        self.registry.register_handler("echo", mock_echo_handler)
        request_payload = {"type": "echo", "data": "hello world"}
        response = await self.registry.handle_request(request_payload)
        self.assertEqual(response, {"success": True, "data": "hello world"})

    async def test_handle_request_class_handler_success(self) -> None:
        """Test successful request handling by a method from a registered class."""
        self.registry.register_handler_class(MockHandlerClass)
        request_payload = {"type": "ping", "params": {"timeout": 100}}
        response = await self.registry.handle_request(request_payload)
        self.assertEqual(response, {"success": True, "response": "pong", "params": {"timeout": 100}})

    async def test_handle_request_missing_type(self) -> None:
        """Test handling a request that is missing the 'type' field."""
        request_payload = {"data": "no type here"}
        response = await self.registry.handle_request(request_payload)
        self.assertEqual(response, {"success": False, "error": "Missing request type"})

    async def test_handle_request_unknown_type(self) -> None:
        """Test handling a request with an unknown 'type'."""
        request_payload = {"type": "nonexistent_type"}
        response = await self.registry.handle_request(request_payload)
        self.assertEqual(response, {"success": False, "error": "Unknown request type: nonexistent_type"})

    async def test_handle_request_handler_exception(self) -> None:
        """Test handling a request where the handler raises an exception."""
        self.registry.register_handler("error_test", mock_error_handler)
        request_payload = {"type": "error_test"}
        response = await self.registry.handle_request(request_payload)
        self.assertEqual(response, {"success": False, "error": "Error handling request: Simulated handler error"})


if __name__ == "__main__":
    unittest.main()
