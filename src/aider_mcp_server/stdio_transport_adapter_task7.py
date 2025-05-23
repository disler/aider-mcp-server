import asyncio
import json
import sys
from typing import Any, Dict, Optional, Callable, Awaitable, Set

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.interfaces.transport_adapter import TransportAdapterBase, ITransportAdapter
from aider_mcp_server.security import SecurityContext


class StdioTransportAdapter(TransportAdapterBase, ITransportAdapter): # Inherit from TransportAdapterBase and ITransportAdapter
    _stdin_task: Optional[asyncio.Task[None]]
    _stdout_lock: asyncio.Lock
    _running: bool
    _request_handler: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]
    # _transport_id will be handled by TransportAdapterBase

    def __init__(
        self,
        transport_id: str,
        request_handler: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
    ):
        super().__init__(transport_id=transport_id, transport_type="stdio")
        self._request_handler = request_handler
        self._stdin_task = None
        self._stdout_lock = asyncio.Lock()
        self._running = False

    # get_transport_id and get_transport_type are inherited from TransportAdapterBase

    async def initialize(self) -> None:
        # Minimal initialization. Active listening starts with start_listening().
        pass

    async def start_listening(self) -> None:
        if self._running:
            # Already running or start attempted
            return
        
        self._running = True
        if self._stdin_task is None or self._stdin_task.done():
            self._stdin_task = asyncio.create_task(self._read_stdin())

    async def shutdown(self) -> None:
        self._running = False
        if self._stdin_task and not self._stdin_task.done():
            self._stdin_task.cancel()
            try:
                await self._stdin_task
            except asyncio.CancelledError:
                pass  # Expected behavior
        self._stdin_task = None

    def get_capabilities(self) -> Set[EventTypes]:
        return {
            EventTypes.STATUS,
            EventTypes.PROGRESS,
            EventTypes.TOOL_RESULT,
        }

    async def send_event(self, event_type: EventTypes, data: Dict[str, Any]) -> None:
        event_payload = {"type": event_type.value}
        event_payload.update(data)  # Merge data keys at top level

        async with self._stdout_lock:
            try:
                event_json = json.dumps(event_payload)
                print(event_json, flush=True)  # As per Task 7 spec
            except Exception as e:
                # As per Task 7 spec for error handling in send_event
                sys.stderr.write(f"Error sending event to stdout: {str(e)}\n")
                sys.stderr.flush()

    def should_receive_event(
        self,
        event_type: EventTypes,
        data: Dict[str, Any],
        request_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return True

    async def _read_stdin(self) -> None:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        
        try:
            await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        except Exception as e:
            sys.stderr.write(f"Fatal: Error connecting stdin reader: {e}. Stdio adapter will not run.\n")
            sys.stderr.flush()
            self._running = False 
            return

        while self._running:
            try:
                line_bytes = await reader.readline()

                if not line_bytes:  # EOF
                    self._running = False 
                    break 
                
                line_str = line_bytes.decode('utf-8').strip()
                if not line_str: 
                    continue

                try: # Inner try for JSON parsing and request handling (Task 7 structure)
                    message = json.loads(line_str)
                    if self._request_handler:
                        response_dict = await self._request_handler(message)
                        
                        if isinstance(response_dict, dict) and "type" in response_dict:
                            event_type_str = response_dict.pop("type", None)
                            try:
                                event_type_enum = EventTypes(str(event_type_str))
                                await self.send_event(event_type=event_type_enum, data=response_dict)
                            except ValueError: 
                                await self.send_event(
                                    event_type=EventTypes.STATUS,
                                    data={
                                        "level": "error",
                                        "message": f"Handler returned unknown event type: {event_type_str}",
                                        "original_response": response_dict, # Include problematic part
                                    },
                                )
                        elif response_dict is not None: # Handler returned something, but not expected format
                            await self.send_event(
                                event_type=EventTypes.STATUS,
                                data={
                                    "level": "error",
                                    "message": "Handler returned invalid or non-dictionary response format",
                                    "original_response": response_dict,
                                },
                            )
                        # If response_dict is None, handler chose not to send a response.
                
                except json.JSONDecodeError: # Specific catch for JSON errors as per Task 7
                    await self.send_event(
                        event_type=EventTypes.STATUS,
                        data={"level": "error", "message": "Invalid JSON message received on stdin"}
                    )
                # Other exceptions from _request_handler or message processing will be caught by the outer Exception block.

            except asyncio.CancelledError:
                break # Task was cancelled
            except Exception as e: # Outer catch-all from Task 7 for errors in the loop iteration
                await self.send_event(
                    event_type=EventTypes.STATUS,
                    data={"level": "error", "message": f"Error processing stdin: {str(e)}"}
                )
                # Loop continues unless error is persistent or fatal (handled by readline exception logic above)
        
        self._running = False # Ensure consistent state upon loop exit

    async def handle_sse_request(self, request_details: Dict[str, Any]) -> None:
        raise NotImplementedError("SSE requests are not supported by StdioTransportAdapter.")

    async def handle_message_request(self, request_details: Dict[str, Any]) -> None:
        # Task 7's StdioTransportAdapter uses a specific `_request_handler` callback
        # for messages from stdin, which is handled in `_read_stdin`.
        raise NotImplementedError("StdioTransportAdapter uses a dedicated stdin loop and request_handler.")

    def validate_request_security(self, request_details: Dict[str, Any]) -> SecurityContext:
        return SecurityContext(
            user_id="stdio_user", # Identifies the source
            permissions=set(),    # Define permissions as needed; empty implies default or restricted
            is_anonymous=False,   # Stdio is tied to a specific local user context
            transport_id=self._transport_id,
        )
