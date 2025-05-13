"""
Type stubs for the sse_starlette library to be used with mypy.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Union, MutableMapping
from starlette.responses import Response
from starlette.types import Receive, Scope, Send
from asyncio import Queue

class EventSourceResponse(Response):
    """
    Server-Sent Events response.
    Implements the Server-Sent Events specification for the Starlette web framework.
    """
    
    def __init__(
        self,
        generator: AsyncGenerator[Union[str, Dict[str, str]], None],
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        media_type: str = "text/event-stream",
        **kwargs: Any
    ) -> None:
        """
        Initialize the EventSourceResponse.
        
        Args:
            generator: An async generator that yields either strings (pre-formatted SSE messages)
                      or dicts with 'event' and 'data' keys.
            status_code: HTTP status code.
            headers: HTTP headers.
            media_type: Media type, defaults to "text/event-stream".
            **kwargs: Additional keyword arguments for the Response constructor.
        """
        pass

    async def listen(self) -> AsyncGenerator[bytes, None]:
        """Listen to the generator and yield bytes for the response."""
        pass
        
    async def __call__(self, scope: MutableMapping[str, Any], receive: Receive, send: Send) -> None:
        """
        ASGI application implementation for the EventSourceResponse.
        
        Args:
            scope: The ASGI connection scope dictionary
            receive: An async callable for receiving ASGI messages
            send: An async callable for sending ASGI messages
        
        Returns:
            None
        
        Raises:
            ClientDisconnect: If the client disconnects during response streaming
        """
        pass