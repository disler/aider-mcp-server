"""
Shutdown context protocol definition extracted to prevent circular imports.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ShutdownContextProtocol(Protocol):
    """Protocol defining only the members needed by shutdown context managers."""

    def get_transport_id(self) -> str: ...

    async def shutdown(self) -> None: ...
