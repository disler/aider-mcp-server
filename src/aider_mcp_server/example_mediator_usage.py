import asyncio
import logging  # Using standard logging for example simplicity
from typing import Any, Dict, Optional, Set

from aider_mcp_server.atoms.event_types import EventTypes
from aider_mcp_server.event_coordinator import EventCoordinator
from aider_mcp_server.event_mediator import EventMediator
from aider_mcp_server.event_participant import EventParticipantBase, IEventParticipant
from aider_mcp_server.event_system import EventSystem
from aider_mcp_server.interfaces.transport_registry import TransportAdapterRegistry
from aider_mcp_server.mcp_types import LoggerFactory, LoggerProtocol


# Basic logger factory for the example
def get_example_logger(name: str) -> LoggerProtocol:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to see verbose messages
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Add a verbose method if it doesn't exist, mapping to debug
    if not hasattr(logger, "verbose"):
        logger.verbose = logger.debug  # type: ignore
    return logger  # type: ignore


example_logger_factory: LoggerFactory = get_example_logger


class StatusReporter(EventParticipantBase):
    def __init__(self, mediator: EventMediator, logger_factory: LoggerFactory):
        super().__init__("StatusReporterParticipant", mediator, logger_factory)
        # Automatically register upon creation for this example
        asyncio.create_task(self.register_with_mediator())

    def get_handled_events(self) -> Set[EventTypes]:
        return {EventTypes.STATUS, EventTypes.PROGRESS}

    async def handle_event(
        self, event_type: EventTypes, data: Dict[str, Any], originator: Optional[IEventParticipant]
    ) -> None:
        originator_name = originator.get_participant_name() if originator else "N/A"
        self._logger.info(
            f"{self.get_participant_name()} received event: {event_type.value} from {originator_name}. Data: {data}"
        )
        if event_type == EventTypes.STATUS and data.get("message") == "System Ready":
            self._logger.info(f"{self.get_participant_name()} sees System is Ready! Might do something...")


class CommandEmitter(EventParticipantBase):
    def __init__(self, mediator: EventMediator, logger_factory: LoggerFactory):
        super().__init__("CommandEmitterParticipant", mediator, logger_factory)
        # No automatic registration, will be done explicitly or not at all if only emitting

    def get_handled_events(self) -> Set[EventTypes]:
        # This component only emits, doesn't handle any specific events from others
        return set()

    async def handle_event(
        self, event_type: EventTypes, data: Dict[str, Any], originator: Optional[IEventParticipant]
    ) -> None:
        # Should not be called if get_handled_events is empty and it's registered
        self._logger.warning(f"{self.get_participant_name()} unexpectedly received event: {event_type.value}")

    async def send_startup_signal(self) -> None:
        self._logger.info(f"{self.get_participant_name()} is sending a startup STATUS event.")
        await self.emit_event_via_mediator(
            EventTypes.STATUS, {"message": "System Ready", "component": self.get_participant_name()}
        )

    async def report_progress(self, percentage: int) -> None:
        self._logger.info(f"{self.get_participant_name()} is reporting progress: {percentage}%")
        await self.emit_event_via_mediator(
            EventTypes.PROGRESS, {"value": percentage, "source": self.get_participant_name()}
        )


async def main() -> None:
    logger = example_logger_factory("MediatorExample")
    logger.info("--- Starting Mediator Pattern Example ---")

    # 1. Initialize core systems
    # TransportRegistry might require async initialization if it discovers plugins
    transport_registry = await TransportAdapterRegistry.get_instance()
    # For this example, assume built-in adapters are sufficient and initialize directly
    # await transport_registry.initialize() # If explicit init is needed and not covered by get_instance

    event_system = EventSystem(transport_registry)
    event_mediator = EventMediator(example_logger_factory, event_system)

    # 2. Initialize EventCoordinator (now uses EventMediator)
    event_coordinator = EventCoordinator(example_logger_factory, event_mediator)

    # 3. Create and register participants
    status_reporter = StatusReporter(event_mediator, example_logger_factory)
    # (StatusReporter registers itself in its __init__ for this example)

    command_emitter = CommandEmitter(event_mediator, example_logger_factory)
    # CommandEmitter does not handle events, so registration is optional if only emitting.
    # If it were to handle events, it would need:
    # await command_emitter.register_with_mediator()

    logger.info("\n--- Simulating Internal Event Emission ---")
    # Simulate a component emitting a STATUS event internally
    await command_emitter.send_startup_signal()
    await asyncio.sleep(0.1)  # Allow time for event processing

    # Simulate a component emitting a PROGRESS event internally
    await command_emitter.report_progress(50)
    await asyncio.sleep(0.1)

    logger.info("\n--- Simulating External Event Broadcast via EventCoordinator ---")
    # Simulate ApplicationCoordinator telling EventCoordinator to broadcast something
    # This will go through Mediator -> EventSystem -> (potentially) actual transports
    # (Actual transport sending depends on EventSystem's dummy setup for this example)
    await event_coordinator.broadcast_event(
        EventTypes.STATUS,
        {"message": "External Broadcast Test", "status": "testing"},
        test_mode=True,  # test_mode in EventSystem might affect logging or direct awaits
    )
    await asyncio.sleep(0.1)

    # Example of sending to a specific (hypothetical) transport
    await event_coordinator.send_event_to_transport(
        "hypothetical_transport_id_123",
        EventTypes.TOOL_RESULT,
        {"tool_name": "example_tool", "result": "success"},
        test_mode=True,
    )
    await asyncio.sleep(0.1)

    logger.info("\n--- Simulating Transport Subscription Management via EventCoordinator ---")
    # These calls will be forwarded by EventCoordinator to EventMediator, then to EventSystem
    await event_coordinator.subscribe_to_event_type("transport_A", EventTypes.STATUS)
    is_sub = await event_coordinator.is_subscribed("transport_A", EventTypes.STATUS)
    logger.info(f"Is transport_A subscribed to STATUS? {is_sub}")  # EventSystem's is_subscribed is dummy
    await asyncio.sleep(0.1)

    logger.info("\n--- Cleaning up participants (example) ---")
    await status_reporter.unregister_from_mediator()
    # command_emitter wasn't registered for handling, so no unregistration needed for that.

    logger.info("--- Mediator Pattern Example Finished ---")


if __name__ == "__main__":
    # Setup basic asyncio loop if running as a script
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Example interrupted.")
