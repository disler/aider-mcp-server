import asyncio
import logging
from typing import Type, TypeVar, Any, Callable, Awaitable, Optional, Dict

T = TypeVar("T")
logger = logging.getLogger(__name__)

class SingletonManager:
    """
    Manages singleton instances with support for asynchronous initialization.
    """
    _instances: Dict[Type[Any], Any] = {}
    _locks: Dict[Type[Any], asyncio.Lock] = {}

    @classmethod
    async def get_instance(
        cls,
        target_cls: Type[T],
        async_init_func: Optional[Callable[..., Awaitable[T]]] = None,
        *init_args: Any,
        **init_kwargs: Any,
    ) -> T:
        """
        Gets or creates a singleton instance of the target class.

        Args:
            target_cls: The class to get/create an instance of.
            async_init_func: An optional asynchronous function to call for initialization.
                             If None, target_cls(*init_args, **init_kwargs) is called.
                             If provided, it's called as async_init_func(*init_args, **init_kwargs).
            *init_args: Positional arguments for target_cls constructor or async_init_func.
            **init_kwargs: Keyword arguments for target_cls constructor or async_init_func.

        Returns:
            The singleton instance of target_cls.
        """
        logger.debug(f"Requesting instance for {target_cls.__name__}")
        if target_cls not in cls._locks:
            # Ensure lock creation is atomic.
            cls._locks.setdefault(target_cls, asyncio.Lock())
            logger.debug(f"Created lock for {target_cls.__name__}")

        lock = cls._locks[target_cls]

        if target_cls not in cls._instances:
            logger.debug(f"Instance for {target_cls.__name__} not found, acquiring lock for creation.")
            async with lock:
                # Double-check after acquiring the lock
                if target_cls not in cls._instances:
                    logger.info(f"Creating new instance for {target_cls.__name__}")
                    try:
                        if async_init_func:
                            logger.debug(f"Using async_init_func for {target_cls.__name__}")
                            instance = await async_init_func(*init_args, **init_kwargs)
                        else:
                            logger.debug(f"Directly instantiating {target_cls.__name__}")
                            instance = target_cls(*init_args, **init_kwargs)
                        cls._instances[target_cls] = instance
                        logger.info(f"Instance for {target_cls.__name__} created and cached.")
                    except Exception as e:
                        logger.error(f"Error creating instance for {target_cls.__name__}: {e}", exc_info=True)
                        raise
                else:
                    logger.debug(f"Instance for {target_cls.__name__} found after acquiring lock (created by another task).")
            logger.debug(f"Lock for {target_cls.__name__} released.")
        else:
            logger.debug(f"Returning existing instance for {target_cls.__name__}")
            
        return cls._instances[target_cls]  # type: ignore[return-value]

    @classmethod
    def reset_instance(cls, target_cls: Type[Any]) -> None:
        """
        Resets a specific singleton instance. Intended for testing purposes.
        """
        if target_cls in cls._instances:
            del cls._instances[target_cls]
            logger.info(f"Reset singleton instance for {target_cls.__name__}")
        # Optionally remove the lock if it's certain it won't be needed again,
        # or to ensure a completely clean state for tests.
        # if target_cls in cls._locks:
        #     del cls._locks[target_cls]

    @classmethod
    def reset_all_instances(cls) -> None:
        """
        Resets all singleton instances. Intended for testing purposes.
        """
        cls._instances.clear()
        cls._locks.clear()
        logger.info("Reset all singleton instances.")
