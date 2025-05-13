import asyncio
import collections
import pickle
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, cast

from aider_mcp_server.mcp_types import AsyncTask


# Helper function for size estimation
def get_object_size(obj: Any) -> int:
    """Estimate the size of a Python object in bytes using pickling."""
    try:
        # Use HIGHEST_PROTOCOL for potentially better efficiency/compactness
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        # Fallback for objects that cannot be pickled (should be rare for diffs)
        # This might not be accurate for complex objects and will likely underestimate.
        # print(f"Warning: Could not pickle object for size estimation: {type(obj)}") # Optional warning
        return sys.getsizeof(obj)  # Fallback might underestimate significantly


class DiffCache:
    """
    A cache for storing file diffs with time-based expiry and size-based LRU eviction.

    Supports tracking cache statistics and periodic cleanup.
    """

    def __init__(
        self,
        expiry_duration: timedelta = timedelta(minutes=5),
        max_size: int = 10 * 1024 * 1024,  # Default 10MB
        cleanup_interval: timedelta = timedelta(minutes=1),  # How often cleanup runs
    ):
        """
        Initializes the DiffCache.

        Args:
            expiry_duration: How long cache entries are valid.
            max_size: Maximum size of the cache in bytes. Defaults to 10MB.
            cleanup_interval: How often the background cleanup task runs.
        """
        self._expiry_duration = expiry_duration
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval

        # Cache stores {'key': {'data': ..., 'size': ..., 'expiry': ...}}
        # OrderedDict maintains LRU order (last accessed/set is at the end)
        self._cache: collections.OrderedDict[str, Dict[str, Any]] = (
            collections.OrderedDict()
        )
        self._current_size: int = 0

        self._stats: Dict[str, Union[int, float]] = {
            "hits": 0,
            "misses": 0,
            "total_accesses": 0,
            "current_size": 0,  # This will mirror _current_size
            "max_size": max_size,  # Include max_size in stats
        }

        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._cleanup_task: Optional[AsyncTask[None]] = None

    async def start(self) -> None:
        """Starts the background cleanup task."""
        if self._cleanup_task is None or (
            self._cleanup_task is not None and self._cleanup_task.done()
        ):
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Periodic task to remove expired items."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for interval or until shutdown is requested
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._cleanup_interval.total_seconds(),
                )
                if self._shutdown_event.is_set():
                    break  # Exit if shutdown was requested
            except asyncio.TimeoutError:
                # Timeout occurred, proceed with cleanup
                pass

            await self._perform_cleanup()

    async def _perform_cleanup(self) -> None:
        """Removes expired items from the cache."""
        async with self._lock:
            now = datetime.now()
            keys_to_remove = []
            # Iterate over a copy to allow deletion during iteration
            for key, entry in list(self._cache.items()):
                if entry["expiry"] < now:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                # Check existence again in case it was removed by another operation (e.g., eviction)
                if key in self._cache:
                    entry = self._cache.pop(key)
                    self._current_size -= entry["size"]
            self._stats["current_size"] = self._current_size  # Update stats

    def _recursive_diff(self, old_data: Any, new_data: Any) -> Any:
        """
        Recursively compare two data structures (dicts or other types).

        Returns:
            None if data is identical.
            A dictionary containing changes if both are dicts.
            The new_data value if types differ or are not dicts and values differ.
        """
        if old_data == new_data:
            return None  # No change

        if isinstance(old_data, dict) and isinstance(new_data, dict):
            changes = {}
            all_keys = set(old_data.keys()) | set(new_data.keys())
            for key in all_keys:
                old_val = old_data.get(key)
                new_val = new_data.get(key)
                # If values differ or a key is only in one dict, include the new value
                if old_val != new_val:
                    # Recursively check for nested changes if both are dicts
                    if isinstance(old_val, dict) and isinstance(new_val, dict):
                        nested_changes = self._recursive_diff(old_val, new_val)
                        if nested_changes is not None:
                            changes[key] = nested_changes
                        # If nested_changes is None, it means the nested dicts are the same,
                        # so we don't add the key to the changes dict.
                    else:
                        # Simple value change or key added/removed
                        changes[key] = new_val
            return changes if changes else None
        else:
            # Simple value comparison (strings, numbers, lists, etc.)
            # If they are not equal and not both dicts, the new value is the change
            return new_data

    async def set(self, key: str, diff: Dict[str, Any]) -> None:
        """Sets a diff in the cache with an expiry time."""
        async with self._lock:
            now = datetime.now()
            expiry = now + self._expiry_duration
            item_size = get_object_size(diff)

            # If the key already exists, remove its old size before adding the new one
            if key in self._cache:
                old_entry = self._cache.pop(key)  # Remove to update LRU position later
                self._current_size -= old_entry["size"]

            # Ensure space for the new item
            # Evict LRU items if adding this one exceeds max_size
            while (
                self._current_size + item_size > self._max_size and len(self._cache) > 0
            ):
                # Pop the oldest item (least recently used)
                _key, _entry = self._cache.popitem(last=False)
                self._current_size -= _entry["size"]

            # Add the new item
            self._cache[key] = {"data": diff, "size": item_size, "expiry": expiry}
            self._current_size += item_size
            self._stats["current_size"] = self._current_size  # Update stats

            # Move the newly set item to the end (most recently used)
            # This is automatically handled by OrderedDict when setting a key,
            # but explicitly calling move_to_end makes the intent clear and
            # handles the case where the key was already present.
            self._cache.move_to_end(key)

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves a diff from the cache if not expired."""
        async with self._lock:
            self._stats["total_accesses"] += 1
            entry = self._cache.get(key)
            now = datetime.now()

            if entry and entry["expiry"] > now:
                self._stats["hits"] += 1
                # Move the accessed item to the end (most recently used)
                self._cache.move_to_end(key)
                return cast(Dict[str, Any], entry["data"])
            else:
                self._stats["misses"] += 1
                # Remove expired or non-existent entry
                if key in self._cache:
                    entry = self._cache.pop(key)
                    self._current_size -= entry["size"]
                    self._stats["current_size"] = self._current_size  # Update stats
                return None

    async def compare_and_cache(
        self,
        key: str,
        new_diff: Dict[str, Any],
        clear_cached_for_unchanged: bool = False,
    ) -> Dict[str, Any]:
        """
        Compares new_diff with cached diff, updates cache, and returns changes.

        If clear_cached_for_unchanged is True, only the changed parts are cached.
        If clear_cached_for_unchanged is True and there are no changes, the entry is removed.

        Args:
            key: The cache key (e.g., file paths string).
            new_diff: The new diff data to compare against and potentially cache.
            clear_cached_for_unchanged: If True, cache only the changes; remove if no changes.

        Returns:
            A dictionary representing the changes between the old cached diff (or empty)
            and the new_diff. Returns {} if there are no changes.
        """
        async with self._lock:
            self._stats["total_accesses"] += 1
            old_entry = self._cache.get(key)
            now = datetime.now()

            old_diff = None
            if old_entry and old_entry["expiry"] > now:
                self._stats["hits"] += 1
                old_diff = old_entry["data"]
                # Move the accessed item to the end (most recently used)
                self._cache.move_to_end(key)
            else:
                self._stats["misses"] += 1
                # Remove expired or non-existent entry
                if key in self._cache:
                    entry = self._cache.pop(key)
                    self._current_size -= entry["size"]
                    self._stats["current_size"] = self._current_size  # Update stats

            # Calculate changes
            if old_diff is None:
                # If no old diff, the entire new diff is considered changes
                changes = new_diff
            else:
                # Otherwise, calculate the difference
                changes = self._recursive_diff(old_diff, new_diff)
                # _recursive_diff returns None if no changes, convert to empty dict for consistency
                changes = changes or {}

            # Determine what data to cache and if we should cache at all
            data_to_cache = new_diff  # Default: cache the full new diff
            should_cache = True

            if clear_cached_for_unchanged:
                if changes:  # If there are changes, cache only the changes
                    data_to_cache = changes
                else:  # If no changes and flag is true, remove from cache
                    should_cache = False
                    data_to_cache = {}  # Use empty dict instead of None for consistent typing

            # --- Eviction and Caching/Removal Logic ---

            # If the key is currently in the cache, remove its old size first
            # This handles the case where the item was accessed and moved to end,
            # but not removed because it wasn't expired.
            if key in self._cache:
                old_entry_for_removal = self._cache.pop(
                    key
                )  # Remove before adding new version
                self._current_size -= old_entry_for_removal["size"]

            if should_cache:
                item_size = get_object_size(data_to_cache)
                expiry = now + self._expiry_duration

                # Ensure space for the item to be cached
                # Evict LRU items if adding this one exceeds max_size
                while (
                    self._current_size + item_size > self._max_size
                    and len(self._cache) > 0
                ):
                    # Pop the oldest item (least recently used)
                    _key, _entry = self._cache.popitem(last=False)
                    self._current_size -= _entry["size"]

                # Add the new item
                self._cache[key] = {
                    "data": data_to_cache,
                    "size": item_size,
                    "expiry": expiry,
                }
                self._current_size += item_size
                # Move the newly set item to the end (most recently used)
                self._cache.move_to_end(key)
            # else: should_cache is False, item was already removed if it existed.

            self._stats["current_size"] = self._current_size  # Update stats

            return changes

    async def clear(self, key: Optional[str] = None) -> None:
        """
        Clears the cache entirely or a specific key if provided.

        Args:
            key: Optional key to clear. If None, clears the entire cache.
        """
        async with self._lock:
            if key is not None:
                if key in self._cache:
                    entry = self._cache.pop(key)
                    self._current_size -= entry["size"]
                    self._stats["current_size"] = self._current_size
            else:
                self._cache.clear()
                self._current_size = 0
                self._stats["current_size"] = 0

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Returns cache statistics including hits, misses, total accesses,
        current size, max size, and hit rate.
        """
        # Acquire lock briefly to get a consistent snapshot
        # Note: This method is synchronous, but acquiring an asyncio.Lock
        # in a sync method is generally discouraged if the lock is also used
        # in async methods, as it can lead to deadlocks if the async method
        # is waiting for the lock held by the sync method.
        # However, in this specific case, the lock is only held briefly
        # to copy the stats dictionary, and no awaits happen inside the lock.
        # A truly thread-safe sync method would need a threading.Lock, but
        # mixing asyncio and threading locks requires careful design.
        # Given the context (async cache), assuming this method is called
        # from an async context or a context where the asyncio loop is running
        # and the lock is not contended by long-running sync operations.
        # A safer approach for a sync method would be to run it in a thread pool,
        # but that adds complexity. Let's keep it simple for now, assuming
        # typical asyncio usage patterns.
        # A better approach might be to make get_stats async.
        # Let's make get_stats async to be consistent with lock usage.
        # async with self._lock: # This would require changing the method signature
        # Let's keep it sync for backward compatibility as requested, but add a note.
        # The risk is low if the lock is only held for a copy.
        # A simple copy without the lock might even be acceptable for stats,
        # accepting slightly inconsistent numbers, but the lock is safer.

        # Reverting to async get_stats for safety with asyncio.Lock
        # This breaks strict backward compatibility of the method signature,
        # but is necessary for correct lock usage.
        # If strict sync signature is required, a threading.Lock would be needed,
        # or accept potential inconsistency/deadlock risk.
        # Let's assume changing get_stats to async is acceptable for correctness.

        # async with self._lock: # This is the correct way with asyncio.Lock
        #     stats_copy = self._stats.copy()
        #     total = stats_copy.get('total_accesses', 0)
        #     hits = stats_copy.get('hits', 0)
        #     stats_copy['hit_rate'] = (hits / total) if total > 0 else 0.0
        #     return stats_copy

        # Alternative: Return a copy without lock. Stats might be slightly stale but won't block.
        # This is often acceptable for monitoring stats.
        stats_copy = self._stats.copy()
        total = stats_copy.get("total_accesses", 0)
        hits = stats_copy.get("hits", 0)
        stats_copy["hit_rate"] = (hits / total) if total > 0 else 0.0
        return stats_copy

    async def shutdown(self) -> None:
        """Signals the cleanup task to stop and waits for it."""
        self._shutdown_event.set()
        if self._cleanup_task is not None and not self._cleanup_task.done():
            try:
                # Wait for the task to finish with a timeout
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except asyncio.TimeoutError:
                # If it doesn't finish in time, cancel it
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None  # Clear reference

    async def __aenter__(self) -> "DiffCache":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Async context manager exit, ensures shutdown."""
        await self.shutdown()
