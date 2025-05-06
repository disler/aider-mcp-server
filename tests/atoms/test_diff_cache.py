import asyncio
import sys
from datetime import timedelta
from unittest.mock import patch

import pytest
import pytest_asyncio

from aider_mcp_server.atoms.diff_cache import DiffCache, get_object_size


@pytest_asyncio.fixture
async def diff_cache():
    """Create a DiffCache with short expiry for testing."""
    cache = DiffCache(expiry_duration=timedelta(seconds=0.1))
    await cache.start()  # Start the cleanup task
    yield cache
    await cache.shutdown()  # Clean up resources after tests

@pytest_asyncio.fixture
async def size_limited_cache():
    """Create a DiffCache with size limit for testing eviction."""
    # Smaller size limit (2KB) to ensure eviction of early entries in tests
    cache = DiffCache(
        expiry_duration=timedelta(minutes=1),
        max_size=2 * 1024  # 2KB
    )
    await cache.start()  # Start the cleanup task
    yield cache
    await cache.shutdown()  # Clean up resources after tests

# Basic functionality tests (similar to original tests)
@pytest.mark.asyncio
async def test_basic_caching(diff_cache):
    file_paths = "file1.txt,file2.txt"
    diff = {"file1.txt": "content1", "file2.txt": "content2"}
    
    await diff_cache.set(file_paths, diff)
    cached_diff = await diff_cache.get(file_paths)
    
    assert cached_diff == diff

@pytest.mark.asyncio
async def test_cache_expiry(diff_cache):
    file_paths = "file1.txt,file2.txt"
    diff = {"file1.txt": "content1", "file2.txt": "content2"}
    
    await diff_cache.set(file_paths, diff)
    await asyncio.sleep(0.2)  # Wait for cache to expire
    cached_diff = await diff_cache.get(file_paths)
    
    assert cached_diff is None

@pytest.mark.asyncio
async def test_compare_and_cache_default(diff_cache):
    file_paths = "file1.txt,file2.txt"
    old_diff = {"file1.txt": "old_content1", "file2.txt": "old_content2"}
    new_diff = {"file1.txt": "new_content1", "file2.txt": "old_content2", "file3.txt": "new_content3"}
    
    await diff_cache.set(file_paths, old_diff)
    changes = await diff_cache.compare_and_cache(file_paths, new_diff)
    
    assert changes == {"file1.txt": "new_content1", "file3.txt": "new_content3"}
    assert await diff_cache.get(file_paths) == new_diff

@pytest.mark.asyncio
async def test_compare_and_cache_clear_unchanged(diff_cache):
    file_paths = "file1.txt,file2.txt"
    old_diff = {"file1.txt": "old_content1", "file2.txt": "old_content2"}
    new_diff = {"file1.txt": "new_content1", "file2.txt": "old_content2", "file3.txt": "new_content3"}
    
    await diff_cache.set(file_paths, old_diff)
    changes = await diff_cache.compare_and_cache(file_paths, new_diff, clear_cached_for_unchanged=True)
    
    assert changes == {"file1.txt": "new_content1", "file3.txt": "new_content3"}
    # With clear_cached_for_unchanged=True, only the changed files should be in the cache
    cached_diff = await diff_cache.get(file_paths)
    assert cached_diff == {"file1.txt": "new_content1", "file3.txt": "new_content3"}
    assert "file2.txt" not in cached_diff  # Unchanged file should be removed

@pytest.mark.asyncio
async def test_nested_diffs(diff_cache):
    file_paths = "file1.txt,file2.txt"
    old_diff = {"file1.txt": {"key1": "value1", "key2": "value2"}, "file2.txt": "content2"}
    new_diff = {"file1.txt": {"key1": "new_value1", "key2": "value2", "key3": "value3"}, "file2.txt": "content2"}
    
    await diff_cache.set(file_paths, old_diff)
    changes = await diff_cache.compare_and_cache(file_paths, new_diff)
    
    # Only key1 (changed) and key3 (new) should be in changes for file1.txt
    assert changes == {"file1.txt": {"key1": "new_value1", "key3": "value3"}}
    assert await diff_cache.get(file_paths) == new_diff

# New tests for improved functionality
@pytest.mark.asyncio
async def test_object_size_estimation():
    """Test that get_object_size correctly estimates object size."""
    # Test with simple objects
    small_obj = "test"
    medium_obj = {"key1": "value1", "key2": "value2"}
    large_obj = {"key1": "x" * 1000, "key2": ["a" * 100] * 10}
    
    # Verify sizes are reasonable
    small_size = get_object_size(small_obj)
    medium_size = get_object_size(medium_obj)
    large_size = get_object_size(large_obj)
    
    assert small_size > 0
    assert medium_size > small_size
    assert large_size > medium_size
    
    # Test fallback for unpicklable objects
    class Unpicklable:
        def __getstate__(self):
            raise TypeError("Cannot pickle")
    
    with patch("pickle.dumps", side_effect=TypeError("Cannot pickle")):
        size = get_object_size(Unpicklable())
        assert size == sys.getsizeof(Unpicklable())

@pytest.mark.asyncio
async def test_lru_eviction(size_limited_cache):
    """Test that least recently used items are evicted when cache exceeds max size."""
    # Create entries with increasing key numbers
    # Each entry will be ~1KB in size
    entries = 10
    base_key = "file"
    
    # Fill cache beyond capacity
    for i in range(entries):
        key = f"{base_key}{i}"
        # Create a large enough diff to trigger eviction
        large_diff = {f"key{j}": "x" * 100 for j in range(10)}
        await size_limited_cache.set(key, large_diff)
    
    # Check that only the first entry was evicted
    # The exact threshold depends on the object size calculation
    assert await size_limited_cache.get(f"{base_key}0") is None  # file0 should be evicted
    
    # The other entries should still be in the cache
    assert await size_limited_cache.get(f"{base_key}1") is not None
    assert await size_limited_cache.get(f"{base_key}2") is not None
    
    # Later entries should still be present
    assert await size_limited_cache.get(f"{base_key}{entries-1}") is not None
    
    # Test that LRU updates work by manually accessing an entry and then adding a new entry
    
    # Make the cache smaller to force another eviction
    size_limited_cache._max_size = 1024
    
    # Access file1 to make it most recently used
    await size_limited_cache.get("file1")
    
    # Add a new entry that will cause eviction
    new_key = f"{base_key}{entries+1}"
    await size_limited_cache.set(new_key, large_diff)
    
    # file1 should still be there because it was recently accessed
    assert await size_limited_cache.get("file1") is not None

@pytest.mark.asyncio
async def test_hit_miss_tracking(diff_cache):
    """Test that hit/miss statistics are tracked correctly."""
    key1 = "file1.txt"
    key2 = "file2.txt"
    
    # Initial stats should show 0 hits and 0 misses
    stats = diff_cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["total_accesses"] == 0
    
    # A miss should increment miss count
    result = await diff_cache.get(key1)
    assert result is None
    stats = diff_cache.get_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0
    assert stats["total_accesses"] == 1
    
    # Set a value and then get it (should be a hit)
    await diff_cache.set(key1, {"content": "test"})
    result = await diff_cache.get(key1)
    assert result is not None
    stats = diff_cache.get_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 1
    assert stats["total_accesses"] == 2
    
    # Another miss for a different key
    result = await diff_cache.get(key2)
    assert result is None
    stats = diff_cache.get_stats()
    assert stats["misses"] == 2
    assert stats["hits"] == 1
    assert stats["total_accesses"] == 3
    
    # Hit rate should be correctly calculated
    assert stats["hit_rate"] == 1/3

@pytest.mark.asyncio
async def test_cache_size_tracking(diff_cache):
    """Test that cache size is tracked correctly."""
    # Cache should start empty
    stats = diff_cache.get_stats()
    assert stats["current_size"] == 0
    
    # Add some content and check size increases
    key = "files.txt"
    diff = {"file1.txt": "x" * 1000}
    await diff_cache.set(key, diff)
    
    stats_after = diff_cache.get_stats()
    assert stats_after["current_size"] > 0
    
    # Clear cache and check size goes back to 0
    await diff_cache.clear()
    stats_cleared = diff_cache.get_stats()
    assert stats_cleared["current_size"] == 0

@pytest.mark.asyncio
async def test_performance_with_recursive_diff():
    """Test the performance of recursive diff calculation with various inputs."""
    cache = DiffCache()
    try:
        # Start the cache
        await cache.start()
        
        # Test with identical diffs (should be fast and return empty changes)
        file_paths = "test_perf.txt"
        diff1 = {"file1.txt": "content", "file2.txt": "content2"}
        
        await cache.set(file_paths, diff1)
        changes = await cache.compare_and_cache(file_paths, diff1.copy())
        
        # Changes should be empty since diffs are identical
        assert not changes
        
        # Test with large nested diffs
        big_diff1 = {
            "file1.txt": {"section1": {"key1": "value1", "key2": "value2"}, "section2": "unchanged"},
            "file2.txt": [1, 2, 3, 4, 5],
            "file3.txt": "x" * 1000
        }
        
        big_diff2 = {
            "file1.txt": {"section1": {"key1": "value1", "key2": "new_value"}, "section2": "unchanged"},
            "file2.txt": [1, 2, 3, 4, 5, 6],
            "file3.txt": "x" * 1000,
            "file4.txt": "new file"
        }
        
        await cache.set("big_files", big_diff1)
        
        # Changes should only contain the actual differences
        changes = await cache.compare_and_cache("big_files", big_diff2)
        
        assert "file1.txt" in changes
        assert "section1" in changes["file1.txt"]
        assert "key2" in changes["file1.txt"]["section1"]
        assert "section2" not in changes["file1.txt"]  # Unchanged section
        assert "file2.txt" in changes  # Modified list
        assert "file3.txt" not in changes  # Unchanged large string
        assert "file4.txt" in changes  # New file
    finally:
        await cache.shutdown()

@pytest.mark.asyncio
async def test_cleanup_task():
    """Test that the cleanup task removes expired entries."""
    # Create cache with very short cleanup interval and expiry
    short_expiry = timedelta(seconds=0.1)
    cleanup_interval = timedelta(seconds=0.2)
    
    cache = DiffCache(
        expiry_duration=short_expiry,
        cleanup_interval=cleanup_interval
    )
    
    try:
        # Start the cache
        await cache.start()
        
        # Add some entries
        await cache.set("key1", {"data": "value1"})
        await cache.set("key2", {"data": "value2"})
        
        # Verify entries exist
        assert await cache.get("key1") is not None
        assert await cache.get("key2") is not None
        
        # Wait for entries to expire and cleanup to run
        await asyncio.sleep(0.3)  # > expiry + cleanup_interval
        
        # Entries should be removed by the cleanup task
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
    finally:
        await cache.shutdown()

@pytest.mark.asyncio
async def test_clear_specific_key(diff_cache):
    """Test that clearing a specific key works correctly."""
    await diff_cache.set("key1", {"data": "value1"})
    await diff_cache.set("key2", {"data": "value2"})
    
    # Clear only key1
    await diff_cache.clear("key1")
    
    # key1 should be gone, key2 should remain
    assert await diff_cache.get("key1") is None
    assert await diff_cache.get("key2") is not None

@pytest.mark.asyncio
async def test_async_context_manager():
    """Test that the async context manager works correctly."""
    async with DiffCache() as cache:
        # Use the cache
        await cache.set("key", {"data": "value"})
        assert await cache.get("key") is not None
    
    # The context manager should have called shutdown
    # This is hard to test directly, but we can verify the task was cancelled
    # by checking if the cleanup_task attribute is None or its done() method returns True
    assert cache._cleanup_task is None or cache._cleanup_task.done()