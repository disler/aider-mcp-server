# Port Isolation Fix for SSE Tests

## Problem
The SSE integration tests were failing when run together because they used hard-coded ports (8766, 8767) which caused conflicts when tests ran in parallel or sequentially.

## Solution
1. Created a `free_port` fixture that dynamically allocates an available port for each test
2. Created a `server_process` fixture to manage subprocess lifecycle and ensure proper cleanup
3. Updated all SSE tests to use the dynamic port allocation
4. Fixed timeout handling in tests to prevent hanging processes

## Changes Made

### tests/conftest.py
- Added `free_port` fixture that finds an available port using socket binding
- Added `server_process` fixture that handles subprocess cleanup

### Updated Tests
1. **test_sse_integration_working_dir.py**
   - Uses `free_port` fixture instead of hard-coded port 8767
   - Uses `server_process` fixture for better process management
   - Fixed timeout handling for subprocess communication

2. **test_sse_simple_working_dir.py**
   - Uses `free_port` fixture instead of hard-coded port 8767
   - Changed from `subprocess.run` to `subprocess.Popen` for better control
   - Fixed timeout handling

3. **test_working_dir_fix.py**
   - Uses `free_port` fixture instead of hard-coded port 8766

4. **test_parallel_port_safety.py** (new)
   - Added test to verify port isolation works correctly
   - Tests that multiple servers can start on different ports

## Result
- Tests now pass when run in parallel or sequentially
- No more port conflicts
- Proper cleanup of server processes
- Better error handling for timeouts
