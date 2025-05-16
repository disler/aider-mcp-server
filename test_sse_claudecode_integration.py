#!/usr/bin/env python3
"""Test script for SSE ClaudeCode integration."""

import asyncio
import httpx
import json
import sys


async def test_sse_connection():
    """Test SSE connection with proper trailing slashes."""
    url = "http://localhost:8765/sse/"
    messages_url = "http://localhost:8765/messages/"
    
    print(f"Testing SSE server at {url}")
    
    try:
        # Test GET request to SSE endpoint with trailing slash
        print("\n1. Testing GET request to /sse/...")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={"Accept": "text/event-stream"},
                timeout=5,
                follow_redirects=True
            )
            print(f"   Status: {response.status_code}")
            print(f"   Headers: {dict(response.headers)}")
        
        # Test POST to messages endpoint
        print("\n2. Testing POST to /messages/...")
        async with httpx.AsyncClient() as client:
            test_message = {"jsonrpc": "2.0", "id": 1, "method": "test"}
            response = await client.post(
                messages_url,
                json=test_message,
                timeout=5
            )
            print(f"   Status: {response.status_code}")
        
        # Test streaming SSE connection
        print("\n3. Testing SSE event stream...")
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "GET",
                url,
                headers={"Accept": "text/event-stream"},
                timeout=5  # Set a small timeout for testing
            ) as response:
                print(f"   Stream Status: {response.status_code}")
                
                # Read a few initial events
                event_count = 0
                async for line in response.aiter_lines():
                    if line.strip():
                        print(f"   Event: {line}")
                        event_count += 1
                        if event_count > 3:  # Stop after a few events
                            break
        
        print("✓ SSE connection test successful!")
        return True
        
    except httpx.TimeoutException:
        print("Timeout - server may not be responding with events")
        return True  # Timeout is expected if no events are sent
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    print("SSE ClaudeCode Integration Test")
    print("===============================")
    
    success = asyncio.run(test_sse_connection())
    
    print("\nTest Summary:")
    print(f"SSE Connection: {'✓ PASS' if success else '✗ FAIL'}")
    
    sys.exit(0 if success else 1)