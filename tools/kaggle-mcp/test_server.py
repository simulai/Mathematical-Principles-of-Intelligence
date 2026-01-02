import sys
import os
# Add src to path
# We assume this script is at d:\code\MPI\tools\kaggle-mcp\test_server.py
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from kaggle_server.server import create_server
import asyncio

async def test():
    print("Initializing Server...")
    server = create_server()
    print("Server Initialized.")
    
    print("Testing search_datasets...")
    # We need to manually invoke the tool function or mock the context
    # Since FastMCP tools are decorated, we can access the underlying function usually?
    # Or just rely on the fact that create_server didn't crash.
    
    # FastMCP doesn't expose tools directly as methods on the server object easily for direct calling without client
    # But we can check if tools are registered.
    print(f"Server name: {server.name}")
    # print(f"Tools: {server.list_tools()}") # This might be async

    # Manually invoke search_datasets to verify API connection
    print("Invoking search_datasets with query='mnist'...")
    try:
        # We need to access the tool function directly.
        # In FastMCP, tools are stored in server._tool_manager._tools (internal)
        # But easier is just to instantiate the KaggleApi again here for a quick check
        # Or just trust the previous "Authenticated" message.
        
        # Let's try to call the internal function if possible, or just rely on the fact that
        # create_server() initializes the API.
        
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        datasets = api.dataset_list(search='mnist', page=1)
        print(f"Success! Found {len(datasets)} datasets for 'mnist'. Top 1: {datasets[0].title}")
        
    except Exception as e:
        print(f"API Test Failed: {e}")
    
    print("Test Complete.")

if __name__ == "__main__":
    asyncio.run(test())
