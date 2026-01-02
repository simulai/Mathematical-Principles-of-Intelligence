from mcp.server.fastmcp import FastMCP, Context
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import json

def create_server():
    server = FastMCP("Kaggle MCP Server")

    # Initialize API
    # Credentials are now correctly set in ~/.kaggle/kaggle.json
    try:
        api = KaggleApi()
        api.authenticate()
        print("Kaggle API Authenticated")
    except Exception as e:
        print(f"Warning: Kaggle API authentication failed: {e}")
        print("Please ensure C:/Users/szzj/.kaggle/kaggle.json contains your {username, key}.")

    @server.tool()
    def search_datasets(query: str, ctx: Context) -> list:
        """
        Search for datasets on Kaggle.
        Returns a list of datasets with their title, ref, and url.
        """
        try:
            datasets = api.dataset_list(search=query, page=1)
            results = []
            for d in datasets[:10]: # Limit to 10
                results.append({
                    "title": d.title,
                    "ref": d.ref,
                    "url": d.url,
                    "size": d.size,
                    "lastUpdated": str(d.lastUpdated)
                })
            return results
        except Exception as e:
            return [f"Error searching datasets: {str(e)}"]

    @server.tool()
    def search_competitions(query: str, ctx: Context) -> list:
        """
        Search for competitions on Kaggle.
        Returns a list of competitions with title, ref, and category.
        """
        try:
            competitions = api.competitions_list(search=query, page=1)
            results = []
            for c in competitions[:10]:
                results.append({
                    "title": c.title,
                    "ref": c.ref,
                    "category": c.category,
                    "reward": c.reward,
                    "url": c.url
                })
            return results
        except Exception as e:
            return [f"Error searching competitions: {str(e)}"]
            
    @server.tool()
    def list_kernels(query: str = "", language: str = None, ctx: Context = None) -> list:
        """
        List kernels (notebooks) on Kaggle.
        """
        try:
            kernels = api.kernels_list(search=query, language=language, page=1)
            results = []
            for k in kernels[:10]:
                results.append({
                    "title": k.title,
                    "ref": k.ref,
                    "language": k.language,
                    "totalVotes": k.totalVotes,
                    "url": f"https://www.kaggle.com/{k.ref}"
                })
            return results
        except Exception as e:
            return [f"Error listing kernels: {str(e)}"]

    @server.tool()
    def deploy_kernel(folder_path: str, ctx: Context) -> str:
        """
        Deploy (push) a kernel to Kaggle.
        folder_path: Absolute path to the folder containing kernel-metadata.json and the notebook/script source.
        """
        try:
            if not os.path.exists(folder_path):
                return f"Error: Folder path {folder_path} does not exist."
            
            # Use the Kaggle API to push the kernel
            # kernel_push looks for kernel-metadata.json in the folder
            result = api.kernel_push(folder_path)
            return f"Kernel pushed successfully: {result}"
        except Exception as e:
            return f"Error pushing kernel: {str(e)}"

    return server
