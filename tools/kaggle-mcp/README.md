# Kaggle MCP Server

This is an MCP server that interfaces with the Kaggle API.

## Features
- Search Datasets
- Search Competitions
- List Kernels

## Setup

1. Ensure you have your `kaggle.json` in `~/.kaggle/kaggle.json`.
2. Install dependencies:
   ```bash
   pip install -e .
   ```
3. Run the server:
   ```bash
   uv run dev
   ```
