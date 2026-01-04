@echo off
set "PATH=%USERPROFILE%\.elan\bin;%PATH%"
set "LEAN_PROJECT_PATH=D:\code\MPI\competitions\AIMO\lean_solver"
"D:\code\MPI\tools\lean-mcp\.venv\Scripts\python.exe" -m lean_lsp_mcp %*
