@echo off
echo ===================================================
echo MPI Kaggle Competition Demo: Disaster Tweets
echo ===================================================
echo.

echo 1. Checking environment...
python -c "import torch; print(f'Torch version: {torch.__version__}')"
if %errorlevel% neq 0 (
    echo Error: PyTorch not found. Please ensure you have a Python environment with PyTorch installed.
    exit /b
)

echo.
echo 2. Attempting to download real competition data...
python scripts/download_data.py
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Download failed. This is expected if you haven't accepted the competition rules.
    echo Please read INSTRUCTIONS_KAGGLE.md for details on how to accept rules.
    echo.
    echo [INFO] Generating dummy data for verification purposes...
    python scripts/create_dummy_data.py
) else (
    echo [INFO] Data downloaded successfully!
)

echo.
echo 3. Running Model Training (SPHA + Cognitive Holonomy)...
echo This will use real data if available, otherwise dummy data.
echo [INFO] Running in FAST DEMO mode (50 samples) because CPU is detected.
python models/kaggle_disaster_tweets_mpi.py --demo

echo.
echo ===================================================
echo Demo Completed!
echo Submission file generated at: submission.csv
echo ===================================================
pause
