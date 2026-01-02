import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

def submit_to_kaggle():
    # Configuration
    COMPETITION = "nlp-getting-started"
    SUBMISSION_FILE = "submission.csv"
    MESSAGE = "MPI SPHA + Zhang Invariant (Automated Submission)"

    # Check if file exists
    if not os.path.exists(SUBMISSION_FILE):
        print(f"Error: {SUBMISSION_FILE} not found. Please run run_kaggle_demo.bat first.")
        sys.exit(1)

    print(f"Preparing to submit {SUBMISSION_FILE} to {COMPETITION}...")
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        # Submit
        print("Submitting...")
        api.competition_submit(
            file_name=SUBMISSION_FILE,
            message=MESSAGE,
            competition=COMPETITION
        )
        print("Submission successful!")
        print("Check your position on the leaderboard: https://www.kaggle.com/c/nlp-getting-started/leaderboard")
        
    except Exception as e:
        print(f"An error occurred during submission: {e}")
        print("Troubleshooting:")
        print("1. Ensure you have accepted the competition rules.")
        print("2. Ensure your kaggle.json is valid.")

if __name__ == "__main__":
    submit_to_kaggle()
