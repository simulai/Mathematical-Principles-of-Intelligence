import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_data():
    target_dir = "data/nlp-getting-started"
    os.makedirs(target_dir, exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()
    
    print(f"Downloading competition data to {target_dir}...")
    api.competition_download_files("nlp-getting-started", path=target_dir)
    
    zip_path = os.path.join(target_dir, "nlp-getting-started.zip")
    if os.path.exists(zip_path):
        print("Unzipping...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("Done!")
        # Optional: remove zip
        # os.remove(zip_path)
    else:
        print("Error: Zip file not found after download attempt.")

if __name__ == "__main__":
    download_data()
