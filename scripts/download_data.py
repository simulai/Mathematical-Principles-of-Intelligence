import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
try:
    from scripts.create_dummy_data import create_dummy_data
except ImportError:
    from create_dummy_data import create_dummy_data

def download_data():
    target_dir = "data/nlp-getting-started"
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        print("Attempting to connect to Kaggle API...")
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
        else:
            print("Error: Zip file not found after download attempt.")
            raise Exception("Download failed")
            
    except Exception as e:
        print(f"\n[Warning] Kaggle API download failed: {e}")
        print("Switching to DUMMY DATA mode for demonstration purposes.")
        print("-------------------------------------------------------")
        create_dummy_data()
        print("Dummy data created successfully.")

if __name__ == "__main__":
    download_data()
