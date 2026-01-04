
import os
from kaggle.api.kaggle_api_extended import KaggleApi

def search_aimo():
    api = KaggleApi()
    api.authenticate()
    
    print("Searching for 'aimo' competitions...")
    competitions = api.competitions_list(search="aimo")
    
    # Debug: print type and dir of competitions
    print(f"Type of competitions: {type(competitions)}")
    # Try to access underlying list if it's a wrapper
    if hasattr(competitions, 'data'):
        print("Using .data attribute")
        competitions = competitions.data
    elif hasattr(competitions, 'competitions'):
        competitions = competitions.competitions
        
    # Check if it's still not iterable, print dir
    try:
        iter(competitions)
    except TypeError:
        print(f"Attributes: {dir(competitions)}")
        print(f"Content: {competitions}")
        return

    found = False
    for comp in competitions:
        print(f"Found: {comp.ref} - {comp.title}")
        # The ref might be a URL or slug depending on SDK version
        slug = comp.ref.split('/')[-1] if '/' in comp.ref else comp.ref
        
        if "olympiad" in slug.lower() or "aimo" in slug.lower():
            found = True
            print(f"Targeting: {slug}")
            download_aimo(api, slug)
            # Don't break immediately, let's see what else is there, but maybe just try the first relevant one
            # actually let's try to find the official one
            if slug == "ai-mathematical-olympiad-prize":
                break
    
    if not found:
        print("No specific AIMO competition found via search. Trying explicit slug 'ai-mathematical-olympiad-prize'...")
        download_aimo(api, "ai-mathematical-olympiad-prize")

def download_aimo(api, comp_ref):
    target_dir = "competitions/AIMO/dataset"
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading files for {comp_ref} to {target_dir}...")
    try:
        api.competition_download_files(comp_ref, path=target_dir)
        print("Download complete. Unzipping...")
        
        import zipfile
        zip_path = os.path.join(target_dir, f"{comp_ref}.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print("Unzip complete.")
        else:
            # Sometimes kaggle downloads as competition name zip
            print(f"Warning: {zip_path} not found. Checking other zips...")
            for f in os.listdir(target_dir):
                if f.endswith(".zip"):
                    with zipfile.ZipFile(os.path.join(target_dir, f), 'r') as zip_ref:
                        zip_ref.extractall(target_dir)
                    print(f"Unzipped {f}")
                    
    except Exception as e:
        print(f"Error downloading: {e}")

if __name__ == "__main__":
    search_aimo()
