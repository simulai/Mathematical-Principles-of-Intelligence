import os
import json
import sys

def setup_kaggle_config():
    print("Setting up Kaggle Configuration...")
    
    home = os.path.expanduser("~")
    kaggle_dir = os.path.join(home, ".kaggle")
    
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
        print(f"Created directory: {kaggle_dir}")
        
    config_path = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(config_path):
        print(f"kaggle.json already exists at {config_path}")
        overwrite = input("Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            return

    username = input("Enter Kaggle Username: ")
    key = input("Enter Kaggle Key: ")
    
    config = {"username": username, "key": key}
    
    with open(config_path, 'w') as f:
        json.dump(config, f)
        
    print(f"Configuration saved to {config_path}")
    print("You can now use the Kaggle MCP server!")

if __name__ == "__main__":
    setup_kaggle_config()
