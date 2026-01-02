import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

def search_competitions():
    # 确保认证信息存在
    # 假设 kaggle.json 在 ~/.kaggle/ 或者 D:\code\MPI\tools\kaggle-mcp\src\kaggle_server\kaggle.json
    # 为了保险，我们手动设置一下环境变量（如果用户把key放在了特定位置）
    # 但通常 kaggle 库会自动查找 ~/.kaggle/kaggle.json
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        print("Searching for competitions...")
        # 搜索 'LLM', 'NLP', 'Transformer' 相关的竞赛
        competitions = api.competitions_list(category="all", search="LLM")
        
        print(f"\nFound {len(competitions)} competitions matching 'LLM':")
        for comp in competitions:
            print(f"- [{comp.ref}] {comp.title}")
            print(f"  Category: {comp.category}, Reward: {comp.reward}")
            print(f"  Deadline: {comp.deadline}")
            print("-" * 30)
            
        # 也可以搜一下 'text'
        competitions_text = api.competitions_list(category="all", search="text")
        print(f"\nFound {len(competitions_text)} competitions matching 'text':")
        for comp in competitions_text[:5]: # 只列出前5个
            print(f"- [{comp.ref}] {comp.title}")
            print("-" * 30)

    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure kaggle.json is in C:\\Users\\<User>\\.kaggle\\kaggle.json")

if __name__ == "__main__":
    search_competitions()
