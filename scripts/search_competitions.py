import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

def search_competitions():
    try:
        api = KaggleApi()
        api.authenticate()
        
        print("Searching for competitions...")
        
        # 1. Search specific terms relevant to our MPI theory
        search_terms = ["LLM", "NLP", "Disaster", "Text"]
        
        for term in search_terms:
            print(f"\n{'='*20}\nSearching for: '{term}'\n{'='*20}")
            competitions = api.competitions_list(category="all", search=term, sort_by="prize")
            
            # Handle list response carefully
            count = 0
            for comp in competitions:
                # Filter out closed competitions if needed, but let's show everything for now
                print(f"* [{comp.ref}] {comp.title}")
                print(f"  - Reward: {comp.reward}")
                print(f"  - Category: {comp.category}")
                print(f"  - Deadline: {comp.deadline}")
                print("-" * 30)
                count += 1
                if count >= 5: break # Limit to top 5 per term to avoid spam
            
            if count == 0:
                print("No competitions found.")

    except Exception as e:
        print(f"Error: {e}")
        # print("Traceback:", e.with_traceback())

if __name__ == "__main__":
    search_competitions()
