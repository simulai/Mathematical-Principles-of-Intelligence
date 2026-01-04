import pandas as pd
import os
import sys
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reasoning_agent import MPIReasoningAgent

def main():
    # Configuration
    # Auto-detect dataset path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset", "DarkAGI-AIMO")
    if not os.path.exists(DATASET_DIR):
        DATASET_DIR = os.path.join(BASE_DIR, "dataset") # Fallback
        
    INPUT_FILE = os.path.join(DATASET_DIR, "test.csv")
    OUTPUT_FILE = os.path.join(DATASET_DIR, "submission.csv")
    
    print(f"Dataset Dir: {DATASET_DIR}")
    print(f"Input File: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    # Initialize Agent
    agent = MPIReasoningAgent()
    
    # Load Data
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Processing {len(df)} problems...")

    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        problem_id = row['id']
        problem_text = row['problem']
        
        # print(f"\nProcessing Problem ID: {problem_id}")
        
        # Create a problem dict
        problem_data = {
            "problem": problem_text,
            "formal_statement_template": "" # Template is usually not provided in CSV
        }
        
        try:
            solution = agent.solve(problem_data)
            
            if solution:
                answer = solution['answer']
                if answer is None:
                    answer = 0 # Default fallback
                
                # AIMO usually requires answer modulo 1000
                answer = int(answer) % 1000
                
                # print(f"  -> Answer: {answer}")
                results.append({
                    "id": problem_id,
                    "answer": answer,
                })
            else:
                # print("  -> No solution found.")
                results.append({
                    "id": problem_id,
                    "answer": 0,
                })
                
        except Exception as e:
            print(f"  -> Error processing {problem_id}: {e}")
            results.append({
                "id": problem_id,
                "answer": 0,
            })

    # Save Results
    submission_df = pd.DataFrame(results)
    # Keep only required columns for Kaggle (id, answer)
    final_submission = submission_df[['id', 'answer']]
    final_submission.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSubmission saved to {OUTPUT_FILE}")
    print("Preview:")
    print(final_submission.head())

if __name__ == "__main__":
    main()
