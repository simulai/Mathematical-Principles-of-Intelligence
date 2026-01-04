import sys
import os
import json
import pandas as pd
import numpy as np
import torch
from reasoning_agent import MPIReasoningAgent
from visualize_entropy import plot_entropy_landscape

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
vis_dir = os.path.join(current_dir, "experiment_results")
os.makedirs(vis_dir, exist_ok=True)

# Define Pre-canned Scenarios (Problem + Paths)
# We manually craft paths to represent different cognitive states
# This serves as a "Ground Truth" control for the Z-score metric.
SCENARIOS = {
    "EXP_001": {
        "problem": "Find the sum of all positive integers less than 20 that are multiples of 3 or 5.",
        "ground_truth": 78,
        "candidates": [
            {
                "name": "Path A (Systematic - Correct)",
                "steps": [
                    "Identify multiples of 3 less than 20: 3, 6, 9, 12, 15, 18.",
                    "Identify multiples of 5 less than 20: 5, 10, 15.",
                    "Combine lists and remove duplicates.",
                    "The list is {3, 5, 6, 9, 10, 12, 15, 18}.",
                    "Calculate sum: 3+5=8, 8+6=14, 14+9=23.",
                    "23+10=33, 33+12=45, 45+15=60.",
                    "60+18=78.",
                    "The answer is \\boxed{78}"
                ]
            },
            {
                "name": "Path B (Double Counting - Wrong)",
                "steps": [
                    "List multiples of 3: 3, 6, 9, 12, 15, 18.",
                    "Sum of these is 63.",
                    "List multiples of 5: 5, 10, 15.",
                    "Sum of these is 30.",
                    "Add the two sums together: 63 + 30.",
                    "The total is 93.",
                    "The answer is \\boxed{93}"
                ]
            },
            {
                "name": "Path C (Confused - Wrong)",
                "steps": [
                    "Multiples of 3 and 5.",
                    "3 times 5 is 15.",
                    "Maybe just 15 is the answer?",
                    "But it says sum.",
                    "Is 20 included? Less than 20.",
                    "So 1, 2, 3... wait.",
                    "I will just guess a number near 20.",
                    "Maybe 50?",
                    "The answer is \\boxed{50}"
                ]
            }
        ]
    },
    "EXP_002": {
        "problem": "Solve for x: 2x + 5 = 13.",
        "ground_truth": 4,
        "candidates": [
            {
                "name": "Path A (Direct - Correct)",
                "steps": [
                    "We have the equation 2x + 5 = 13.",
                    "Subtract 5 from both sides.",
                    "2x = 13 - 5.",
                    "2x = 8.",
                    "Divide by 2.",
                    "x = 4.",
                    "The answer is \\boxed{4}"
                ]
            },
             {
                "name": "Path B (Sign Error - Wrong)",
                "steps": [
                    "Equation is 2x + 5 = 13.",
                    "Move 5 to the other side.",
                    "2x = 13 + 5.",
                    "2x = 18.",
                    "Divide by 2.",
                    "x = 9.",
                    "The answer is \\boxed{9}"
                ]
            },
            {
                "name": "Path C (Wandering - Wrong)",
                "steps": [
                    "2x + 5 = 13.",
                    "Maybe x is 1? 2+5=7. No.",
                    "Maybe x is 2? 4+5=9. No.",
                    "Maybe x is 10? 20+5=25. Too big.",
                    "It's somewhere between 2 and 10.",
                    "Let's try 5. 10+5=15. Close.",
                    "I'll say it's 5.",
                    "The answer is \\boxed{5}"
                ]
            }
        ]
    }
}

class ControlledMPIAgent(MPIReasoningAgent):
    def __init__(self):
        # Initialize parent with defaults (loads DistilBERT for Z-score calc)
        super().__init__()
        
    def _generate_candidates(self, problem_text):
        # Override to return our controlled scenarios instead of calling LLM
        for pid, data in SCENARIOS.items():
            if data["problem"] == problem_text:
                return data["candidates"]
        
        print("Warning: Unknown problem text, returning empty.")
        return []

def run_experiment():
    print("==================================================")
    print("   MPI ENTROPY LANDSCAPE: MINIMAL REPRODUCIBLE EXPERIMENT")
    print("==================================================")
    print("Goal: Validate if correct reasoning paths exhibit lower 'Cognitive Holonomy' (Surrogate Entropy).")
    print(f"Output Directory: {vis_dir}\n")

    agent = ControlledMPIAgent()
    
    results = []
    
    for pid, data in SCENARIOS.items():
        print(f"Processing Problem {pid}...")
        
        # Create problem dict as expected by solve()
        problem_input = {"problem": data["problem"]}
        
        # Solve (this calculates H-scores using the real embedding model)
        solution = agent.solve(problem_input)
        
        if solution and 'all_scores' in solution:
            # 1. Generate Visualization
            # Note: Requires visualize_entropy.py update to support ground_truth
            # We map 'h_score' to 'z_score' for compatibility with existing vis code if needed,
            # or update vis code. Let's assume we update vis code or just pass the list.
            # But the list has 'h_score'.
            
            # 2. Collect Data
            for path in solution['all_scores']:
                # Determine correctness
                ans = path['answer']
                is_correct = False
                if ans is not None:
                    try:
                        # Exact integer match
                        is_correct = (int(ans) == int(data['ground_truth']))
                    except:
                        pass
                
                results.append({
                    "problem_id": pid,
                    "path_type": path['name'],
                    "h_score": path['h_score'],
                    "is_correct": is_correct,
                    "final_answer": ans
                })
        else:
            print(f"Error solving {pid}")

    # Analysis
    if results:
        df = pd.DataFrame(results)
        print("\n--- EXPERIMENT DATA ---")
        print(df[['problem_id', 'path_type', 'is_correct', 'h_score']])
        
        # Save
        csv_path = os.path.join(vis_dir, "experiment_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nData saved to: {csv_path}")
        
        # Stats
        correct_group = df[df['is_correct'] == True]['h_score']
        wrong_group = df[df['is_correct'] == False]['h_score']
        
        print("\n--- STATISTICAL SUMMARY ---")
        if not correct_group.empty:
            print(f"Correct Paths (n={len(correct_group)}): Mean H = {correct_group.mean():.6f}, Std = {correct_group.std():.6f}")
        
        if not wrong_group.empty:
            print(f"Incorrect Paths (n={len(wrong_group)}): Mean H = {wrong_group.mean():.6f}, Std = {wrong_group.std():.6f}")
            
        if not correct_group.empty and not wrong_group.empty:
            delta = wrong_group.mean() - correct_group.mean()
            print(f"\nDelta (Wrong - Correct): {delta:.6f}")
            if delta > 0:
                print("RESULT: HYPOTHESIS SUPPORTED. Correct paths have lower Holonomy (Curvature).")
            else:
                print("RESULT: HYPOTHESIS REFUTED. Correct paths have higher/equal Holonomy.")
    
    print("\nExperiment Complete.")

if __name__ == "__main__":
    run_experiment()