import json
import os
import sys
from reasoning_agent import MPIReasoningAgent

# Define the e-base scaling law verification problem
problem_data = {
    "problem": "Prove that the function f(x) = ln(x)/x achieves its global maximum at x = e for x > 0. This is known as the Steiner's problem or the maximum of x^(1/x).",
    "formal_statement_template": """import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

open Real

theorem max_ln_div_x (x : ℝ) (hx : x > 0) :
  (Real.log x) / x ≤ (Real.log (Real.exp 1)) / (Real.exp 1) := by"""
}

def run_demo():
    print("Initializing MPI Reasoning Agent...")
    # Initialize with a smaller embedding model for speed if needed, but distilbert is fine
    agent = MPIReasoningAgent()
    
    if not agent.api_key:
        print("Error: DEEPSEEK_API_KEY not set. Cannot run formalization demo.")
        return

    print("\n" + "="*50)
    print("STARTING DEMO: E-Base Scaling Law Verification")
    print("="*50 + "\n")

    # Solve
    lean_code = agent.solve(problem_data)
    
    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    
    if lean_code:
        print("Generated Lean Code:")
        print(lean_code)
        
        # Save to file for inspection
        output_file = "EBaseProof.lean"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(lean_code)
        print(f"\nSaved proof to {output_file}")
    else:
        print("Failed to generate a solution.")

if __name__ == "__main__":
    run_demo()
