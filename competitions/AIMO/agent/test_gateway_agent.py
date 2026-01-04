import sys
import os
import polars as pl

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "DarkAGI-AIMO")

# Add dataset dir to sys.path to allow importing kaggle_evaluation
sys.path.append(DATASET_DIR)

try:
    from kaggle_evaluation.aimo_2_inference_server import AIMO2InferenceServer
except ImportError:
    print("Error: Could not import kaggle_evaluation. Make sure the dataset is downloaded and extracted.")
    sys.exit(1)

from reasoning_agent import MPIReasoningAgent

from visualize_entropy import plot_entropy_landscape

# Initialize Agent
print("Initializing MPI Agent...")
agent = MPIReasoningAgent()

# Output dir for visualizations
VIS_DIR = os.path.join(CURRENT_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

def predict(*args):
    """
    Inference function called by the gateway.
    """
    try:
        # print(f"DEBUG: Received {len(args)} arguments")
        
        # Try to identify columns from args (Series)
        id_series = None
        problem_series = None
        
        for arg in args:
            if isinstance(arg, pl.Series):
                if arg.name == 'id':
                    id_series = arg
                elif arg.name == 'problem':
                    problem_series = arg
        
        if id_series is not None and problem_series is not None:
            problem = problem_series[0]
            problem_id = id_series[0]
            # Create a submission dataframe from scratch or based on id_series
            submission_df = pl.DataFrame({"id": [problem_id], "answer": [0]}) 
        elif len(args) > 0 and isinstance(args[0], pl.DataFrame) and 'problem' in args[0].columns:
             # Case where it might be passed as a DataFrame
             df = args[0]
             problem = df['problem'][0]
             problem_id = df['id'][0]
             submission_df = pl.DataFrame({"id": [problem_id], "answer": [0]})
        else:
             print("Error: Could not find id/problem in args. Args received:")
             for i, arg in enumerate(args):
                  print(f"{i}: {type(arg)} - {arg}")
             return pl.DataFrame({"id": [], "answer": []})

        print(f"\n--- Processing Problem ID: {problem_id} ---")
        
        problem_data = {
            "problem": problem,
            "formal_statement_template": ""
        }
        
        solution = agent.solve(problem_data)
        
        if solution and solution['answer'] is not None:
            answer = solution['answer']
            # AIMO requires answer modulo 1000
            answer = int(answer) % 1000
            
            # Phase Transition Logging
            if 'all_scores' in solution:
                print(f"  [Phase Transition Data] Problem: {problem_id}")
                for score in solution['all_scores']:
                    print(f"    Path: {score['name']} | Z-Score: {score['z_score']:.6f} | Answer: {score['answer']}")
                
                # Generate Visualization
                try:
                    plot_entropy_landscape(str(problem_id), solution['all_scores'], VIS_DIR)
                except Exception as ve:
                    print(f"Visualization Error: {ve}")

        else:
            answer = 0 # Default fallback
            
        print(f"Final Answer: {answer}")
        
        # Update submission DataFrame
        return submission_df.with_columns(pl.Series("answer", [answer]))
        
    except Exception as e:
        print(f"Error processing {problem_id if 'problem_id' in locals() else 'unknown'}: {e}")
        import traceback
        traceback.print_exc()
        # Return fallback
        if 'submission_df' in locals() and submission_df is not None:
             return submission_df.with_columns(pl.Series("answer", [0]))
        else:
             # Try to construct minimal fallback if possible
             if 'problem_id' in locals():
                 return pl.DataFrame({"id": [problem_id], "answer": [0]})
             return pl.DataFrame({"id": [], "answer": []})

def main():
    print("Starting AIMO 2 Inference Server (Local Test Mode)...")
    
    # Check if test.csv exists
    test_path = os.path.join(DATASET_DIR, "test.csv")
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found.")
        return

    # Initialize Server with the predict callback
    # Note: passing the function directly because templates.py seems to pass the argument as is to a *args function
    server = AIMO2InferenceServer(predict)
    
    # Run local gateway
    # This simulates the competition loop
    server.run_local_gateway(data_paths=[test_path])

if __name__ == "__main__":
    main()
