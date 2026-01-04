import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys

# Setup
plt.style.use('dark_background')
def plot_entropy_landscape(problem_id, candidate_scores, output_dir, ground_truth=None):
    """
    Plots the MPI Entropy Landscape for a given problem.
    """
    fig, ax = plt.subplots(figsize=(12, 8)) # Create new figure each time
    
    # Define styles
    correct_color = '#00FF00' # Bright Green
    wrong_color = '#FF4500'   # Orange Red
    neutral_color = '#1E90FF' # Dodger Blue
    
    for i, score in enumerate(candidate_scores):
        entropy_traj = score['step_entropy']
        steps = range(1, len(entropy_traj) + 1)
        
        # Determine color based on correctness (if known)
        if ground_truth is not None:
            # Check if answer matches ground truth (allow for some tolerance if needed)
            try:
                # Assuming simple integer comparison for AIMO
                is_correct = int(score['answer']) % 1000 == int(ground_truth) % 1000
                color = correct_color if is_correct else wrong_color
                status_label = "Correct" if is_correct else "Incorrect"
            except:
                color = neutral_color
                status_label = "Unknown"
        else:
            # Fallback to plasma map if no ground truth
            color = plt.cm.plasma(i / len(candidate_scores))
            status_label = f"Path {i}"
    
        # Plot trajectory
        label = f"{status_label} (Z={score['z_score']:.4f})"
        ax.plot(steps, entropy_traj, marker='o', linestyle='-', linewidth=2, color=color, label=label, alpha=0.7)
        
        # Highlight answer extraction point
        ax.scatter(steps[-1], entropy_traj[-1], color=color, s=100, edgecolors='white', zorder=10)

    ax.set_title(f'Thermodynamic Trajectory of Thought: Problem {problem_id}\n(Entropy Production vs. Reasoning Step)', fontsize=16, color='white')
    ax.set_xlabel('Cognitive Time (Steps)', fontsize=12)
    ax.set_ylabel('Surrogate Entropy ($\mathcal{S}_{proxy}$)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)
    
    # Critical Point Annotation
    # Heuristic: Find step where standard deviation of entropy across paths maximizes (divergence point)
    try:
        max_len = max(len(s['step_entropy']) for s in candidate_scores)
        std_devs = []
        for t in range(max_len):
            vals = [s['step_entropy'][t] for s in candidate_scores if t < len(s['step_entropy'])]
            if len(vals) > 1:
                std_devs.append(np.std(vals))
            else:
                std_devs.append(0)
        
        if std_devs:
            critical_step = np.argmax(std_devs)
            critical_val = std_devs[critical_step]
            # Plot vertical line at critical point
            ax.axvline(x=critical_step + 1, color='yellow', linestyle=':', alpha=0.8, label='Critical Point')
            ax.annotate('Critical Divergence\n(Bifurcation)', 
                        xy=(critical_step + 1, max(s['step_entropy'][min(critical_step, len(s['step_entropy'])-1)] for s in candidate_scores)), 
                        xytext=(critical_step + 1.5, max(s['step_entropy'][0] for s in candidate_scores)),
                        arrowprops=dict(facecolor='yellow', shrink=0.05),
                        fontsize=10, color='yellow')
    except Exception as e:
        print(f"Could not compute critical point: {e}")

    output_path = os.path.join(output_dir, f'{problem_id}_entropy_landscape.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Explicitly close figure
    print(f"Generated Entropy Landscape: {output_path}")

if __name__ == "__main__":
    # Test with dummy data if run directly
    dummy_scores = [
        {"name": "Path A (Correct)", "z_score": 0.001, "step_entropy": [0.1, 0.05, 0.01, 0.001, 0.0005]},
        {"name": "Path B (Wrong)", "z_score": 0.05, "step_entropy": [0.1, 0.08, 0.06, 0.05, 0.07]},
        {"name": "Path C (Confused)", "z_score": 0.12, "step_entropy": [0.1, 0.15, 0.12, 0.20, 0.18]}
    ]
    plot_entropy_landscape("TEST_001", dummy_scores, ".")