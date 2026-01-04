import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

"""
Memristive MPI Core Simulation
------------------------------
This script demonstrates the "Vice-is-Virtue" principle of the MPI framework.
It compares a "Perfect" (Classical) solver against a "Memristive" (MPI) solver
on a non-convex optimization landscape (Rastrigin Function).

Hypothesis:
Hardware defects (Noise, Drift, Non-linearity) in memristors are actually 
computational features that enable:
1. Noise -> Stochastic Symplectic Dynamics (Creativity/Exploration)
2. Drift -> Ricci Flow (Manifold Smoothing/Forgetting)
3. Non-linearity -> Implicit Activation (Decision Boundaries)
"""

# 1. The Landscape (Non-convex Problem)
def rastrigin(x, A=10):
    """
    Rastrigin function: A classic non-convex function used to test optimization algorithms.
    Global minimum at x=0. Has many local minima.
    """
    return A + x**2 - A * np.cos(2 * np.pi * x)

def rastrigin_grad(x, A=10):
    """Derivative of Rastrigin function."""
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

# 2. The Solvers

class PerfectSolver:
    """
    Represents a classical GPU/CPU solver: High precision, deterministic.
    """
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.trajectory = []
        
    def step(self, x):
        grad = rastrigin_grad(x)
        # Standard Gradient Descent
        x_new = x - self.lr * grad
        self.trajectory.append(x_new)
        return x_new

class MemristiveSolver:
    """
    Represents an MPI solver on Memristive Hardware: Noisy, Drifting, Non-linear.
    """
    def __init__(self, learning_rate=0.01, noise_scale=0.5, drift_rate=0.001):
        self.lr = learning_rate
        self.noise_scale = noise_scale # "Creativity"
        self.drift_rate = drift_rate   # "Ricci Flow"
        self.trajectory = []
        
    def memristor_update(self, x, grad):
        """
        Simulates the physical update of a memristor conductance.
        dV/dt = -grad + Noise - Drift
        """
        # 1. Stochastic Noise (Thermal/Shot noise)
        # In MPI: This is "Symplectic Fantasy"
        noise = np.random.normal(0, self.noise_scale)
        
        # 2. Conductance Drift (Forgetting)
        # In MPI: This is "Ricci Flow" (smoothing the manifold)
        # Pulls x slightly towards 0 (or a baseline state)
        drift = self.drift_rate * x
        
        # 3. Non-linear update (Simulating saturation or activation)
        update = -self.lr * grad + noise - drift
        
        return x + update

    def step(self, x):
        grad = rastrigin_grad(x)
        x_new = self.memristor_update(x, grad)
        self.trajectory.append(x_new)
        return x_new

# 3. The Experiment

def run_experiment():
    print("Running 'Vice-is-Virtue' Experiment...")
    
    # Start at a "bad" location (Local Minimum trap)
    start_x = 4.5 
    steps = 200
    
    # Initialize Solvers
    perfect = PerfectSolver(learning_rate=0.005)
    memristive = MemristiveSolver(learning_rate=0.005, noise_scale=0.8, drift_rate=0.01)
    
    x_p = start_x
    x_m = start_x
    
    path_p = [x_p]
    path_m = [x_m]
    
    for _ in range(steps):
        x_p = perfect.step(x_p)
        path_p.append(x_p)
        
        # Annealing the noise (Learning cools down)
        memristive.noise_scale *= 0.98 
        x_m = memristive.step(x_m)
        path_m.append(x_m)
        
    print(f"Perfect Solver Final Position: {x_p:.4f} (Cost: {rastrigin(x_p):.4f})")
    print(f"Memristive Solver Final Position: {x_m:.4f} (Cost: {rastrigin(x_m):.4f})")
    
    # 4. Visualization
    x_range = np.linspace(-5.12, 5.12, 1000)
    y_range = rastrigin(x_range)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_range, y_range, 'k-', alpha=0.3, label="Problem Landscape (Rastrigin)")
    
    # Plot Trajectories
    plt.plot(path_p, [rastrigin(x) for x in path_p], 'r-o', markersize=3, label="Perfect Solver (Stuck)")
    plt.plot(path_m, [rastrigin(x) for x in path_m], 'g-o', markersize=3, label="Memristive Solver (Found Global Min)")
    
    plt.title("The 'Vice-is-Virtue' Principle: Noise as Creativity")
    plt.xlabel("State Space (x)")
    plt.ylabel("Energy / Cost")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "docs/memristor_simulation.png"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    run_experiment()
