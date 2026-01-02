import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

def sinkhorn_projection(matrix, iterations=10):
    """
    Projects a matrix onto the Birkhoff Polytope (Doubly Stochastic Matrices).
    This represents the mHC (DeepSeek) approach: rigid stability.
    """
    M = np.exp(matrix) # Ensure positivity
    # Handle numerical instability
    M[np.isinf(M)] = 1e30
    M[np.isnan(M)] = 1e-10
    
    for _ in range(iterations):
        # Row normalization
        row_sum = M.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1e-10
        M /= row_sum
        
        # Column normalization
        col_sum = M.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1e-10
        M /= col_sum
    return M

class CognitiveAgent:
    def __init__(self, dim, mode='mHC'):
        self.dim = dim
        self.mode = mode # 'mHC' or 'MPI'
        self.W = np.random.randn(dim, dim) * np.sqrt(2/dim)
        if self.mode == 'mHC':
            self.W = sinkhorn_projection(self.W)
        
        self.temperature = 0.01 # "Cognitive Temperature"
        self.learning_rate = 0.1
        
        # History
        self.loss_history = []
        self.norm_history = []
        self.temp_history = []

    def forward(self, x):
        return np.dot(self.W, x)

    def adapt(self, x, y_target):
        # 1. Prediction
        y_pred = self.forward(x)
        
        # 2. Loss (Prediction Error)
        error = y_pred - y_target
        loss = np.mean(error**2)
        
        # 3. Gradient Descent (Standard Backprop)
        grad_W = 2 * np.outer(error, x) / self.dim
        
        # 4. MPI Special: Thermodynamics of Learning
        if self.mode == 'MPI':
            # Temperature rises with Surprise (Loss)
            # T ~ Loss (Simplified)
            target_temp = min(loss * 5.0, 1.0) 
            self.temperature = 0.9 * self.temperature + 0.1 * target_temp
        else:
            self.temperature = 0.0 # mHC is "Zero-Temperature" Limit
            
        # 5. Weight Update with "Elastic" Constraint
        # W_new = W_old - lr * grad
        self.W -= self.learning_rate * grad_W
        
        # 6. Manifold Projection (The Key Difference)
        if self.mode == 'mHC':
            # Hard Constraint: ALWAYS project to Birkhoff Polytope
            self.W = sinkhorn_projection(self.W)
            norm_deviation = 0.0
        elif self.mode == 'MPI':
            # Soft Constraint: Projection depends on Temperature
            # If High Temp (Confusion) -> Relax Constraint (Explore)
            # If Low Temp (Mastery) -> Enforce Constraint (Consolidate)
            
            W_projected = sinkhorn_projection(self.W)
            
            # Phase Transition Function
            # Alpha = 1 (Full Projection) when T -> 0
            # Alpha = 0 (No Projection) when T -> High
            alpha = np.exp(-5.0 * self.temperature)
            
            self.W = alpha * W_projected + (1 - alpha) * self.W
            
            # Measure how far we are from the manifold (Zhang Invariant Violation)
            norm_deviation = np.linalg.norm(self.W - W_projected)

        self.loss_history.append(loss)
        self.norm_history.append(norm_deviation if self.mode == 'MPI' else 0)
        self.temp_history.append(self.temperature)

def run_experiment():
    dim = 16
    steps = 100
    
    agent_mhc = CognitiveAgent(dim, mode='mHC')
    agent_mpi = CognitiveAgent(dim, mode='MPI')
    
    # Target Function (Environment)
    # Phase 1: Identity-like
    W_target = np.eye(dim)
    
    losses_mhc = []
    losses_mpi = []
    norms_mpi = []
    
    print("Starting Simulation...")
    
    for t in range(steps):
        # Paradigm Shift at t=50
        if t == 50:
            print("!!! PARADIGM SHIFT (Distribution Change) !!!")
            # Target becomes random and scaled up (harder)
            W_target = np.random.randn(dim, dim) * 1.5
            
        # Input
        x = np.random.randn(dim)
        y_target = np.dot(W_target, x)
        
        # Adapt
        agent_mhc.adapt(x, y_target)
        agent_mpi.adapt(x, y_target)
        
        losses_mhc.append(agent_mhc.loss_history[-1])
        losses_mpi.append(agent_mpi.loss_history[-1])
        norms_mpi.append(agent_mpi.norm_history[-1])

    # Visualization
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    
    ax_loss = fig.add_subplot(gs[0, :])
    ax_norm = fig.add_subplot(gs[1, :])
    
    # Plot Loss
    ax_loss.plot(losses_mhc, label='mHC (Static Constraint)', color='gray', linestyle='--')
    ax_loss.plot(losses_mpi, label='MPI (Dynamic Ricci Flow)', color='red', linewidth=2)
    ax_loss.axvline(x=50, color='black', linestyle=':', alpha=0.5)
    ax_loss.text(51, max(losses_mpi)*0.8, 'Paradigm Shift', fontsize=10)
    ax_loss.set_title('Learning Performance: MPI vs mHC')
    ax_loss.set_ylabel('Loss (MSE)')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    # Plot Manifold Deviation (Creativity)
    ax_norm.plot(norms_mpi, color='orange', fillstyle='full')
    ax_norm.fill_between(range(steps), norms_mpi, color='orange', alpha=0.3)
    ax_norm.axvline(x=50, color='black', linestyle=':', alpha=0.5)
    ax_norm.set_title('Cognitive Temperature (Manifold Deviation)')
    ax_norm.set_ylabel(r'Zhang Invariant Violation ($\Delta \mathcal{Z}$)')
    ax_norm.set_xlabel('Time Steps')
    ax_norm.text(10, 0.05, 'Stable Phase\n(Low Entropy)', ha='center')
    ax_norm.text(60, 0.05, 'Phase Transition\n(High Entropy/Insight)', ha='center')
    ax_norm.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/images/ricci_flow_learning.png')
    print("Simulation Complete. Saved to docs/images/ricci_flow_learning.png")

if __name__ == "__main__":
    run_experiment()
