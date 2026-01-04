import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- MPI Utilities ---

def sinkhorn_projection(matrix, iterations=5):
    """
    Projects the router weights onto the Birkhoff Polytope (Doubly Stochastic).
    Ensures that the 'Energy' sent to experts and received by experts is conserved.
    """
    # Softmax first to ensure positivity
    M = torch.exp(matrix)
    
    for _ in range(iterations):
        # Row norm (Conservation of Token Energy)
        row_sum = M.sum(dim=1, keepdim=True) + 1e-6
        M = M / row_sum
        
        # Col norm (Conservation of Expert Capacity)
        col_sum = M.sum(dim=0, keepdim=True) + 1e-6
        M = M / col_sum
        
    return M

# --- MiniMind-style MoE Components ---

class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, dim, num_experts, k=2, mode='Baseline'):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k
        self.mode = mode # 'Baseline' or 'MPI'
        
        # Router
        self.router = nn.Linear(dim, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([Expert(dim, dim*4) for _ in range(num_experts)])
        
        # MPI Specific: Dynamic Temperature
        self.temperature = 1.0

    def forward(self, x):
        # x: [batch_size, dim]
        batch_size = x.shape[0]
        
        # 1. Routing Logits
        router_logits = self.router(x) # [batch, num_experts]
        
        # --- MPI INTERVENTION ---
        if self.mode == 'MPI':
            # Apply Cognitive Holonomy (Sinkhorn) to logits before selection
            # This ensures the routing probability mass is "conserved"
            # Instead of raw softmax, we view the router-token matrix as a transport plan
            
            # Note: Sinkhorn usually works on square matrices. For rectangular [batch, experts],
            # we perform a "Soft-Sinkhorn" that balances expert load.
            
            # Step A: Normalize Logits to avoid explosion
            router_logits = router_logits / self.temperature
            
            # Step B: Sinkhorn-like Balancing (simplified for batch)
            # This encourages equal usage of experts (Load Balancing) naturally
            # without auxiliary loss!
            M = torch.exp(router_logits)
            for _ in range(3):
                M = M / (M.sum(dim=1, keepdim=True) + 1e-6) # Token constraint
                M = M / (M.sum(dim=0, keepdim=True) + 1e-6) # Expert constraint
            
            routing_weights = M
            
            # Select Top-K from the balanced matrix
            top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=1)
            
            # Re-normalize weights to sum to 1 for the token
            top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
            
        else: # Baseline (Standard Top-K)
            routing_weights = F.softmax(router_logits, dim=1)
            top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=1)
            # Standard renormalization
            top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)

        # 2. Dispatch and Aggregate
        final_output = torch.zeros_like(x)
        
        # Naive loop implementation for clarity (not optimized for speed)
        for i in range(batch_size):
            for j in range(self.k):
                expert_idx = top_k_indices[i, j].item()
                weight = top_k_weights[i, j]
                expert_out = self.experts[expert_idx](x[i].unsqueeze(0))
                final_output[i] += weight * expert_out.squeeze(0)
                
        return final_output, top_k_indices

# --- Simulation Experiment ---

def run_simulation():
    # Settings
    dim = 64
    num_experts = 8
    k = 2
    num_tokens = 1000 # Batch size
    
    # Data: A sequence of tokens with shifting patterns
    # Cluster 1: First 500 tokens
    data1 = torch.randn(num_tokens // 2, dim) + torch.tensor([2.0] * dim)
    # Cluster 2: Second 500 tokens (Distribution Shift)
    data2 = torch.randn(num_tokens // 2, dim) - torch.tensor([2.0] * dim)
    data = torch.cat([data1, data2], dim=0)
    
    # Models
    model_baseline = MoELayer(dim, num_experts, k=k, mode='Baseline')
    model_mpi = MoELayer(dim, num_experts, k=k, mode='MPI')
    
    # Copy weights to ensure fair start
    model_mpi.load_state_dict(model_baseline.state_dict())
    
    # Metric: Expert Utilization (Histogram)
    # We want to see if MPI prevents "Expert Collapse" (where only a few experts are used)
    
    print("Running Baseline MoE...")
    _, indices_baseline = model_baseline(data)
    
    print("Running MPI-Enhanced MoE...")
    _, indices_mpi = model_mpi(data)
    
    # Flatten indices
    indices_baseline = indices_baseline.flatten().numpy()
    indices_mpi = indices_mpi.flatten().numpy()
    
    # Calculate Entropy of Expert Distribution (Higher is Better/More Balanced)
    def calculate_entropy(indices, n_experts):
        counts = np.bincount(indices, minlength=n_experts)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log(probs + 1e-10))
    
    entropy_baseline = calculate_entropy(indices_baseline, num_experts)
    entropy_mpi = calculate_entropy(indices_mpi, num_experts)
    
    print(f"Baseline Expert Entropy: {entropy_baseline:.4f}")
    print(f"MPI Expert Entropy:      {entropy_mpi:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(indices_baseline, bins=range(num_experts+1), rwidth=0.8, color='gray', alpha=0.7)
    axes[0].set_title(f"Baseline MoE (Entropy: {entropy_baseline:.2f})\nRisk: Expert Collapse")
    axes[0].set_xlabel("Expert ID")
    axes[0].set_ylabel("Token Count")
    
    axes[1].hist(indices_mpi, bins=range(num_experts+1), rwidth=0.8, color='green', alpha=0.7)
    axes[1].set_title(f"MPI-Enhanced MoE (Entropy: {entropy_mpi:.2f})\nBenefit: Load Balancing (Cognitive Holonomy)")
    axes[1].set_xlabel("Expert ID")
    
    plt.tight_layout()
    plt.savefig('docs/images/moe_mpi_verification.png')
    print("Saved comparison to docs/images/moe_mpi_verification.png")

if __name__ == "__main__":
    run_simulation()