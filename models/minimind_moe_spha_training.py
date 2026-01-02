import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os

# Add parent directory to path to import tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.manifold_metrics import compute_erpm_proxy

# --- MPI Utilities (Sinkhorn) ---
def sinkhorn_projection(matrix, iterations=5):
    M = torch.exp(matrix)
    for _ in range(iterations):
        M = M / (M.sum(dim=1, keepdim=True) + 1e-6)
        M = M / (M.sum(dim=0, keepdim=True) + 1e-6)
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
    def __init__(self, dim, num_experts, k=2, mode='Baseline', expansion_factor=4.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k
        self.mode = mode 
        self.router = nn.Linear(dim, num_experts)
        hidden_dim = int(dim * expansion_factor)
        self.experts = nn.ModuleList([Expert(dim, hidden_dim) for _ in range(num_experts)])
        self.temperature = 1.0

    def forward(self, x):
        batch_size = x.shape[0]
        router_logits = self.router(x)
        
        if self.mode == 'MPI':
            # Sinkhorn Routing (Zhang Invariant)
            router_logits = router_logits / self.temperature
            M = torch.exp(router_logits)
            for _ in range(3):
                M = M / (M.sum(dim=1, keepdim=True) + 1e-6)
                M = M / (M.sum(dim=0, keepdim=True) + 1e-6)
            routing_weights = M
            top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
        else:
            # Baseline Softmax Routing
            routing_weights = F.softmax(router_logits, dim=1)
            top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)

        final_output = torch.zeros_like(x)
        for i in range(batch_size):
            for j in range(self.k):
                expert_idx = top_k_indices[i, j].item()
                weight = top_k_weights[i, j]
                expert_out = self.experts[expert_idx](x[i].unsqueeze(0))
                final_output[i] += weight * expert_out.squeeze(0)
                
        return final_output, top_k_indices

# --- Simple Task Model ---
class MoEClassifier(nn.Module):
    def __init__(self, dim, num_experts, num_classes, mode='Baseline', expansion_factor=4.0):
        super().__init__()
        self.moe = MoELayer(dim, num_experts, k=2, mode=mode, expansion_factor=expansion_factor)
        self.classifier = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        features, indices = self.moe(x)
        logits = self.classifier(features)
        return logits, features, indices

# --- SPHA Scheduler ---
class SPHAScheduler:
    def __init__(self, optimizer, base_lr, lambda_stress=1.0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.lambda_stress = lambda_stress
        self.stress_history = []
        self.lr_history = []

    def step(self, model):
        # Calculate Stress (Gradient Norm)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # SPHA Protocol: eta_eff = eta_0 * exp(-lambda * ||stress||)
        # We normalize stress by parameter count or just use a sensitivity factor
        # Here we treat lambda_stress as the sensitivity
        
        eff_lr = self.base_lr * np.exp(-self.lambda_stress * total_norm)
        
        # Update optimizer LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = eff_lr
            
        self.stress_history.append(total_norm)
        self.lr_history.append(eff_lr)
        return total_norm, eff_lr

# --- Training Loop ---
def run_training_experiment():
    # Setup
    torch.manual_seed(42)
    dim = 64
    num_experts = 8
    num_classes = 2
    num_samples = 1000
    epochs = 50
    base_lr = 0.1 # High LR to test stability
    
    # Data: Two clusters
    # Cluster 0: Mean +1
    X1 = torch.randn(num_samples // 2, dim) + 1.0
    Y1 = torch.zeros(num_samples // 2).long()
    # Cluster 1: Mean -1
    X2 = torch.randn(num_samples // 2, dim) - 1.0
    Y2 = torch.ones(num_samples // 2).long()
    
    X = torch.cat([X1, X2], dim=0)
    Y = torch.cat([Y1, Y2], dim=0)
    
    # Shuffle
    perm = torch.randperm(num_samples)
    X = X[perm]
    Y = Y[perm]
    
    # Models
    # 1. Baseline + Constant LR
    model_base = MoEClassifier(dim, num_experts, num_classes, mode='Baseline', expansion_factor=4.0)
    opt_base = optim.SGD(model_base.parameters(), lr=0.01) # Standard conservative LR
    
    # 2. MPI + SPHA (e-Bottleneck: Reduced Expansion Factor 2.7)
    # Testing the hypothesis that MPI allows smaller models to perform as well as larger ones
    model_mpi = MoEClassifier(dim, num_experts, num_classes, mode='MPI', expansion_factor=2.7)
    # Init weights partially from base for fair comparison of router, but sizes differ so cannot load full state
    # model_mpi.load_state_dict(model_base.state_dict()) # Cannot load due to size mismatch
    opt_mpi = optim.SGD(model_mpi.parameters(), lr=base_lr) # Aggressive Base LR
    scheduler_spha = SPHAScheduler(opt_mpi, base_lr, lambda_stress=0.5)
    
    history = {
        'base': {'loss': [], 'erpm_rank': [], 'erpm_entropy': []},
        'mpi': {'loss': [], 'erpm_rank': [], 'erpm_entropy': [], 'stress': [], 'lr': []}
    }
    
    print(f"Training Start: {epochs} epochs")
    print(f"Baseline: SGD(lr=0.01), ExpFactor=4.0 | MPI: SPHA(base_lr={base_lr}, lambda=0.5), ExpFactor=2.7 (e-Bottleneck)")
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # --- Baseline Step ---
        opt_base.zero_grad()
        logits, feats, _ = model_base(X)
        loss = criterion(logits, Y)
        loss.backward()
        opt_base.step()
        
        # Log Baseline Metrics
        erpm = compute_erpm_proxy(feats.detach())
        history['base']['loss'].append(loss.item())
        history['base']['erpm_rank'].append(erpm['stable_rank'])
        history['base']['erpm_entropy'].append(erpm['spectral_entropy'])
        
        # --- MPI + SPHA Step ---
        opt_mpi.zero_grad()
        logits_mpi, feats_mpi, _ = model_mpi(X)
        loss_mpi = criterion(logits_mpi, Y)
        loss_mpi.backward()
        
        # SPHA Update
        stress, lr = scheduler_spha.step(model_mpi)
        opt_mpi.step() # Step with adaptive LR
        
        # Log MPI Metrics
        erpm_mpi = compute_erpm_proxy(feats_mpi.detach())
        history['mpi']['loss'].append(loss_mpi.item())
        history['mpi']['erpm_rank'].append(erpm_mpi['stable_rank'])
        history['mpi']['erpm_entropy'].append(erpm_mpi['spectral_entropy'])
        history['mpi']['stress'].append(stress)
        history['mpi']['lr'].append(lr)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Base Loss {loss.item():.4f} | MPI Loss {loss_mpi.item():.4f} | SPHA LR {lr:.4f}")

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Loss
    axes[0,0].plot(history['base']['loss'], label='Baseline (Const LR)')
    axes[0,0].plot(history['mpi']['loss'], label='MPI + SPHA')
    axes[0,0].set_title("Training Loss")
    axes[0,0].legend()
    
    # 2. ERPM (Spectral Entropy)
    axes[0,1].plot(history['base']['erpm_entropy'], label='Baseline')
    axes[0,1].plot(history['mpi']['erpm_entropy'], label='MPI + SPHA')
    axes[0,1].set_title("ERPM: Spectral Entropy (Higher = Better Info Preservation)")
    axes[0,1].legend()
    
    # 3. SPHA Dynamics
    ax3 = axes[1,0]
    ax3.plot(history['mpi']['stress'], color='red', label='Stress (Grad Norm)')
    ax3.set_ylabel('Stress', color='red')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(history['mpi']['lr'], color='blue', label='SPHA Learning Rate')
    ax3_twin.set_ylabel('Learning Rate', color='blue')
    ax3.set_title("SPHA Dynamics: Stress vs LR")
    
    # 4. Final ERPM Rank
    axes[1,1].bar(['Baseline', 'MPI+SPHA'], 
                  [history['base']['erpm_rank'][-1], history['mpi']['erpm_rank'][-1]],
                  color=['gray', 'green'])
    axes[1,1].set_title("Final Stable Rank (Effective Dim)")
    
    plt.tight_layout()
    plt.savefig('docs/images/spha_training_experiment.png')
    print("Saved plot to docs/images/spha_training_experiment.png")

if __name__ == "__main__":
    run_training_experiment()
