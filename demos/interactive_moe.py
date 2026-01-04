import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import os
import time

# Ensure we can import from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-define classes here for standalone usage to avoid import issues if user runs from different paths
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
            # Sinkhorn Routing
            router_logits = router_logits / self.temperature
            M = torch.exp(router_logits)
            for _ in range(3):
                M = M / (M.sum(dim=1, keepdim=True) + 1e-6)
                M = M / (M.sum(dim=0, keepdim=True) + 1e-6)
            routing_weights = M
            top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
        else:
            # Baseline Softmax
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
                
        return final_output, top_k_indices, routing_weights

class MoEClassifier(nn.Module):
    def __init__(self, dim, num_experts, num_classes, mode='Baseline', expansion_factor=4.0):
        super().__init__()
        self.moe = MoELayer(dim, num_experts, k=2, mode=mode, expansion_factor=expansion_factor)
        self.classifier = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        features, indices, weights = self.moe(x)
        logits = self.classifier(features)
        return logits, indices, weights

def train_quick_models():
    print("🚀 Initializing & Training Models (Quick POC)...")
    dim = 64
    num_experts = 8
    num_classes = 2
    
    # 1. Baseline (Big & Dumb)
    model_base = MoEClassifier(dim, num_experts, num_classes, mode='Baseline', expansion_factor=4.0)
    opt_base = optim.SGD(model_base.parameters(), lr=0.01)
    
    # 2. MPI (Small & Smart)
    model_mpi = MoEClassifier(dim, num_experts, num_classes, mode='MPI', expansion_factor=2.7)
    opt_mpi = optim.SGD(model_mpi.parameters(), lr=0.1) # Aggressive LR for MPI
    
    # Synthetic Data
    X = torch.randn(200, dim)
    Y = torch.randint(0, 2, (200,))
    
    criterion = nn.CrossEntropyLoss()
    
    # Train for 20 epochs
    for epoch in range(20):
        # Base
        opt_base.zero_grad()
        out, _, _ = model_base(X)
        loss_base = criterion(out, Y)
        loss_base.backward()
        opt_base.step()
        
        # MPI
        opt_mpi.zero_grad()
        out_mpi, _, _ = model_mpi(X)
        loss_mpi = criterion(out_mpi, Y)
        loss_mpi.backward()
        opt_mpi.step()
        
        if epoch % 5 == 0:
            print(f"   Epoch {epoch}: Base Loss {loss_base.item():.3f} | MPI Loss {loss_mpi.item():.3f}")
            
    print("✅ Training Complete!\n")
    return model_base, model_mpi

def interactive_demo():
    model_base, model_mpi = train_quick_models()
    model_base.eval()
    model_mpi.eval()
    
    print("="*60)
    print("🤖 MPI MoE Interactive Demo")
    print("Compare 'Baseline' (Standard MoE) vs 'MPI' (Sinkhorn MoE)")
    print("="*60)
    print("Press ENTER to feed a new random signal (Batch of 32) to the models.")
    print("Type 'q' and ENTER to quit.")
    print("-" * 60)
    
    # Auto-run for 5 steps for verification
    for i in range(5):
        print(f"\n[Test Step {i+1}/5]")
        user_input = "" # Auto-enter
            
        # Generate random signal BATCH
        batch_size = 32
        x = torch.randn(batch_size, 64)
        
        # Inference
        with torch.no_grad():
            logits_base, idx_base, weights_base = model_base(x)
            logits_mpi, idx_mpi, weights_mpi = model_mpi(x)
            
            # --- Analysis ---
            
            # 1. Token Confidence (Average Entropy per Token)
            # We want LOW entropy per token (Confident selection)
            token_entropy_base = -torch.sum(weights_base * torch.log(weights_base + 1e-10), dim=1).mean().item()
            token_entropy_mpi = -torch.sum(weights_mpi * torch.log(weights_mpi + 1e-10), dim=1).mean().item()
            
            # 2. Expert Load Balancing (Entropy of Batch-wise Expert Usage)
            # We want HIGH entropy across the batch (All experts used)
            expert_usage_base = idx_base.view(-1).float().histc(bins=8, min=0, max=7)
            expert_probs_base = expert_usage_base / expert_usage_base.sum()
            batch_entropy_base = -torch.sum(expert_probs_base * torch.log(expert_probs_base + 1e-10)).item()
            
            expert_usage_mpi = idx_mpi.view(-1).float().histc(bins=8, min=0, max=7)
            expert_probs_mpi = expert_usage_mpi / expert_usage_mpi.sum()
            batch_entropy_mpi = -torch.sum(expert_probs_mpi * torch.log(expert_probs_mpi + 1e-10)).item()
            
            # Prediction (Just take first sample for display)
            pred_base = torch.argmax(logits_base[0]).item()
            pred_mpi = torch.argmax(logits_mpi[0]).item()
            
        print(f"\n🧪 Batch Processed (Size {batch_size}):")
        print(f"   [Baseline Model (Exp=4.0)]")
        print(f"     -> Token Confidence (Entropy): {token_entropy_base:.4f} (Lower is better)")
        print(f"     -> Load Balance (Entropy):     {batch_entropy_base:.4f} (Higher is better)")
        print(f"     -> Experts Used (Histogram):   {expert_usage_base.int().tolist()}")
        
        print(f"   [MPI Model (Exp=2.7)]")
        print(f"     -> Token Confidence (Entropy): {token_entropy_mpi:.4f} (Lower is better)")
        print(f"     -> Load Balance (Entropy):     {batch_entropy_mpi:.4f} (Higher is better)")
        print(f"     -> Experts Used (Histogram):   {expert_usage_mpi.int().tolist()}")
        
        if batch_entropy_mpi > batch_entropy_base:
            print(f"   ✨ MPI Win: Better Load Balancing (Cognitive Holonomy)!")
        elif token_entropy_mpi < token_entropy_base:
             print(f"   ✨ MPI Win: Higher Confidence!")
        else:
            print(f"   ⚠️ Tie or Baseline Win.")

if __name__ == "__main__":
    interactive_demo()
