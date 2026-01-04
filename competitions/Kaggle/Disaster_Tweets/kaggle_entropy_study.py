
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import math
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# ==========================================
# 0. Global Constants & Configuration
# ==========================================
EMBED_DIM = 64
DT = 0.2
GAMMA = 0.75
PHASE_LIMIT = np.pi / 10
BATCH_SIZE = 32
EPOCHS = 3 # Short run to observe entropy dynamics

# ==========================================
# 1. Physics Engine Components (Shared)
# ==========================================

class PhysicalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, q, positions=None):
        batch_size, seq_len, _ = q.shape
        if positions is None:
            positions = torch.arange(seq_len, device=q.device).expand(batch_size, seq_len)
        p0 = self.pe[positions]
        z = torch.complex(q, p0)
        return z

class EnergyNormalization(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        energy = torch.sum(z.real**2 + z.imag**2, dim=-1, keepdim=True) + self.eps
        scale = torch.rsqrt(energy)
        return torch.complex(z.real * scale, z.imag * scale)

class BistableHamiltonian(nn.Module):
    def __init__(self, embed_dim, dt=0.1):
        super().__init__()
        self.dt = dt
        self.alpha = nn.Parameter(torch.tensor(1.0)) 
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.coupling = nn.Linear(embed_dim, embed_dim, bias=False)
        with torch.no_grad():
            self.coupling.weight.copy_(torch.eye(embed_dim) * 0.1)

    def potential_gradient(self, q):
        alpha = F.softplus(self.alpha)
        beta = F.softplus(self.beta)
        grad_V = -2 * alpha * q + 4 * beta * q**3
        grad_V = grad_V + self.coupling(q)
        return grad_V

    def forward(self, z):
        q, p = z.real, z.imag
        grad_V = self.potential_gradient(q)
        p_half = p - (self.dt / 2) * grad_V
        q_next = q + self.dt * p_half
        grad_V_next = self.potential_gradient(q_next)
        p_next = p_half - (self.dt / 2) * grad_V_next
        return torch.complex(q_next, p_next)

# ==========================================
# 2. Quantum Physics Engine Components
# ==========================================

class DensityMatrixEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.mix_param = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        v = F.normalize(x, p=2, dim=-1)
        rho_pure = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))
        epsilon = torch.sigmoid(self.mix_param)
        identity = torch.eye(self.embed_dim, device=x.device).unsqueeze(0).expand_as(rho_pure)
        rho = (1 - epsilon) * rho_pure + epsilon * (identity / self.embed_dim)
        return rho

class LindbladEvolution(nn.Module):
    def __init__(self, embed_dim, num_jump_ops=2, dt=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dt = dt
        self.H_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.jump_ops = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.5)
            for _ in range(num_jump_ops)
        ])

    def get_hamiltonian(self):
        H = self.H_params
        return (H + H.t()) / 2

    def forward(self, rho):
        H = self.get_hamiltonian()
        H_expanded = H.unsqueeze(0).expand_as(rho)
        commutator = torch.bmm(H_expanded, rho) - torch.bmm(rho, H_expanded)
        d_rho_unitary = -1j * commutator 
        d_rho_dissipative = torch.zeros_like(rho)
        for L in self.jump_ops:
            L_exp = L.unsqueeze(0).expand_as(rho)
            L_dag = L.t().conj().unsqueeze(0).expand_as(rho)
            term1 = torch.bmm(L_exp, torch.bmm(rho, L_dag))
            L_dag_L = torch.matmul(L.t().conj(), L).unsqueeze(0).expand_as(rho)
            term2 = 0.5 * (torch.bmm(L_dag_L, rho) + torch.bmm(rho, L_dag_L))
            d_rho_dissipative += (term1 - term2)
        return rho + (d_rho_unitary + d_rho_dissipative) * self.dt

class QuantumMeasurement(nn.Module):
    def __init__(self, embed_dim, num_classes=1):
        super().__init__()
        self.measure_vectors = nn.Parameter(torch.randn(num_classes, embed_dim))

    def forward(self, rho):
        M = self.measure_vectors
        M = F.normalize(M, p=2, dim=-1)
        u = M.unsqueeze(0).expand(rho.size(0), -1, -1)
        u_T = u.transpose(1, 2).type_as(rho)
        inner = torch.bmm(rho, u_T)
        u_complex = u.type_as(rho)
        expectation = torch.bmm(u_complex, inner)
        return expectation.squeeze().real

class Quantum_ECHT_Hybrid(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", hidden_dim=64):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.hidden_dim = hidden_dim
        self.physical_pe = PhysicalPositionalEncoding(hidden_dim)
        self.hamiltonian = BistableHamiltonian(hidden_dim, dt=DT)
        self.energy_norm = EnergyNormalization()
        self.dropout = nn.Dropout(0.3)
        self.density_embedding = DensityMatrixEmbedding(hidden_dim)
        self.lindblad = LindbladEvolution(hidden_dim, num_jump_ops=2, dt=0.1)
        self.measurement = QuantumMeasurement(hidden_dim, num_classes=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state 
        q = self.projector(sequence_output)
        z = self.physical_pe(q)
        z = self.hamiltonian(z)
        z = self.energy_norm(z)
        z_pooled = z.mean(dim=1) 
        x_classical = z_pooled.real
        x_classical = self.dropout(x_classical)
        rho = self.density_embedding(x_classical)
        rho = self.lindblad(rho)
        logits = self.measurement(rho)
        prob = torch.clamp(logits, 1e-6, 1.0 - 1e-6)
        logit_out = torch.log(prob / (1 - prob))
        return logit_out, rho

# ==========================================
# 3. Entropy Analysis Tools
# ==========================================

def calc_von_neumann_entropy(rho):
    """
    Calculates Von Neumann Entropy: S(rho) = -Tr(rho * ln(rho))
    Using eigenvalues: S = - sum(lambda * ln(lambda))
    """
    # Ensure rho is Hermitian
    # Although construction guarantees it, numerical errors might exist
    # rho shape: [batch, dim, dim] (complex)
    
    # Eigenvalues of Hermitian matrix are real
    try:
        L = torch.linalg.eigvalsh(rho.real) # Use real part approximation if complex is tricky, but rho is complex Hermitian
        # Ideally: L = torch.linalg.eigvalsh(rho)
        # But PyTorch eigvalsh supports complex inputs since recent versions.
        # Let's try complex first.
    except:
        # Fallback to real part if complex eigvalsh not supported on this device/version
        L = torch.linalg.eigvalsh(rho.real)

    # Filter out small negative values due to numerical noise
    L = torch.clamp(L, min=1e-10)
    
    # S = - sum(p * log(p))
    entropy = -torch.sum(L * torch.log(L), dim=-1)
    return entropy

# ==========================================
# 4. Data Logic
# ==========================================

class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df['text'].astype(str).tolist()
        if not is_test:
            self.labels = df['target'].tolist()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze()
        }
        if not self.is_test:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# ==========================================
# 5. Training with Entropy Monitoring
# ==========================================

def run_entropy_study():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {DEVICE}")
    
    # Use TinyBERT for speed
    MODEL_NAME = "prajjwal1/bert-tiny" 
        
    possible_dirs = [
        "/kaggle/input/nlp-getting-started",
        "/kaggle/input/npl-disaster-tweets",
        "../input/nlp-getting-started",
        "data/nlp-getting-started",
        r"d:\code\MPI\data\nlp-getting-started",
    ]
    
    DATA_DIR = None
    for d in possible_dirs:
        check_path = os.path.join(d, "train.csv")
        if os.path.exists(check_path):
            DATA_DIR = d
            print(f"Found dataset at: {DATA_DIR}")
            break
            
    if DATA_DIR is None:
        print("Error: Dataset not found.")
        return

    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    
    # Split Train/Val (Single Fold for Study)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(kf.split(train_df, train_df['target']))
    
    train_fold = train_df.iloc[train_idx]
    val_fold = train_df.iloc[val_idx]
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TweetDataset(train_fold, tokenizer)
    val_ds = TweetDataset(val_fold, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Quantum_ECHT_Hybrid(model_name=MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4) # Higher LR for TinyBERT
    criterion = nn.BCEWithLogitsLoss()
    
    print("\n=== Starting Entropy Monitoring Training ===\n")
    print(f"Initial Von Neumann Entropy Check...")
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_acc': [],
        'avg_entropy_disaster': [],
        'avg_entropy_normal': [],
        'avg_entropy_correct': [],
        'avg_entropy_wrong': []
    }
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Training Loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            logits, rho = model(input_ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation & Entropy Analysis
        model.eval()
        val_entropies = []
        val_labels = []
        val_preds = []
        correct_count = 0
        total_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                logits, rho = model(input_ids, mask)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                # Calculate Entropy
                S = calc_von_neumann_entropy(rho)
                
                val_entropies.extend(S.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                
                correct_count += (preds == labels).sum().item()
                total_count += labels.size(0)
        
        val_acc = correct_count / total_count
        val_entropies = np.array(val_entropies)
        val_labels = np.array(val_labels)
        val_preds = np.array(val_preds)
        
        # Statistics
        entropy_disaster = val_entropies[val_labels == 1].mean()
        entropy_normal = val_entropies[val_labels == 0].mean()
        
        mask_correct = (val_preds == val_labels)
        entropy_correct = val_entropies[mask_correct].mean()
        entropy_wrong = val_entropies[~mask_correct].mean()
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Avg Entropy (Disaster): {entropy_disaster:.4f}")
        print(f"  Avg Entropy (Normal):   {entropy_normal:.4f}")
        print(f"  Avg Entropy (Correct):  {entropy_correct:.4f}")
        print(f"  Avg Entropy (Wrong):    {entropy_wrong:.4f}")
        print(f"  Entropy Delta (Normal - Disaster): {entropy_normal - entropy_disaster:.4f}")
        
        history['epoch'].append(epoch+1)
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_acc'].append(val_acc)
        history['avg_entropy_disaster'].append(entropy_disaster)
        history['avg_entropy_normal'].append(entropy_normal)
        history['avg_entropy_correct'].append(entropy_correct)
        history['avg_entropy_wrong'].append(entropy_wrong)

    print("\n=== Training Complete ===")
    
    # Save Results
    res_df = pd.DataFrame(history)
    res_df.to_csv("entropy_study_results.csv", index=False)
    print("Results saved to entropy_study_results.csv")

if __name__ == "__main__":
    run_entropy_study()
