# ==========================================
# Kaggle Pseudo-Labeling Script
# Strategy: Use high-confidence ensemble predictions to augment training data
# Architecture: Retrain Quantum-ECHT on Train + Pseudo-Test
# ==========================================

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import math

# ==========================================
# 0. Global Constants & Configuration
# ==========================================
EMBED_DIM = 64
DT = 0.2
EPOCHS = 5 # Retrain for fewer epochs on larger data
BATCH_SIZE = 32
CONFIDENCE_THRESHOLD = 0.95 # Only take samples with >95% confidence

# ==========================================
# 1. Quantum Physics Components (Same as before)
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

class EnergyNormalization(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        energy = torch.sum(z.real**2 + z.imag**2, dim=-1, keepdim=True) + self.eps
        scale = torch.rsqrt(energy)
        return torch.complex(z.real * scale, z.imag * scale)

# ==========================================
# 2. Hybrid Architecture: Quantum-ECHT
# ==========================================
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
        return logit_out, rho, 0.0

class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df['text'].astype(str).tolist()
        if not is_test:
            self.labels = df['target'].tolist()
        elif 'target' in df.columns: # Support pseudo-labels in test
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
        if hasattr(self, 'labels'):
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# ==========================================
# 3. Main Pseudo-Labeling Pipeline
# ==========================================
def run_pseudo_labeling():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {DEVICE}")
    
    if DEVICE.type == 'cpu':
        print("!!! CPU Detected: Switching to TinyBERT for speed !!!")
        MODEL_NAME = "prajjwal1/bert-tiny"
    else:
        MODEL_NAME = "distilbert-base-uncased"
        
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
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 1. Get Ensemble Probabilities (Re-run inference logic briefly)
    print("\n=== Generating Probabilities for Pseudo-Labeling ===")
    test_ds = TweetDataset(test_df, tokenizer, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Accumulate predictions from all 5 Quantum Folds (using Quantum as base as it's our focus)
    # Ideally use ensemble, but we need to re-instantiate both classes.
    # For simplicity and consistency with `kaggle_quantum_submission.py`, we use the Quantum models.
    # The ensemble was 0.95 correlated anyway.
    
    all_probs = np.zeros((len(test_df), 5))
    
    for fold in range(5):
        model_path = f"best_model_quantum_fold_{fold}.pth"
        if not os.path.exists(model_path):
            print(f"Skipping fold {fold}, model not found.")
            continue
            
        model = Quantum_ECHT_Hybrid(model_name=MODEL_NAME).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        fold_p = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Fold {fold+1}"):
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                logits, _, _ = model(input_ids, mask)
                probs = torch.sigmoid(logits)
                fold_p.extend(probs.cpu().numpy())
        all_probs[:, fold] = fold_p
        
    avg_probs = np.mean(all_probs, axis=1)
    
    # 2. Select High Confidence Samples
    high_conf_indices = np.where((avg_probs > CONFIDENCE_THRESHOLD) | (avg_probs < (1 - CONFIDENCE_THRESHOLD)))[0]
    print(f"\nFound {len(high_conf_indices)} high-confidence samples (Threshold: {CONFIDENCE_THRESHOLD})")
    
    pseudo_test_df = test_df.iloc[high_conf_indices].copy()
    pseudo_test_df['target'] = (avg_probs[high_conf_indices] > 0.5).astype(int)
    
    # 3. Combine Data
    print("Augmenting Training Data...")
    print(f"Original Train Size: {len(train_df)}")
    print(f"Pseudo-Labeled Size: {len(pseudo_test_df)}")
    
    # Re-initialize dataset with combined data
    full_train_ds = TweetDataset(train_df, tokenizer, is_test=False)
    pseudo_ds = TweetDataset(pseudo_test_df, tokenizer, is_test=False) # Now has 'target'
    combined_ds = ConcatDataset([full_train_ds, pseudo_ds])
    
    train_loader = DataLoader(combined_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Retrain a SINGLE robust model on ALL data (for submission)
    # Or retrain 5 folds? Retraining 5 folds takes time.
    # Let's train ONE production model on all data + pseudo data.
    print("\n=== Retraining Final Model on Augmented Data ===")
    
    final_model = Quantum_ECHT_Hybrid(model_name=MODEL_NAME).to(DEVICE)
    optimizer = optim.AdamW(final_model.parameters(), lr=2e-5 if "tiny" not in MODEL_NAME else 2e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    final_model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for batch in pbar:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            logits, _, _ = final_model(input_ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    # Save Final Model
    torch.save(final_model.state_dict(), "final_quantum_pseudo_model.pth")
    print("Final Pseudo-Label Model Saved.")
    
    # 5. Generate Submission
    print("\n=== Generating Final Submission ===")
    final_model.eval()
    test_ds = TweetDataset(test_df, tokenizer, is_test=True) # Full test set
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    final_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            logits, _, _ = final_model(input_ids, mask)
            probs = torch.sigmoid(logits)
            final_preds.extend(probs.cpu().numpy())
            
    submission = pd.DataFrame({
        'id': test_df['id'],
        'target': (np.array(final_preds) > 0.5).astype(int)
    })
    
    submission.to_csv('submission_pseudo_quantum.csv', index=False)
    print("Submission saved to submission_pseudo_quantum.csv")

if __name__ == "__main__":
    run_pseudo_labeling()
