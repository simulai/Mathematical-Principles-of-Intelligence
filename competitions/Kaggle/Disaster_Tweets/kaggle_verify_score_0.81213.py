# ==========================================
# Kaggle Verification Script (Score: 0.81213)
# Architecture: ECHT-BERT-Hybrid (Bistable Hamiltonian + PhaseLockedSARS)
# Date: 2026-01-04
#
# Potential Improvements (To Reach 0.83+):
# 1. SWA (Stochastic Weight Averaging): Average weights from last few epochs.
# 2. Pseudo-labeling: Use high-confidence test predictions as training data.
# 3. Seed Tuning: Try different random seeds (e.g., 42, 1024, 2024) to find better data splits.
# 4. Ensemble: Combine predictions with a different architecture (e.g., RoBERTa).
# ==========================================

import os
import sys
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel

# ==========================================
# 0. Global Constants & Configuration
# ==========================================
# Kaggle verification settings
EMBED_DIM = 64
DT = 0.2
GAMMA = 0.75
PHASE_LIMIT = np.pi / 10
EPOCHS = 10         # Default 10, early stopping via best model checkpointing
BATCH_SIZE = 32     # Adjustable based on GPU memory

# ==========================================
# 1. Physics Engine Components (ECHT-SARS v2 Enhanced)
# ==========================================

class PhysicalPositionalEncoding(nn.Module):
    """
    物理位置编码：将位置信息编码为初始动量
    """
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            (-math.log(10000.0) / embed_dim))
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
    """
    能量归一化：基于系统总能量 |z|^2 进行缩放 (Hard Constraint)
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        energy = torch.sum(z.real**2 + z.imag**2, dim=-1, keepdim=True) + self.eps
        scale = torch.rsqrt(energy)
        return torch.complex(z.real * scale, z.imag * scale)

class BistableHamiltonian(nn.Module):
    """
    双稳态哈密顿量：模拟 Duffing Oscillator (双井势能)
    势能函数 V(q) = -alpha * q^2 + beta * q^4
    """
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
        # Symplectic Euler / Leapfrog
        grad_V = self.potential_gradient(q)
        p_half = p - (self.dt / 2) * grad_V
        q_next = q + self.dt * p_half
        grad_V_next = self.potential_gradient(q_next)
        p_next = p_half - (self.dt / 2) * grad_V_next
        return torch.complex(q_next, p_next)

class PhaseLockedSARS(nn.Module):
    """
    相位锁定机制：SARS (Synchronization-Aware Re-calibration System)
    """
    def __init__(self, dim):
        super().__init__()
        self.phase_tuner = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim)
        )

    def forward(self, z):
        q = z.real
        phase_orig = z.angle()
        psi = torch.abs(torch.exp(1j * phase_orig).mean(dim=1)).mean(dim=-1)
        
        phi_corr = torch.tanh(self.phase_tuner(q.mean(dim=1))) * PHASE_LIMIT
        phi_corr = phi_corr.unsqueeze(1)
        z_tuned = z * torch.exp(1j * phi_corr)
        return z_tuned, psi

class ECHT_BERT_Hybrid(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", hidden_dim=64):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size
        
        # 1. 语义投影
        self.projector = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.hidden_dim = hidden_dim
        
        # 2. 物理组件
        self.physical_pe = PhysicalPositionalEncoding(hidden_dim)
        self.hamiltonian = BistableHamiltonian(hidden_dim, dt=DT)
        self.energy_norm = EnergyNormalization()
        self.sars = PhaseLockedSARS(hidden_dim)
        
        # 3. 分类器 (拼接实部和虚部)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state 
        q = self.projector(sequence_output)
        z = self.physical_pe(q)
        
        # Evolution
        z = self.hamiltonian(z)
        z = self.energy_norm(z)
        z, psi = self.sars(z)
        
        # Collapse & Classify
        collapsed_state = torch.cat([z.real.mean(dim=1), z.imag.mean(dim=1)], dim=-1) 
        logits = self.classifier(collapsed_state)
        
        energy = z.real**2 + z.imag**2
        avg_energy = energy.mean()
        
        return logits.squeeze(), psi, avg_energy

# ==========================================
# 2. Data Pipeline
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
# 3. Main Execution Logic
# ==========================================

def run_kaggle_verification():
    # Setup Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {DEVICE}")
    
    # Model Config
    if DEVICE.type == 'cpu':
        print("!!! CPU Detected: Switching to TinyBERT for speed !!!")
        MODEL_NAME = "prajjwal1/bert-tiny"
    else:
        MODEL_NAME = "distilbert-base-uncased" # Or "bert-base-uncased" if GPU allows
        
    # Dataset Path Logic
    possible_dirs = [
        "/kaggle/input/nlp-getting-started",       # Standard Kaggle
        "/kaggle/input/npl-disaster-tweets",       # Your current dataset name
        "../input/nlp-getting-started",            
        "data/nlp-getting-started",                
        r"d:\code\MPI\data\nlp-getting-started",   
    ]
    
    DATA_DIR = None
    print("Searching for dataset...")
    for d in possible_dirs:
        check_path = os.path.join(d, "train.csv")
        if os.path.exists(check_path):
            DATA_DIR = d
            print(f"Found dataset at: {DATA_DIR}")
            break
            
    if DATA_DIR is None:
        print("Error: Dataset not found.")
        return

    # Load Data
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Prepare Training
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = np.zeros(len(test_df))
    
    # Training Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df['target'])):
        print(f"\n=== Fold {fold+1}/5 ===")
        
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        train_ds = TweetDataset(train_fold, tokenizer)
        val_ds = TweetDataset(val_fold, tokenizer)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
        
        print("Initializing model (downloading if necessary)...")
        model = ECHT_BERT_Hybrid(model_name=MODEL_NAME).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=2e-5 if "tiny" not in MODEL_NAME else 2e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_acc = 0
        best_model_path = f"best_model_fold_{fold}.pth"
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            
            # Add progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
            
            batch_idx = 0
            for batch in pbar:
                batch_idx += 1
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                optimizer.zero_grad()
                preds, psi, _ = model(input_ids, mask)
                
                loss = criterion(preds, labels) + 0.1 * (1.0 - psi.mean())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # Explicit print every 50 batches (for environments where tqdm fails)
                if batch_idx % 50 == 0:
                    print(f"  [Batch {batch_idx}] Loss: {loss.item():.4f}")
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    mask = batch['attention_mask'].to(DEVICE)
                    labels = batch['label'].to(DEVICE)
                    preds, _, _ = model(input_ids, mask)
                    predicted = (torch.sigmoid(preds) > 0.5).float()
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            
            val_acc = correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
            
            print(f"  Epoch {epoch+1}: Loss {train_loss/len(train_loader):.4f} | Val Acc {val_acc:.4f}")
            
        print(f"Fold {fold+1} Best Acc: {best_val_acc:.4f}")
        
        # Inference
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_ds = TweetDataset(test_df, tokenizer, is_test=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)
        
        fold_preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                preds, _, _ = model(input_ids, mask)
                fold_preds.extend(torch.sigmoid(preds).cpu().numpy())
        
        test_preds += np.array(fold_preds) / 5.0

    # Submission
    submission = pd.DataFrame({'id': test_df['id'], 'target': (test_preds > 0.5).astype(int)})
    submission.to_csv('submission.csv', index=False)
    print("Saved submission.csv")

if __name__ == "__main__":
    run_kaggle_verification()
