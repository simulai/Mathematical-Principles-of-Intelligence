
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
from scipy.optimize import minimize_scalar

# ==========================================
# 0. Global Constants & Configuration
# ==========================================
EMBED_DIM = 64
DT = 0.2
GAMMA = 0.75
PHASE_LIMIT = np.pi / 10
BATCH_SIZE = 64  # Faster inference
EPOCHS = 10

# ==========================================
# 1. Shared Physics Engine Components
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

class PhaseLockedSARS(nn.Module):
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

# ==========================================
# 2. Classical Model
# ==========================================

class ECHT_BERT_Hybrid(nn.Module):
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
        self.sars = PhaseLockedSARS(hidden_dim)
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
        z = self.hamiltonian(z)
        z = self.energy_norm(z)
        z, psi = self.sars(z)
        collapsed_state = torch.cat([z.real.mean(dim=1), z.imag.mean(dim=1)], dim=-1) 
        logits = self.classifier(collapsed_state)
        energy = z.real**2 + z.imag**2
        avg_energy = energy.mean()
        return logits.squeeze(), psi, avg_energy

# ==========================================
# 3. Quantum Model
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
        return logit_out, rho, 0.0

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
# 5. Optimization & Main Loop
# ==========================================

def run_optimization():
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
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Storage
    oof_classical = np.zeros(len(train_df))
    oof_quantum = np.zeros(len(train_df))
    test_preds_classical = np.zeros(len(test_df))
    test_preds_quantum = np.zeros(len(test_df))
    
    test_ds = TweetDataset(test_df, tokenizer, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print("\n=== Reconstructing OOF Predictions and Test Inference ===\n")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df['target'])):
        print(f"--- Processing Fold {fold+1}/5 ---")
        
        val_fold = train_df.iloc[val_idx]
        val_ds = TweetDataset(val_fold, tokenizer, is_test=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # 1. Classical Model
        c_path = f"best_model_fold_{fold}.pth"
        if os.path.exists(c_path):
            model_c = ECHT_BERT_Hybrid(model_name=MODEL_NAME).to(DEVICE)
            model_c.load_state_dict(torch.load(c_path, map_location=DEVICE))
            model_c.eval()
            
            # Val
            preds = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    mask = batch['attention_mask'].to(DEVICE)
                    logits, _, _ = model_c(input_ids, mask)
                    preds.extend(torch.sigmoid(logits).cpu().numpy())
            oof_classical[val_idx] = np.array(preds)
            
            # Test
            t_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    mask = batch['attention_mask'].to(DEVICE)
                    logits, _, _ = model_c(input_ids, mask)
                    t_preds.extend(torch.sigmoid(logits).cpu().numpy())
            test_preds_classical += np.array(t_preds) / 5.0
        else:
            print(f"Warning: {c_path} not found.")

        # 2. Quantum Model
        q_path = f"best_model_quantum_fold_{fold}.pth"
        if os.path.exists(q_path):
            model_q = Quantum_ECHT_Hybrid(model_name=MODEL_NAME).to(DEVICE)
            model_q.load_state_dict(torch.load(q_path, map_location=DEVICE))
            model_q.eval()
            
            # Val
            preds = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    mask = batch['attention_mask'].to(DEVICE)
                    logits, _, _ = model_q(input_ids, mask)
                    preds.extend(torch.sigmoid(logits).cpu().numpy())
            oof_quantum[val_idx] = np.array(preds)
            
            # Test
            t_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    mask = batch['attention_mask'].to(DEVICE)
                    logits, _, _ = model_q(input_ids, mask)
                    t_preds.extend(torch.sigmoid(logits).cpu().numpy())
            test_preds_quantum += np.array(t_preds) / 5.0
        else:
            print(f"Warning: {q_path} not found.")

    # Optimization
    print("\n=== Optimizing Ensemble Weights ===")
    y_true = train_df['target'].values
    
    def objective(w):
        # Minimize Log Loss (BCE)
        w = np.clip(w, 0, 1)
        pred = w * oof_classical + (1 - w) * oof_quantum
        # Clip pred to avoid log(0)
        pred = np.clip(pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred))
        return loss
    
    res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    best_w = res.x
    best_loss = res.fun
    
    # Calculate Accuracy at best_w
    final_oof_pred = best_w * oof_classical + (1 - best_w) * oof_quantum
    acc = np.mean((final_oof_pred > 0.5).astype(int) == y_true)
    
    print(f"Best Weight (Classical): {best_w:.4f}")
    print(f"Best Weight (Quantum): {1-best_w:.4f}")
    print(f"OOF LogLoss: {best_loss:.5f}")
    print(f"OOF Accuracy: {acc:.5f}")
    
    # Generate Submission
    final_test_pred = best_w * test_preds_classical + (1 - best_w) * test_preds_quantum
    
    sub = pd.DataFrame({'id': test_df['id'], 'target': (final_test_pred > 0.5).astype(int)})
    sub.to_csv('submission_optimized_ensemble.csv', index=False)
    print(f"\nSaved submission_optimized_ensemble.csv with weight {best_w:.4f}")

if __name__ == "__main__":
    run_optimization()
