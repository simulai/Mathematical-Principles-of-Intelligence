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
# 0. Global Constants
# ==========================================
EMBED_DIM = 64
DT = 0.2
GAMMA = 0.75
PHASE_LIMIT = np.pi / 10

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
        # 如果 q 的维度比 embed_dim 大 (比如拼接了 q, p)，需要截断或者重新映射
        # 这里假设 q 已经是 projector 后的 [batch, seq, dim]
        if q.shape[-1] != self.embed_dim:
             # Fallback or error handling if dimensions mismatch
             # For this implementation, we assume q matches embed_dim
             pass
        z = torch.complex(q, p0)
        return z

class EnergyNormalization(nn.Module):
    """
    能量归一化：基于系统总能量 |z|^2 进行缩放
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
    这会产生两个稳定的低能态（对应二分类的两个类别），中间有一个势垒。
    """
    def __init__(self, embed_dim, dt=0.1):
        super().__init__()
        self.dt = dt
        # alpha 控制势垒高度 (负的二次项系数)
        self.alpha = nn.Parameter(torch.tensor(1.0)) 
        # beta 控制束缚强度 (四次项系数)
        self.beta = nn.Parameter(torch.tensor(0.5))
        
        # 耦合矩阵：允许不同维度之间交换能量
        self.coupling = nn.Linear(embed_dim, embed_dim, bias=False)
        # 初始化为接近单位阵，保持维度独立性作为起点
        with torch.no_grad():
            self.coupling.weight.copy_(torch.eye(embed_dim) * 0.1)

    def potential_gradient(self, q):
        # V(q) = -alpha * q^2 + beta * q^4
        # dV/dq = -2*alpha*q + 4*beta*q^3
        # 加上耦合项：coupling(q)
        
        # 限制参数为正，保证物理意义
        alpha = F.softplus(self.alpha)
        beta = F.softplus(self.beta)
        
        grad_V = -2 * alpha * q + 4 * beta * q**3
        grad_V = grad_V + self.coupling(q)
        return grad_V

    def forward(self, z):
        q, p = z.real, z.imag
        
        # 辛几何积分 (Symplectic Euler / Leapfrog)
        # 1. 半步更新动量
        grad_V = self.potential_gradient(q)
        p_half = p - (self.dt / 2) * grad_V
        
        # 2. 全步更新位置 (假设动能 T = p^2/2 => dq/dt = p)
        q_next = q + self.dt * p_half
        
        # 3. 半步更新动量
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
        
        # 2. 增强物理组件 (激进版：双稳态系统)
        self.physical_pe = PhysicalPositionalEncoding(hidden_dim)
        # 替换为双稳态哈密顿量
        self.hamiltonian = BistableHamiltonian(hidden_dim, dt=0.1)
        self.energy_norm = EnergyNormalization() # 依然需要 Norm 防止四次项爆炸
        self.sars = PhaseLockedSARS(hidden_dim)
        
        # 3. 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input_ids, attention_mask):
        # 1. 获取BERT语义特征
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state 
        
        # 2. 投影到物理空间 q
        q = self.projector(sequence_output)
        
        # 3. 注入物理位置动量 -> 形成复数 z
        z = self.physical_pe(q)
        
        # 4. 物理演化 (Evolution)
        # 4.1 双稳态势能场演化 (Chaos & Bistability)
        z = self.hamiltonian(z)
        
        # 4.2 能量归一化 (硬约束)
        z = self.energy_norm(z)
        
        # 4.3 SARS 相位锁定
        z, psi = self.sars(z)
        
        # 5. 坍缩与分类
        collapsed_state = torch.cat([z.real.mean(dim=1), z.imag.mean(dim=1)], dim=-1) 
        
        logits = self.classifier(collapsed_state)
        
        # 计算平均能量用于观察
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

def run_kaggle_training():
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {DEVICE}")
    
    # Auto-switch for CPU
    if DEVICE.type == 'cpu':
        print("!!! CPU Detected: Switching to TinyBERT for speed !!!")
        MODEL_NAME = "prajjwal1/bert-tiny"
    else:
        MODEL_NAME = "distilbert-base-uncased"
        
    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 2e-4 if "tiny" in MODEL_NAME else 2e-5 # TinyBERT needs larger LR
    
    # Path Detection
    possible_dirs = [
        "/kaggle/input/nlp-getting-started",
        "../input/nlp-getting-started",
        "/kaggle/working/data/nlp-getting-started",
        "data/nlp-getting-started",
        r"d:\code\MPI\data\nlp-getting-started",
        r"d:\code\MPI\data\nlp-getting-started\nlp-getting-started", 
        r"d:\code\MPI\competitions\Kaggle\Disaster_Tweets\data\nlp-getting-started",
    ]
    
    DATA_DIR = None
    print("Searching for dataset...")
    for d in possible_dirs:
        check_path = os.path.join(d, "train.csv")
        exists = os.path.exists(check_path)
        print(f"  Checking: {check_path} -> {'Found' if exists else 'Not Found'}")
        if exists:
            DATA_DIR = d
            break
            
    if DATA_DIR is None:
        print("\nError: Dataset not found in any known location!")
        sys.exit(1)

    print(f"\nUsing Data Directory: {DATA_DIR}")
    
    # Load Data
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    
    print(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 5-Fold CV
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = np.zeros(len(test_df))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df['target'])):
        print(f"\n=== Fold {fold+1}/5 ===")
        
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        train_ds = TweetDataset(train_fold, tokenizer)
        val_ds = TweetDataset(val_fold, tokenizer)
        
        # Explicitly set num_workers to 0
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
        
        model = ECHT_BERT_Hybrid(model_name=MODEL_NAME).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_acc = 0
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                optimizer.zero_grad()
                preds, psi, avg_energy = model(input_ids, mask)
                
                # 主损失：BCEWithLogitsLoss
                loss_main = criterion(preds, labels)
                
                # 辅助损失1：PSI 正则化 (鼓励全局一致性)
                loss_psi = 0.1 * (1.0 - psi.mean())
                
                # 辅助损失2：能量正则化 (已恢复硬约束，此处移除 Soft Constraint)
                # loss_energy = 0.05 * torch.abs(avg_energy - 1.0)
                
                loss = loss_main + loss_psi # + loss_energy
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            
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
                torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")
            
            print(f"  Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} {'(New Best)' if val_acc == best_val_acc else ''}")
            
        print(f"Fold {fold+1} Best Acc: {best_val_acc:.4f}")
        
        # Predict on Test Set using BEST model
        print(f"Loading best model for Fold {fold+1}...")
        model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
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
        
    # Generate Submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'target': (test_preds > 0.5).astype(int)
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    run_kaggle_training()
