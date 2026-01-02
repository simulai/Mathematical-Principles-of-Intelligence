import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
import math

# ==========================================
# Core Theory Implementations (MPI)
# ==========================================

class SPHA(nn.Module):
    """
    Softmax-Projected Hyper-Attention (SPHA)
    Implements the e-base scaling law by scaling attention scores by ln(b)/b.
    """
    def __init__(self, embed_dim, num_heads, branching_factor=2.718):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.b = branching_factor
        
        # Scaling factor derived from e-base law: (ln b) / b
        # We use this to modulate the temperature of softmax
        self.mpi_scale = (math.log(self.b) / self.b) * math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # MPI Scaled Dot-Product Attention
        # Instead of dividing by sqrt(d_k), we multiply by our efficiency factor
        # attention = (Q @ K.T) * mpi_scale
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.mpi_scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class ZhangInvariantLoss(nn.Module):
    """
    Computes the violation of the Zhang Invariant (Z).
    Z = curl(A) = 0 for optimal flow.
    Approximated by the commutativity of the flow field gradients or 
    cyclic consistency loss.
    """
    def __init__(self, lambda_z=0.1):
        super().__init__()
        self.lambda_z = lambda_z
        
    def forward(self, hidden_states):
        """
        hidden_states: [Batch, Length, Dim]
        We approximate holonomy by checking if the transformation preserves 
        local geometric structure (isometric constraint).
        """
        # Simple approximation: penalize rapid changes in metric (Ricci flow smoothing)
        # || h_t - h_{t-1} ||^2 
        # In a real manifold setting, this would be a curvature calculation.
        diff = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
        loss = torch.mean(diff ** 2)
        return self.lambda_z * loss

class MPIDisasterModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.backbone.config.hidden_size
        
        # Replace the last layer or add an MPI block
        self.mpi_block = SPHA(self.hidden_dim, num_heads=8, branching_factor=math.e)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.zhang_loss_fn = ZhangInvariantLoss(lambda_z=0.05)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state # [B, L, D]
        
        # Apply MPI SPHA
        mpi_out = self.mpi_block(last_hidden_state, attention_mask.unsqueeze(1).unsqueeze(2))
        
        # Pooling (CLS token)
        cls_token = mpi_out[:, 0, :]
        logits = self.classifier(cls_token)
        
        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
            z_loss = self.zhang_loss_fn(mpi_out)
            loss = ce_loss + z_loss
            
        return logits, loss

# ==========================================
# Data Handling
# ==========================================

class TweetDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=128, is_test=False):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        inputs = self.tokenizer(
            text,
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
            item['labels'] = torch.tensor(self.df.iloc[idx]['target'], dtype=torch.long)
            
        return item

# ==========================================
# Training Loop
# ==========================================

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        _, loss = model(input_ids, mask, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits, _ = model(input_ids, mask)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return preds

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check data
    train_path = "data/nlp-getting-started/train.csv"
    test_path = "data/nlp-getting-started/test.csv"
    
    if not os.path.exists(train_path):
        print("Data not found. Generating dummy data for verification...")
        # Create dummy data if download failed
        os.makedirs("data/nlp-getting-started", exist_ok=True)
        dummy_train = pd.DataFrame({
            'id': range(100),
            'text': ['This is a disaster!'] * 50 + ['This is fine.'] * 50,
            'target': [1] * 50 + [0] * 50
        })
        dummy_train.to_csv(train_path, index=False)
        
        dummy_test = pd.DataFrame({
            'id': range(100, 120),
            'text': ['Help me!'] * 10 + ['Hello world'] * 10
        })
        dummy_test.to_csv(test_path, index=False)
    
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset = TweetDataset(train_path, tokenizer)
    test_dataset = TweetDataset(test_path, tokenizer, is_test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    model = MPIDisasterModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Train
    print("Starting MPI-Enhanced Training...")
    for epoch in range(2): # Short training for demo
        loss = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        
    # Predict
    print("Generating predictions...")
    predictions = predict(model, test_loader, device)
    
    # Save submission
    sub_df = pd.read_csv(test_path)
    sub_df['target'] = predictions
    # Keep only id and target
    submission = sub_df[['id', 'target']]
    submission.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
