# MPI Disaster Tweets Classification - Kaggle Source Code
# Version: 1.1 (DistilBERT + SPHA + ZhangInvariant)
# Copy and paste this ENTIRE code into a single Code Cell in your Kaggle Notebook.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
import math
from tqdm.notebook import tqdm

# Check device
print("MPI Version: 1.1 (DistilBERT + SPHA + ZhangInvariant) - Loaded Successfully")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 1. Core Theory Modules
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
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.mpi_scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class ZhangInvariantLoss(nn.Module):
    def __init__(self, lambda_z=0.1):
        super().__init__()
        self.lambda_z = lambda_z
        
    def forward(self, hidden_states):
        # Penalize rapid metric changes (Ricci flow smoothing)
        diff = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
        loss = torch.mean(diff ** 2)
        return self.lambda_z * loss

# ==========================================
# 2. Model Architecture
# ==========================================

class MPIDisasterModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.backbone.config.hidden_size
        self.mpi_block = SPHA(self.hidden_dim, num_heads=8, branching_factor=math.e)
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.zhang_loss_fn = ZhangInvariantLoss(lambda_z=0.05)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mpi_out = self.mpi_block(last_hidden_state, attention_mask.unsqueeze(1).unsqueeze(2))
        cls_token = mpi_out[:, 0, :]
        logits = self.classifier(cls_token)
        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
            z_loss = self.zhang_loss_fn(mpi_out)
            loss = ce_loss + z_loss
        return logits, loss

# ==========================================
# 3. Data Loading & Training
# ==========================================

class TweetDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=128, is_test=False):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        
        # Robust column detection
        self.text_col = 'text'
        if 'text' not in self.df.columns:
            # Try to find a likely text column
            possible_cols = ['text_clean', 'text_cleaned', 'clean_text', 'tweet', 'content']
            for col in possible_cols:
                if col in self.df.columns:
                    self.text_col = col
                    break
        
        # Ensure text column is string and handle NaNs
        self.df[self.text_col] = self.df[self.text_col].fillna("").astype(str)

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        text = self.df.iloc[idx][self.text_col]
        inputs = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        item = {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze()}
        if not self.is_test: item['labels'] = torch.tensor(self.df.iloc[idx]['target'], dtype=torch.long)
        return item

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc='Training', leave=False)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        _, loss = model(input_ids, mask, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def find_dataset_path():
    """
    Robustly find the dataset path and filenames.
    Returns: (base_path, train_filename, test_filename)
    """
    # 1. Check standard competition paths
    standard_paths = [
        "/kaggle/input/nlp-getting-started",
        "data/nlp-getting-started",
        "nlp-getting-started"
    ]
    
    # Possible filename variations
    train_files = ["train.csv", "train_preprocessed.csv"]
    test_files = ["test.csv", "test_preprocessed.csv"]
    
    for path in standard_paths:
        for t_file in train_files:
            if os.path.exists(f"{path}/{t_file}"):
                # Find corresponding test file
                found_test = "test.csv"
                for test_f in test_files:
                    if os.path.exists(f"{path}/{test_f}"):
                        found_test = test_f
                        break
                return path, t_file, found_test
            
    # 2. Search recursively in /kaggle/input
    search_root = "/kaggle/input"
    if os.path.exists(search_root):
        print(f"Searching for train files in {search_root}...")
        for root, dirs, files in os.walk(search_root):
            for t_file in train_files:
                if t_file in files:
                    print(f"Found {t_file} at: {root}")
                    # Find corresponding test file
                    found_test = "test.csv"
                    for test_f in test_files:
                        if test_f in files:
                            found_test = test_f
                            break
                    return root, t_file, found_test
                
    return None, None, None

def main():
    # Execution Setup
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Check for Kaggle input paths
    base_path, train_file, test_file = find_dataset_path()
            
    if base_path is None:
        print("Dataset not found in standard paths or /kaggle/input.")
        
        # Debug: List contents of /kaggle/input if it exists
        if os.path.exists("/kaggle/input"):
            print("Contents of /kaggle/input:")
            for root, dirs, files in os.walk("/kaggle/input"):
                print(f"  {root}: {files}")
        
        print("Attempting to download via Kaggle API...")
        try:
            import subprocess
            import zipfile
            # Ensure target directory exists
            os.makedirs("data/nlp-getting-started", exist_ok=True)
            
            # Download
            subprocess.run(["kaggle", "competitions", "download", "-c", "nlp-getting-started"], check=True)
            
            # Unzip
            zip_path = "nlp-getting-started.zip"
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("data/nlp-getting-started")
                base_path = "data/nlp-getting-started"
                print(f"Download and extraction successful to {base_path}")
            else:
                print("Download command ran but zip file not found.")
                
        except Exception as e:
            print(f"Auto-download failed: {e}")
            print("Please manually add the 'nlp-getting-started' competition dataset to your Kaggle Notebook.")
            print("Steps: Add Data -> Competitions -> NLP with Disaster Tweets -> Add")

    print(f"Loading data from: {base_path}")
    print(f"Train file: {train_file}, Test file: {test_file}")

    if base_path and train_file:
        train_dataset = TweetDataset(f"{base_path}/{train_file}", tokenizer)
        test_dataset = TweetDataset(f"{base_path}/{test_file}", tokenizer, is_test=True)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        model = MPIDisasterModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        print("Starting training...")
        for epoch in range(3):
            loss = train(model, train_loader, optimizer)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

        # ==========================================
        # 4. Inference & Submission
        # ==========================================
        print("Generating predictions...")
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Predicting'):
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits, _ = model(input_ids, mask)
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

        # Load original test file to get IDs, even if using preprocessed for input
        # We need 'id' column from the file we used for prediction
        sub_df = pd.read_csv(f"{base_path}/{test_file}")
        
        # Ensure 'id' column exists, otherwise create dummy index
        if 'id' not in sub_df.columns:
             sub_df['id'] = sub_df.index
             
        sub_df['target'] = preds
        submission = sub_df[['id', 'target']]
        submission.to_csv("submission.csv", index=False)
        print("Submission saved to submission.csv!")
        
        return model, submission
    else:
        print("Error: Dataset not found. Please add 'nlp-getting-started' dataset to your Kaggle Notebook.")
        return None, None

if __name__ == "__main__":
    main()
