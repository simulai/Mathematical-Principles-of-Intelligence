import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import sys
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Add models directory to path
models_path = r"d:\code\MPI\models"
if models_path not in sys.path:
    sys.path.insert(0, models_path)

try:
    from echy_bert_hybrid import ECHT_BERT_Hybrid
except ImportError as e:
    print(f"Error importing model: {e}")
    sys.exit(1)

# --- Configuration ---
MODEL_NAME = "prajjwal1/bert-tiny" # Fast CPU model
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 10 # Increase epochs to reach convergence
LEARNING_RATE = 2e-5 # Standard BERT LR
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = r"d:\code\MPI\competitions\Kaggle\Disaster_Tweets\data\nlp-getting-started"
if not os.path.exists(DATA_DIR):
    DATA_DIR = r"d:\code\MPI\data\nlp-getting-started"

# --- 1. Data Processing ---
class BertTweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = df['text'].astype(str).tolist()
        self.labels = df['target'].tolist()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# --- 2. Training Loop ---
def train_hybrid():
    print(f"--- ECHT-BERT Hybrid Training (CPU Fast Mode) ---")
    print(f"Device: {DEVICE}")
    print(f"Backbone: {MODEL_NAME}")
    
    # Load Data
    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        print("Data not found. Please download dataset first.")
        return

    df = pd.read_csv(train_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    # Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_ds = BertTweetDataset(train_df, tokenizer, MAX_LEN)
    val_ds = BertTweetDataset(val_df, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # Model
    print("Initializing Hybrid Model...")
    model = ECHT_BERT_Hybrid(model_name=MODEL_NAME).to(DEVICE)
    
    # Optimizer (Layer-wise LR could be better, but keep simple)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_psi = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            preds, psi = model(input_ids, mask)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_psi += psi.mean().item()
            
        avg_loss = total_loss / len(train_loader)
        avg_psi = total_psi / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                preds, _ = model(input_ids, mask)
                predicted = (torch.sigmoid(preds) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Psi: {avg_psi:.4f} | Val Acc: {val_acc:.4f}")

    print(f"Training Complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train_hybrid()
