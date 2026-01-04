import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import sys
import re
from collections import Counter
from sklearn.model_selection import train_test_split

# Add models directory to path
import sys
import os
import importlib

models_path = r"d:\code\MPI\models"
if models_path not in sys.path:
    sys.path.insert(0, models_path)

try:
    import echy_sars
    # Force reload to ensure we get the latest version
    importlib.reload(echy_sars)
    from echy_sars import ECHT_SARS_v2, DEVICE
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

# --- Configuration ---
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
VOCAB_SIZE = 5000  # Limit vocab size to avoid OOM on small embeddings
DATA_DIR = r"d:\code\MPI\competitions\Kaggle\Disaster_Tweets\data\nlp-getting-started"
if not os.path.exists(DATA_DIR):
    DATA_DIR = r"d:\code\MPI\data\nlp-getting-started"

# --- 1. Data Preprocessing ---

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove special chars
    return text

def build_vocab(texts, max_vocab=5000):
    words = []
    for text in texts:
        words.extend(text.split())
    
    counter = Counter(words)
    common_words = counter.most_common(max_vocab - 2) # Reserve 0 for PAD, 1 for UNK
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in common_words:
        vocab[word] = len(vocab)
        
    return vocab

def text_to_indices(text, vocab, max_len):
    tokens = text.split()
    indices = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
        
    return indices

class KaggleTweetDataset(Dataset):
    def __init__(self, df, vocab, max_len=64):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        self.df['clean_text'] = self.df['text'].apply(clean_text)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['clean_text']
        indices = text_to_indices(text, self.vocab, self.max_len)
        label = self.df.iloc[idx]['target']
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# --- 2. Training Pipeline ---

def train_echt_kaggle():
    print(f"--- ECHT-SARS v2 Kaggle Training ---")
    print(f"Device: {DEVICE}")
    
    # 1. Load Data
    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        print(f"Error: Data file not found at {train_path}")
        # Check alternate location
        alt_path = "data/nlp-getting-started/train.csv"
        if os.path.exists(alt_path):
            train_path = alt_path
            print(f"Found data at {train_path}")
        else:
            print("Creating dummy data for testing...")
            df = pd.DataFrame({
                'text': ['disaster fire burn'] * 50 + ['happy love peace'] * 50,
                'target': [1] * 50 + [0] * 50
            })
            train_path = None # Use df directly

    if train_path:
        df = pd.read_csv(train_path)
    
    print(f"Dataset shape: {df.shape}")
    
    # 2. Split Data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    # 3. Build Vocab
    print("Building vocabulary...")
    full_text = df['text'].apply(clean_text).tolist()
    vocab = build_vocab(full_text, max_vocab=VOCAB_SIZE)
    print(f"Vocabulary size: {len(vocab)}")
    
    # 4. Create Loaders
    train_ds = KaggleTweetDataset(train_df, vocab, MAX_LEN)
    val_ds = KaggleTweetDataset(val_df, vocab, MAX_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # 5. Initialize Model
    model = ECHT_SARS_v2(len(vocab)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    # 6. Train Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_psi = 0
        
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            preds, psi = model(X)
            loss = criterion(preds, y)
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
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds, _ = model(X)
                predicted = (torch.sigmoid(preds) > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Psi: {avg_psi:.4f} | Val Acc: {val_acc:.4f}")

    print(f"Training Complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train_echt_kaggle()
