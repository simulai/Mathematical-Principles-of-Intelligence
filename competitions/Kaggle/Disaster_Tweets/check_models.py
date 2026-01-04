# ==========================================
# Kaggle Ensemble Submission Script
# Ensembles Quantum-ECHT and Classical-ECHT models
# ==========================================

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

# Re-import definitions from both files by copying them here to ensure self-contained script
# Or import if possible. Since they are in the same dir, we can try to import.
# But for robustness, I will redefine the minimal inference classes here.
# Actually, let's assume we can run inference using the weights we (hopefully) have.

# Wait, `kaggle_verify_score_0.81213.py` saves `best_model_fold_{fold}.pth`
# `kaggle_quantum_submission.py` saves `best_model_quantum_fold_{fold}.pth`

# If the user ran `kaggle_verify_score_0.81213.py` previously, the weights should be there.
# If not, we might need to train it first.
# The user's todo says "在 Kaggle 上提交 Quantum-ECHT (0.8162+) 结果" is completed.
# The "verify_echt_kaggle" task was completed earlier.
# So `best_model_fold_*.pth` should exist? Let's check.

def check_models_exist():
    missing_classical = []
    missing_quantum = []
    for i in range(5):
        if not os.path.exists(f"best_model_fold_{i}.pth"):
            missing_classical.append(i)
        if not os.path.exists(f"best_model_quantum_fold_{i}.pth"):
            missing_quantum.append(i)
    
    return missing_classical, missing_quantum

if __name__ == "__main__":
    missing_c, missing_q = check_models_exist()
    if missing_c:
        print(f"Warning: Missing Classical models for folds: {missing_c}")
    else:
        print("All Classical models found.")
        
    if missing_q:
        print(f"Warning: Missing Quantum models for folds: {missing_q}")
    else:
        print("All Quantum models found.")
        
    # If classical models are missing, we might need to generate them or just rely on quantum.
    # But wait, did we actually RUN `kaggle_verify_score_0.81213.py` fully on this machine?
    # The user history implies we did verify it.
