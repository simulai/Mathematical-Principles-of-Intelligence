import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

try:
    from echy_sars import HamiltonianReasoningLayer
except ImportError:
    import sys
    sys.path.append(r"d:\code\MPI\models")
    from echy_sars import HamiltonianReasoningLayer

class ECHT_BERT_Hybrid(nn.Module):
    def __init__(self, model_name="prajjwal1/bert-tiny", hidden_dim=64):
        super().__init__()
        
        # 1. 语义传感器 (BERT)
        # 使用 TinyBERT (L=2, H=128) 保证 CPU 上的极速运行
        print(f"Loading Semantic Sensor: {model_name}...")
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 冻结 BERT 前几层以加快训练 (可选)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
            
        bert_dim = self.bert.config.hidden_size
        
        # 2. 维度投影 (Projection to Phase Space)
        # 将 BERT 的高维语义映射到 ECHT 的低维流形
        self.projector = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim * 2), # Project to complex dim (Real + Imag)
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU()
        )
        
        self.hidden_dim = hidden_dim
        
        # 3. 物理推理引擎 (ECHT)
        self.reasoning = HamiltonianReasoningLayer(hidden_dim)
        
        # 4. 坍缩分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input_ids, attention_mask):
        # A. 语义感知
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use full sequence for reasoning flow
        sequence_output = outputs.last_hidden_state # [batch, seq, bert_dim]
        
        # B. 注入相空间 (Phase Space Injection)
        # 将 BERT 向量投影为 复数 z = q + ip
        projected = self.projector(sequence_output)
        q = projected[:, :, :self.hidden_dim]
        p = projected[:, :, self.hidden_dim:]
        z = torch.complex(q, p)
        
        # C. 辛几何演化 (Symplectic Evolution)
        # 让语义在哈密顿场中自然演化，寻找能量最低的解释
        z_final, psi = self.reasoning(z)
        
        # D. 观测坍缩 (Observation Collapse)
        # 取实部作为可观测的物理量
        logits = self.classifier(z_final.real.mean(dim=1))
        
        return logits.squeeze(), psi
