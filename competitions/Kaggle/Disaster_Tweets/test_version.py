import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt

# ==================== 0. 常量与配置 ====================
GAMMA = 0.75       # 相干性判定阈值（喜爱阈值）
PHASE_LIMIT = np.pi / 10  # 相位微调的最大幅度

# ==================== 1. 物理位置编码：动量注入法 ====================
class PhysicalPositionalEncoding(nn.Module):
    """
    物理位置编码：将位置信息编码为初始动量，让序列顺序在辛演化中自然融合。
    """
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        # 生成与Transformer PE相同频率的波长，但用于初始化动量
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # 正弦部分
        pe[:, 1::2] = torch.cos(position * div_term)  # 余弦部分
        self.register_buffer('pe', pe)  # [max_len, embed_dim]

    def forward(self, q, positions=None):
        """
        q: [batch, seq_len, embed_dim] 语义实部
        返回: 携带位置信息的复数状态 z
        """
        batch_size, seq_len, _ = q.shape
        if positions is None:
            positions = torch.arange(seq_len, device=q.device).expand(batch_size, seq_len)
        
        # 获取位置对应的动量编码
        p0 = self.pe[positions]  # [batch, seq_len, embed_dim]
        
        # 关键：将位置信息作为初始动量，与语义实部构成复数
        z = torch.complex(q, p0)
        return z


# ==================== 2. 能量归一化层 ====================
class EnergyNormalization(nn.Module):
    """
    能量归一化：基于系统总能量 |z|^2 进行缩放，替代LayerNorm。
    具有明确的物理意义（保持能量稳定），且处处可微。
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        """
        z: 复数张量 [..., dim]
        返回: 能量归一化后的复数张量
        """
        # 计算每个位置的总能量 E = |z|^2 = q^2 + p^2
        # 保持维度以便广播
        energy = torch.sum(z.real**2 + z.imag**2, dim=-1, keepdim=True) + self.eps
        scale = torch.rsqrt(energy)  # 1 / sqrt(E)
        
        # 应用缩放：这不会改变相位，只改变振幅
        return torch.complex(z.real * scale, z.imag * scale)


# ==================== 3. 多视角势能场（物理多头） ====================
class MultiPerspectiveHamiltonian(nn.Module):
    """
    多视角势能场：物理原生的“多头”机制。
    每个势能场从不同“物理视角”理解序列，最后通过相干叠加合并。
    """
    def __init__(self, embed_dim, num_perspectives=4, dt=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_perspectives = num_perspectives
        self.dt = dt
        
        # 为每个视角创建独立的势能场参数
        self.W_k = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02 / math.sqrt(num_perspectives))
            for _ in range(num_perspectives)
        ])
        
        # 可学习的干涉权重，用于合并不同视角
        self.interference_weights = nn.Parameter(torch.ones(num_perspectives))

    def get_symmetric_weights(self, W):
        """确保势能场权重对称，维持保守场性质"""
        return (W + W.t()) / 2

    def single_perspective_flow(self, z, W):
        """单个势能场下的辛积分演化"""
        sym_W = self.get_symmetric_weights(W)
        q, p = z.real, z.imag
        
        # 蛙跳积分 (二阶辛，精度更好)
        p_half = p - (self.dt / 2) * torch.tanh(q @ sym_W)
        q_next = q + self.dt * p_half
        p_next = p_half - (self.dt / 2) * torch.tanh(q_next @ sym_W)
        
        return torch.complex(q_next, p_next)

    def forward(self, z):
        """
        z: 复数输入 [batch, seq_len, embed_dim]
        返回: 多视角演化并相干叠加后的状态
        """
        perspective_outputs = []
        
        # 并行计算每个视角的演化
        for k in range(self.num_perspectives):
            z_k = self.single_perspective_flow(z, self.W_k[k])
            perspective_outputs.append(z_k)
        
        # 相干叠加：带权重的复数相加（类似波干涉）
        weights = F.softmax(self.interference_weights, dim=0)
        
        # 加权求和
        z_combined = torch.zeros_like(z)
        for k in range(self.num_perspectives):
            z_combined += weights[k] * perspective_outputs[k]
        
        return z_combined

# ==================== 3.5 SmoothSARS (PhaseLockedSARS) ====================
class SmoothSARS(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        # 相位调制器：学习如何通过微调相位实现“共振”
        self.phase_tuner = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim)
        )

    def forward(self, z):
        # z: [batch, seq, dim] (complex64)
        q = z.real
        
        # 1. 计算当前的全局相干性能量 Psi (序参量)
        # 衡量 Token 之间的相位是否趋于同步
        phase_orig = z.angle()
        psi = torch.abs(torch.exp(1j * phase_orig).mean(dim=1)).mean(dim=-1)
        
        # 2. 逻辑感应反馈：生成相位修正量 phi_corr
        # 当模型“喜爱”某个信号时，试图通过调整相位使其对齐
        # 这种操作不改变模长（不增加能量），仅通过几何对齐抑制熵增
        phi_corr = torch.tanh(self.phase_tuner(q.mean(dim=1))) * PHASE_LIMIT
        phi_corr = phi_corr.unsqueeze(1) # [batch, 1, dim]
        
        # 3. 执行相位锁定 (Phase-Locking)
        z_tuned = z * torch.exp(1j * phi_corr)
        
        # 4. 判定是否达到饱和状态（即逻辑完全自洽）
        # is_saturated = psi > GAMMA
        return z_tuned, psi # 返回 z_tuned 和 psi

# ==================== 4. 增强的ECHT-SARS引擎 ====================
class EnhancedECHT_SARS(nn.Module):
    """
    集成所有物理原生组件的增强版引擎
    """
    def __init__(self, vocab_size, embed_dim=64, 
                 num_perspectives=4, num_evolution_steps=3):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 基础组件
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 新增物理原生组件
        self.physical_pe = PhysicalPositionalEncoding(embed_dim)
        self.energy_norm = EnergyNormalization()
        self.multi_hamiltonian = MultiPerspectiveHamiltonian(
            embed_dim, num_perspectives=num_perspectives
        )
        
        # SARS监控器
        self.sars_monitor = SmoothSARS(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)
        
        self.num_evolution_steps = num_evolution_steps

    def forward(self, x, return_physics=False, record_trajectory=False):
        """
        增强的前向传播流程
        """
        physics_metrics = {}
        trajectories = {
            'psi_history': [],
            'energy_history': [],
            'q_history': [],
            'p_history': []
        }
        
        # 1. 语义嵌入
        q0 = self.embedding(x)  # [batch, seq_len, dim]
        
        # 2. 物理位置编码（将位置信息注入动量）
        z = self.physical_pe(q0)  # 现在是复数
        physics_metrics['initial_energy'] = torch.norm(z, p='fro').item()
        
        if record_trajectory:
            trajectories['psi_history'].append(0.0) # Initial psi placeholder
            trajectories['energy_history'].append(physics_metrics['initial_energy'])
            trajectories['q_history'].append(z.real.detach().cpu().numpy())
            trajectories['p_history'].append(z.imag.detach().cpu().numpy())

        # 3. 多步物理演化
        psi_step = torch.zeros(1) # Default
        for step in range(self.num_evolution_steps):
            # 3.1 多视角势能场演化
            z = self.multi_hamiltonian(z)
            
            # 3.2 能量归一化（保持稳定）
            z = self.energy_norm(z)
            
            # 3.3 SARS监控与能量增强
            z, psi_step = self.sars_monitor(z)
            
            if step == 0:
                physics_metrics['psi_initial'] = psi_step.mean().item()
            
            if record_trajectory:
                trajectories['psi_history'].append(psi_step.mean().item())
                trajectories['energy_history'].append(torch.norm(z, p='fro').item())
                trajectories['q_history'].append(z.real.detach().cpu().numpy())
                trajectories['p_history'].append(z.imag.detach().cpu().numpy())
        
        # 4. 最终测量
        physics_metrics['final_energy'] = torch.norm(z, p='fro').item()
        physics_metrics['psi_final'] = psi_step.mean().item()
        
        # 5. 行为坍缩：取实部平均进行分类
        collapsed_state = z.real.mean(dim=1)
        logits = self.classifier(collapsed_state).squeeze(-1)
        
        if record_trajectory:
            # Stack histories for easier plotting
            trajectories['q_history'] = np.stack(trajectories['q_history'])
            trajectories['p_history'] = np.stack(trajectories['p_history'])
            return logits, trajectories

        if return_physics:
            return logits, physics_metrics
        return logits

# ==================== 5. 消融实验验证框架 ====================
class IdentityNorm(nn.Module):
    def forward(self, z):
        return z

class ZeroMomentumPE(nn.Module):
    def forward(self, q, positions=None):
        return torch.complex(q, torch.zeros_like(q))

def create_model_based_on_config(config):
    model = EnhancedECHT_SARS(
        vocab_size=10000,
        embed_dim=64,
        num_perspectives=config['num_perspectives'],
        num_evolution_steps=3
    )
    # 动态调整组件
    if not config['use_pe']:
        # 替换为普通的嵌入，没有动量注入 (p0=0)
        model.physical_pe = ZeroMomentumPE()
        
    if not config['use_norm']:
        # 替换为恒等映射
        model.energy_norm = IdentityNorm()
        
    return model

def train_and_evaluate(model):
    # 模拟快速训练过程
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 模拟数据集 (Batch=32, Len=32)
    X = torch.randint(0, 10000, (32, 32))
    y = torch.randint(0, 2, (32,)).float()
    
    train_losses = []
    
    # 训练 20 步
    model.train()
    for _ in range(20):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    # 验证并获取物理指标
    model.eval()
    with torch.no_grad():
        logits, physics = model(X, return_physics=True)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean().item()
        
    return train_losses, acc, physics

def run_ablation_study(dataset_name='demo'):
    """
    运行消融实验，验证每个物理原生组件的贡献
    """
    import pandas as pd
    
    configs = [
        {'name': '基础模型', 'use_pe': False, 'use_norm': False, 'num_perspectives': 1},
        {'name': '+物理位置编码', 'use_pe': True, 'use_norm': False, 'num_perspectives': 1},
        {'name': '+能量归一化', 'use_pe': True, 'use_norm': True, 'num_perspectives': 1},
        {'name': '+多势能场(4视角)', 'use_pe': True, 'use_norm': True, 'num_perspectives': 4},
    ]
    
    results = []
    
    print("\n" + "="*80)
    print("开始运行消融实验...")
    print("="*80)
    
    for config in configs:
        print(f"训练配置: {config['name']}")
        
        # 根据配置创建模型
        model = create_model_based_on_config(config)
        
        # 训练模型并收集指标
        train_losses, val_acc, physics_data = train_and_evaluate(model)
        
        # 分析物理指标
        # 防止除零错误
        psi_initial = physics_data.get('psi_initial', 1e-6)
        if psi_initial == 0: psi_initial = 1e-6
        
        initial_energy = physics_data.get('initial_energy', 1e-6)
        if initial_energy == 0: initial_energy = 1e-6
        
        coh_gain = physics_data.get('psi_final', 0) / psi_initial
        energy_ratio = physics_data.get('final_energy', 0) / initial_energy
        
        results.append({
            '配置': config['name'],
            '验证准确率': val_acc,
            '相干性增益': coh_gain,
            '能量变化比': energy_ratio,
            '训练稳定性': np.mean(train_losses[-5:])  # 最后5个loss的平均
        })
    
    # 结果表格
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("消融实验结果汇总")
    print("="*80)
    # 设置显示格式
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_results.to_string(index=False))
    
    return df_results


# ==================== 6. 可视化工具 ====================
def visualize_physics_trajectories(model, sample_input):
    """
    可视化物理演化轨迹，用于直观理解模型内部工作原理
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    with torch.no_grad():
        # 获取中间状态（需要在前向传播中添加状态记录）
        _, trajectories = model(sample_input, record_trajectory=True)
        
        # 创建多子图可视化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 相位相干性演化
        axes[0, 0].plot(trajectories['psi_history'], marker='o')
        axes[0, 0].set_title('Phase Coherence Ψ Evolution')
        axes[0, 0].axhline(y=GAMMA, color='r', linestyle='--', label='Threshold')
        axes[0, 0].set_xlabel('Evolution Step')
        axes[0, 0].set_ylabel('Ψ')
        axes[0, 0].legend()
        
        # 2. 能量演化
        axes[0, 1].plot(trajectories['energy_history'], marker='o', color='orange')
        axes[0, 1].set_title('System Total Energy Evolution')
        axes[0, 1].set_xlabel('Evolution Step')
        axes[0, 1].set_ylabel('Energy |z|^2')
        
        # 3. 实部-虚部相图（选前2个维度）
        # trajectories['q_history'] shape: [steps, batch, seq, dim]
        # We take first batch, first token, first dim
        q = trajectories['q_history'][:, 0, 0, 0]
        p = trajectories['p_history'][:, 0, 0, 0]
        axes[1, 0].plot(q, p, 'b.-')
        axes[1, 0].set_title('Phase Space Trajectory (q-p)')
        axes[1, 0].set_xlabel('Semantic Position q')
        axes[1, 0].set_ylabel('Logical Momentum p')
        axes[1, 0].grid(True)
        
        # 4. 多视角权重可视化
        if hasattr(model.multi_hamiltonian, 'interference_weights'):
            weights = model.multi_hamiltonian.interference_weights.detach()
            probs = F.softmax(weights, dim=0).numpy()
            axes[1, 1].bar(range(len(weights)), probs, color='purple')
            axes[1, 1].set_title('Multi-Perspective Interference Weights')
            axes[1, 1].set_xlabel('Perspective Index')
            axes[1, 1].set_ylabel('Weight')
        
        plt.tight_layout()
        output_path = 'physics_trajectories.png'
        plt.savefig(output_path, dpi=150)
        print(f"\n可视化图表已保存至: {output_path}")
        # plt.show() # 在服务器/无头环境中通常不调用show


# ==================== 使用示例 ====================
if __name__ == "__main__":
    print("正在初始化增强版 ECHT-SARS 物理引擎...")
    
    # 1. 创建增强模型
    model = EnhancedECHT_SARS(
        vocab_size=10000,
        embed_dim=64,
        num_perspectives=4,
        num_evolution_steps=5 # 增加步数以更好地观察演化
    )
    
    print("\n模型参数量统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 2. 测试前向传播
    batch_size, seq_len = 4, 32
    test_input = torch.randint(0, 10000, (batch_size, seq_len))
    
    print("\n执行前向传播测试...")
    logits, physics = model(test_input, return_physics=True)
    
    print("\n物理指标快照:")
    for key, value in physics.items():
        print(f"  {key}: {value:.4f}")
    
    # 3. 运行可视化
    print("\n生成物理演化轨迹可视化...")
    visualize_physics_trajectories(model, test_input)
    
    # 4. 运行消融实验
    run_ablation_study()

    # 5. 下一步建议
    print("\n" + "="*80)
    print("下一步行动建议:")
    print("  1. 检查生成的 'physics_trajectories.png' 以直观理解模型动力学")
    print("  2. 根据消融实验结果，决定是否在主模型中保留所有物理组件")
    print("  3. 尝试调整 'num_evolution_steps' 观察相空间轨迹的变化")
    print("  4. 将 EnhancedECHT_SARS 集成到 Kaggle 提交脚本中以提升性能")
