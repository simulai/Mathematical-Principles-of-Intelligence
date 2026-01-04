import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DensityMatrixEmbedding(nn.Module):
    """
    Step 1: 将经典向量转换为量子密度矩阵 (Density Matrix)
    输入: 向量 v [batch, dim]
    输出: 密度矩阵 rho [batch, dim, dim] (Hermitian, Positive Semi-definite, Trace=1)
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # 可学习的混合参数，控制初始状态的“纯度”
        self.mix_param = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # 1. 归一化输入向量，使其模长为1 (对应纯态 |v>)
        # x: [batch, dim]
        v = F.normalize(x, p=2, dim=-1)
        
        # 2. 构建纯态密度矩阵 rho_pure = |v><v|
        # v.unsqueeze(2): [batch, dim, 1]
        # v.unsqueeze(1): [batch, 1, dim]
        # bmm: [batch, dim, 1] x [batch, 1, dim] -> [batch, dim, dim]
        rho_pure = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))
        
        # 3. 引入混合度 (Mixed State)
        # rho = (1 - epsilon) * rho_pure + epsilon * I / d
        # 这模拟了环境噪声或初始的不确定性
        epsilon = torch.sigmoid(self.mix_param)
        identity = torch.eye(self.embed_dim, device=x.device).unsqueeze(0).expand_as(rho_pure)
        
        rho = (1 - epsilon) * rho_pure + epsilon * (identity / self.embed_dim)
        
        return rho

class LindbladEvolution(nn.Module):
    """
    Step 2: Lindblad Master Equation 演化
    d_rho/dt = -i[H, rho] + sum_k (L_k rho L_k^dagger - 1/2 {L_k^dagger L_k, rho})
    """
    def __init__(self, embed_dim, num_jump_ops=2, dt=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dt = dt
        
        # Hamiltonian H (必须是厄米矩阵)
        # 我们用一个可学习的下三角矩阵构建 H = A + A^dagger
        self.H_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        
        # Jump Operators L_k (描述耗散/退相干)
        # 比如：L_1 可能是“遗忘算符”，L_2 可能是“语义漂移算符”
        self.jump_ops = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.1)
            for _ in range(num_jump_ops)
        ])

    def get_hamiltonian(self):
        # 保证 H 是厄米矩阵
        H = self.H_params
        return (H + H.t()) / 2

    def forward(self, rho):
        # rho: [batch, dim, dim]
        H = self.get_hamiltonian()
        
        # 1. 幺正演化项 (Unitary Part): -i[H, rho]
        # Commutator [H, rho] = H rho - rho H
        # H 是 [dim, dim], rho 是 [batch, dim, dim]
        # 需要广播 H
        H_expanded = H.unsqueeze(0).expand_as(rho)
        commutator = torch.bmm(H_expanded, rho) - torch.bmm(rho, H_expanded)
        d_rho_unitary = -1j * commutator 
        
        # 2. 耗散项 (Dissipative Part): L rho L^dagger - 0.5 {L^dagger L, rho}
        d_rho_dissipative = torch.zeros_like(rho)
        for L in self.jump_ops:
            L_exp = L.unsqueeze(0).expand_as(rho)
            L_dag = L.t().conj().unsqueeze(0).expand_as(rho)
            
            # Term 1: L rho L^dagger
            term1 = torch.bmm(L_exp, torch.bmm(rho, L_dag))
            
            # Term 2: {L^dagger L, rho} = (L^dagger L) rho + rho (L^dagger L)
            L_dag_L = torch.matmul(L.t().conj(), L).unsqueeze(0).expand_as(rho)
            term2 = 0.5 * (torch.bmm(L_dag_L, rho) + torch.bmm(rho, L_dag_L))
            
            d_rho_dissipative += (term1 - term2)
            
        return rho + (d_rho_unitary + d_rho_dissipative) * self.dt

class QuantumMeasurement(nn.Module):
    """
    Step 3: 量子测量 (POVM 或 Projective Measurement)
    P(y) = Tr(M rho)
    """
    def __init__(self, embed_dim, num_classes=1):
        super().__init__()
        # 测量算符 M (Positive Semi-definite)
        # M = V^dagger V
        self.measure_vectors = nn.Parameter(torch.randn(num_classes, embed_dim))

    def forward(self, rho):
        # M: [num_classes, dim]
        # rho: [batch, dim, dim]
        
        # P = Tr(M rho) = Tr(|u><u| rho) = <u| rho |u>
        # 其中 |u> 是测量向量
        
        M = self.measure_vectors # [1, dim]
        M = F.normalize(M, p=2, dim=-1) # 归一化
        
        # 计算期望值 <u| rho |u>
        # u: [1, dim] -> [batch, 1, dim]
        u = M.unsqueeze(0).expand(rho.size(0), -1, -1)
        
        # rho @ u^T : [batch, dim, dim] @ [batch, dim, 1] -> [batch, dim, 1]
        # u @ (rho @ u^T) : [batch, 1, dim] @ [batch, dim, 1] -> [batch, 1, 1]
        
        # 注意：如果是复数 rho，这里应该是 u.conj()
        # 这里必须使用 complex 逻辑
        # u: [batch, 1, dim] (real) -> complex? We assume measurement vectors are real for now, 
        # or we make them complex parameters. Let's stick to real M for simplicity, acting on complex rho.
        
        # complex rho: [batch, dim, dim]
        # u: [batch, 1, dim]
        
        # inner: rho @ u^T -> [batch, dim, 1] (complex)
        u_T = u.transpose(1, 2).type_as(rho) # [batch, dim, 1]
        inner = torch.bmm(rho, u_T)
        
        # expectation: u @ inner -> [batch, 1, 1] (complex)
        u_complex = u.type_as(rho)
        expectation = torch.bmm(u_complex, inner)
        
        # 物理上，测量结果必须是实数。
        # 由于 rho 是 Hermit 矩阵，<u|rho|u> 必然是实数。
        # 我们取实部即可。
        return expectation.squeeze().real

