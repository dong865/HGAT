"""
ST-HGAT: Spatio-Temporal Heterogeneous Graph Attention Network
时空异构图注意力网络 - 用于无人机追逃场景

核心创新:
1. 动态异构边权重 - 根据距离自适应调整不同类型边的注意力权重
2. 时空联合编码 - 将时序记忆嵌入图节点，替代级联GNN+LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class DynamicEdgeWeight(nn.Module):
    """
    动态异构边权重模块
    根据边类型和距离计算动态权重
    """
    def __init__(self, edge_types=['adversarial', 'cooperative', 'obstacle'],
                 d_threshold: float = 2.0,
                 sigma_cooperative: float = 0.3,
                 d_safe: float = 0.4,
                 map_size: float = 8.0):
        """
        Args:
            edge_types: 边类型列表
            d_threshold: 对抗边激活距离阈值 (米, 相对真实地图尺度)
            sigma_cooperative: 协同边高斯衰减参数
            d_safe: 避障安全距离阈值
            map_size: 地图边长（米），用于把归一化距离换算回真实米
        """
        super(DynamicEdgeWeight, self).__init__()
        self.edge_types = edge_types

        # 距离阈值配置
        self.d_threshold = d_threshold
        self.sigma_cooperative = sigma_cooperative
        self.d_safe = d_safe
        self.map_size = float(map_size)

    def compute_weight(self, distances: torch.Tensor, edge_type: str) -> torch.Tensor:
        """
        根据边类型和距离计算动态权重

        Args:
            distances: [N] 距离向量 (已按 map_size 归一化)
            edge_type: 边类型 ('adversarial', 'cooperative', 'obstacle')

        Returns:
            weights: [N] 动态权重
        """
        if edge_type == 'adversarial':
            # 追击-逃逸边: 仅当进入追击范围时激活 (距离按 map_size 归一化)
            # g(d) = 1_{[d <= d_thres/map_size]}
            weights = (distances <= self.d_threshold / self.map_size).float()
            
        elif edge_type == 'cooperative':
            # 追击-追击边: 高斯衰减，近距离协同强
            # g(d) = exp(-d^2 / 2*sigma^2)
            weights = torch.exp(-distances**2 / (2 * self.sigma_cooperative**2))
            
        elif edge_type == 'obstacle':
            # 无人机-障碍边: 距离越近权重线性增强
            # g(d) = max(0, 1 - d/d_safe)
            weights = torch.clamp(1.0 - distances / self.d_safe, min=0.0)
            
        else:
            # 默认: 固定权重1.0
            weights = torch.ones_like(distances)
            
        return weights


class HeterogeneousGATLayer(nn.Module):
    """
    单层异构图注意力层
    支持多种边类型的独立注意力计算
    """
    def __init__(self, in_dim: int, out_dim: int, edge_types: List[str], 
                 num_heads: int = 1, dropout: float = 0.1, map_size: float = 8.0):
        super(HeterogeneousGATLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_types = edge_types
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        # 每种边类型的独立参数
        self.W = nn.ModuleDict({
            etype: nn.Linear(in_dim, out_dim, bias=False)
            for etype in edge_types
        })
        
        # 每种边类型的注意力向量
        self.a = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(2 * out_dim, 1))
            for etype in edge_types
        })
        
        # 动态边权重模块
        self.edge_weight = DynamicEdgeWeight(edge_types, map_size=map_size)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for etype in self.edge_types:
            nn.init.xavier_uniform_(self.W[etype].weight)
            nn.init.xavier_uniform_(self.a[etype])
            
    def forward(self, h: torch.Tensor, adj_dict: Dict[str, torch.Tensor],
                dist_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h: [N, in_dim] 节点特征
            adj_dict: {edge_type: [N, N] 邻接矩阵}
            dist_dict: {edge_type: [N, N] 距离矩阵}
            
        Returns:
            h_out: [N, out_dim] 更新后的节点特征
        """
        N = h.size(0)
        aggregated = torch.zeros(N, self.out_dim, device=h.device)
        
        for etype in self.edge_types:
            if etype not in adj_dict or adj_dict[etype].sum() == 0:
                continue
                
            adj = adj_dict[etype]  # [N, N]
            dist = dist_dict[etype]  # [N, N]
            
            # 线性变换
            Wh = self.W[etype](h)  # [N, out_dim]
            
            # 计算注意力系数
            # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
            e = self._compute_attention(Wh, etype)  # [N, N]
            
            # 应用动态边权重 g_tau(d_ij)
            g_dist = self.edge_weight.compute_weight(dist, etype)  # [N, N]
            e = e * g_dist
            # 真正屏蔽被门控掉 (g==0) 的边: 置为 -inf, 否则 softmax 仍会分配 attention
            e = e.masked_fill(g_dist <= 0, float('-inf'))
            
            # 应用邻接掩码
            e = e.masked_fill(adj == 0, float('-inf'))
            
            # 归一化
            alpha = F.softmax(e, dim=1)  # [N, N]
            alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)
            alpha = self.dropout(alpha)
            
            # 消息聚合
            h_agg = torch.matmul(alpha, Wh)  # [N, out_dim]
            aggregated += h_agg
            
        return F.elu(aggregated)
    
    def _compute_attention(self, Wh: torch.Tensor, etype: str) -> torch.Tensor:
        """计算注意力系数 - 使用向量化的注意力计算"""
        N = Wh.size(0)
        
        # 方法: 使用拼接后矩阵乘法计算注意力
        # Wh_i: [N, out_dim], Wh_j: [N, out_dim]
        
        # 扩展维度以便广播: [N, 1, out_dim] 和 [1, N, out_dim]
        Wh_i = Wh.unsqueeze(1).expand(N, N, -1)  # [N, N, out_dim]
        Wh_j = Wh.unsqueeze(0).expand(N, N, -1)  # [N, N, out_dim]
        
        # 拼接: [N, N, 2*out_dim]
        concat = torch.cat([Wh_i, Wh_j], dim=-1)
        
        # 计算注意力分数: [N, N, 2*out_dim] @ [2*out_dim, 1] = [N, N, 1]
        a_expanded = self.a[etype].unsqueeze(0).expand(N, -1, -1)  # [N, 2*out_dim, 1]
        e = torch.bmm(concat, a_expanded).squeeze(-1)  # [N, N]
        
        return self.leakyrelu(e)


class ST_HGAT_Cell(nn.Module):
    """
    时空异构图注意力单元 (ST-HGAT Cell)
    融合GRU门控机制的时空消息传递
    """
    def __init__(self, node_dim: int, hidden_dim: int, edge_types: List[str],
                 num_heads: int = 1, dropout: float = 0.1, map_size: float = 8.0):
        super(ST_HGAT_Cell, self).__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        # 观测嵌入层
        self.obs_embed = nn.Linear(node_dim, hidden_dim)
        
        # HGAT消息聚合
        self.hgat = HeterogeneousGATLayer(
            hidden_dim, hidden_dim, edge_types, num_heads, dropout, map_size=map_size
        )
        
        # GRU门控参数
        self.W_z = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W_r = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W_h = nn.Linear(2 * hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.obs_embed.weight)
        nn.init.xavier_uniform_(self.W_z.weight)
        nn.init.xavier_uniform_(self.W_r.weight)
        nn.init.xavier_uniform_(self.W_h.weight)
        
    def forward(self, obs: torch.Tensor, h_prev: torch.Tensor,
                adj_dict: Dict[str, torch.Tensor],
                dist_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        单步前向传播
        
        Args:
            obs: [N, node_dim] 当前观测
            h_prev: [N, hidden_dim] 上一时刻隐藏状态
            adj_dict: 邻接字典
            dist_dict: 距离字典
            
        Returns:
            h_t: [N, hidden_dim] 当前隐藏状态
        """
        # 观测嵌入
        s_t = self.obs_embed(obs)  # [N, hidden_dim]
        
        # HGAT消息聚合 (使用上一时刻隐藏状态)
        if h_prev is None:
            h_prev = torch.zeros_like(s_t)
            
        m_t = self.hgat(h_prev, adj_dict, dist_dict)  # [N, hidden_dim]
        
        # 拼接当前观测和消息
        concat = torch.cat([s_t, m_t], dim=-1)  # [N, 2*hidden_dim]
        concat = self.dropout(concat)
        
        # GRU门控更新
        z_t = torch.sigmoid(self.W_z(concat))  # 更新门
        r_t = torch.sigmoid(self.W_r(concat))  # 重置门
        
        concat_reset = torch.cat([s_t, r_t * m_t], dim=-1)
        h_tilde = torch.tanh(self.W_h(concat_reset))  # 候选状态
        
        # 最终隐藏状态
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t


class ST_HGAT(nn.Module):
    """
    完整的ST-HGAT网络
    处理时序图序列，输出时空特征
    """
    def __init__(self, node_dim: int, hidden_dim: int, edge_types: List[str],
                 num_layers: int = 1, num_heads: int = 1, dropout: float = 0.1,
                 map_size: float = 8.0):
        super(ST_HGAT, self).__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 多层ST-HGAT
        self.cells = nn.ModuleList([
            ST_HGAT_Cell(node_dim if i == 0 else hidden_dim, 
                        hidden_dim, edge_types, num_heads, dropout, map_size=map_size)
            for i in range(num_layers)
        ])
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, obs_seq: List[torch.Tensor], 
                adj_seq: List[Dict[str, torch.Tensor]],
                dist_seq: List[Dict[str, torch.Tensor]],
                h_init: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（序列）
        
        Args:
            obs_seq: List of [N, node_dim] 观测序列
            adj_seq: List of adj_dict 邻接序列
            dist_seq: List of dist_dict 距离序列
            h_init: [N, hidden_dim] 初始隐藏状态
            
        Returns:
            h_final: [N, hidden_dim] 最终隐藏状态
            h_all: [T, N, hidden_dim] 所有时刻隐藏状态
        """
        T = len(obs_seq)
        N = obs_seq[0].size(0)
        
        if h_init is None:
            h = torch.zeros(N, self.hidden_dim, device=self.device)
        else:
            h = h_init.to(self.device)
            
        h_all = []
        
        for t in range(T):
            obs_t = obs_seq[t].to(self.device)
            adj_t = {k: v.to(self.device) for k, v in adj_seq[t].items()}
            dist_t = {k: v.to(self.device) for k, v in dist_seq[t].items()}
            
            # 逐层传播
            h_layer = obs_t
            for cell in self.cells:
                h_layer = cell(h_layer, h, adj_t, dist_t)
            h = h_layer
            
            h_all.append(h)
            
        h_all = torch.stack(h_all, dim=0)  # [T, N, hidden_dim]
        
        return h, h_all
    
    def step(self, obs: torch.Tensor, h_prev: torch.Tensor,
             adj_dict: Dict[str, torch.Tensor],
             dist_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        单步前向（用于在线推理）
        
        Args:
            obs: [N, node_dim] 当前观测
            h_prev: [N, hidden_dim] 上一时刻隐藏状态
            adj_dict: 当前邻接
            dist_dict: 当前距离
            
        Returns:
            h_t: [N, hidden_dim] 当前隐藏状态
        """
        obs = obs.to(self.device)
        h_prev = h_prev.to(self.device) if h_prev is not None else None
        adj_dict = {k: v.to(self.device) for k, v in adj_dict.items()}
        dist_dict = {k: v.to(self.device) for k, v in dist_dict.items()}
        
        h = obs
        for cell in self.cells:
            h = cell(h, h_prev, adj_dict, dist_dict)
            h_prev = h
            
        return h


class GraphBuilder:
    """
    异构图构建器
    根据无人机追逃场景构建图结构
    """
    def __init__(self, num_pursuers: int = 6, num_evaders: int = 2, 
                 num_obstacles: int = 9, map_size: float = 4.0):
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders
        self.num_obstacles = num_obstacles
        self.map_size = map_size
        self.edge_types = ['adversarial', 'cooperative', 'obstacle']
        
    def build_graph(self, positions: np.ndarray,
                   velocities: np.ndarray) -> Tuple[Dict, Dict, torch.Tensor]:
        """
        构建异构图

        Args:
            positions: [N_total, 3] 所有智能体与障碍物位置 (x, y, z)
            velocities: [N, 3] 所有智能体速度 (障碍物速度可填 0)

        Returns:
            adj_dict: 邻接字典
            dist_dict: 距离字典
            node_features: [N_total, 9] 节点特征 [pos(3), vel(3), type(3)]
        """
        N = self.num_pursuers + self.num_evaders
        N_total = N + self.num_obstacles

        # 初始化邻接矩阵
        adj_adversarial = np.zeros((N_total, N_total))
        adj_cooperative = np.zeros((N_total, N_total))
        adj_obstacle = np.zeros((N_total, N_total))

        # 初始化距离矩阵 (3-D 欧氏距离，按 map_size 归一化)
        dist_matrix = np.zeros((N_total, N_total))
        for i in range(N_total):
            for j in range(N_total):
                dist_matrix[i, j] = np.linalg.norm(positions[i] - positions[j]) / self.map_size

        # 1. 对抗边: 追击者 <-> 逃逸者 (双向)
        #    追捕者接收逃逸者信息用于拦截决策; 逃逸者节点(预测器输入)也必须
        #    接收追捕者信息, 才能建模"追捕导致的逃逸响应" (原实现仅单向).
        for p in range(self.num_pursuers):
            for e in range(self.num_evaders):
                evader_idx = self.num_pursuers + e
                adj_adversarial[p, evader_idx] = 1
                adj_adversarial[evader_idx, p] = 1

        # 2. 协同边: 追击者 <-> 追击者
        for p1 in range(self.num_pursuers):
            for p2 in range(self.num_pursuers):
                if p1 != p2:
                    adj_cooperative[p1, p2] = 1

        # 3. 障碍边: 所有智能体 -> 障碍物
        for agent in range(N):
            for obs in range(self.num_obstacles):
                obs_idx = N + obs
                adj_obstacle[agent, obs_idx] = 1

        # 构建节点特征 [pos(3), vel(3), type(3)]
        node_features = np.zeros((N_total, 9))
        for i in range(N):
            node_features[i, :3] = positions[i, :3] / self.map_size  # 归一化位置
            node_features[i, 3:6] = velocities[i, :3] / 0.12  # 归一化速度
            if i < self.num_pursuers:
                node_features[i, 6] = 1.0  # 追击者类型
            else:
                node_features[i, 7] = 1.0  # 逃逸者类型

        # 障碍物节点特征
        for obs in range(self.num_obstacles):
            obs_idx = N + obs
            node_features[obs_idx, :3] = positions[obs_idx, :3] / self.map_size
            node_features[obs_idx, 3:6] = 0.0  # 障碍物速度为0
            node_features[obs_idx, 8] = 1.0    # 障碍物类型标记

        # 转换为tensor
        adj_dict = {
            'adversarial': torch.FloatTensor(adj_adversarial),
            'cooperative': torch.FloatTensor(adj_cooperative),
            'obstacle': torch.FloatTensor(adj_obstacle)
        }

        dist_dict = {
            'adversarial': torch.FloatTensor(dist_matrix),
            'cooperative': torch.FloatTensor(dist_matrix),
            'obstacle': torch.FloatTensor(dist_matrix)
        }

        node_features = torch.FloatTensor(node_features)

        return adj_dict, dist_dict, node_features


if __name__ == '__main__':
    # 测试代码
    print("Testing ST-HGAT...")
    
    # 参数
    num_pursuers, num_evaders, num_obstacles = 6, 2, 9
    node_dim = 9
    hidden_dim = 128

    # 创建网络
    st_hgat = ST_HGAT(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        edge_types=['adversarial', 'cooperative', 'obstacle'],
        num_layers=1,
        num_heads=1
    )

    # 创建图构建器
    graph_builder = GraphBuilder(num_pursuers, num_evaders, num_obstacles)

    # 模拟数据
    N_total = num_pursuers + num_evaders + num_obstacles
    positions = np.random.rand(N_total, 3) * 4.0
    velocities = np.random.randn(N_total, 3) * 0.05
    
    # 构建图
    adj_dict, dist_dict, node_features = graph_builder.build_graph(positions, velocities)
    
    # 单步测试
    h_prev = torch.zeros(N_total, hidden_dim)
    h_t = st_hgat.step(node_features, h_prev, adj_dict, dist_dict)
    
    print(f"Input shape: {node_features.shape}")
    print(f"Output shape: {h_t.shape}")
    print("ST-HGAT test passed!")
