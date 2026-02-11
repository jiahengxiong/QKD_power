import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GlobalSelfAttention(nn.Module):
    """
    去残差版的自注意力：强制 NN 学习如何重新加权全局特征。
    """
    def __init__(self, in_channels, num_heads=4):
        super(GlobalSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query(x).view(batch_size, channels, -1).permute(0, 2, 1) 
        proj_key = self.key(x).view(batch_size, channels, -1) 
        
        energy = torch.bmm(proj_query, proj_key) 
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(batch_size, channels, -1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        
        out = out.view(batch_size, channels, height, width)
        # 彻底移除残差连接 (x + gamma * out)
        return out

class QKDGraphNet(nn.Module):
    """
    Refined QKDGraphNet: 
    保留空间结构，引入 Request Mask 和 瓶颈感知 (Max Pooling)。
    参数量极低，但信息流无损。
    """
    def __init__(self, num_global_features=7, num_wl_features=5, num_wavelengths=10, actual_nodes=12, is_bypass=True, hidden_dim=8):
        super(QKDGraphNet, self).__init__()
        self.num_nodes = actual_nodes
        self.is_bypass = is_bypass
        self.hidden_dim = hidden_dim
        
        # === 1. 特征预处理 ===
        # 使用 GroupNorm(1, C) 代替 InstanceNorm
        # InstanceNorm 会把全0或全1的 Binary 特征抹平成 0，导致信息丢失
        # GroupNorm(1) = LayerNorm，能保留 Channel 间的相对信息
        wl_in_channels = num_wl_features * num_wavelengths
        self.global_norm = nn.GroupNorm(1, num_global_features, affine=True)
        self.wl_norm = nn.GroupNorm(1, wl_in_channels, affine=True)
        
        # 特征压缩 (1x1 Conv)
        self.static_enc = nn.Conv2d(num_global_features, hidden_dim, kernel_size=1)
        self.dynamic_enc = nn.Conv2d(wl_in_channels, hidden_dim, kernel_size=1)
        
        # === 2. 记忆单元 (Global GRU) ===
        # 用于记录历史拥堵趋势 (虽然是全局的，但输入会经过 Max Pooling 强化瓶颈信息)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # === 3. 边评分器 (The "Brain") ===
        # 输入通道分析：
        # - Edge Feature (hidden): 当前边的物理/波长状态
        # - Node In (hidden): 源节点 u 的入度能力 ("能到 u 吗？")
        # - Node Out (hidden): 宿节点 v 的出度能力 ("能离开 v 吗？")
        # - Global Memory (hidden): 历史趋势
        # - Request Src Mask (1): 当前边是否始于源点
        # - Request Dst Mask (1): 当前边是否指向终点
        # - Traffic & Protocol (2): 标量广播
        in_dim = hidden_dim * 4 + 1 + 1 + 2
        
        self.scorer = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x_global, x_wl, context, last_action, h_prev):
        # x_global: [B, C_g, N, N]
        # x_wl: [B, C_w, N, N]
        # context: [B, 4] -> (src, dst, traffic, protocol)
        
        B, _, N, N = x_global.shape
        
        # === 0. Input Preprocessing (Log Transform) ===
        # 对所有输入特征统一做 Log1p 处理
        # 这对于长尾分布特征 (Power, Capacity) 是必要的
        # 对于线性分布特征 (0-1, Hops) 也是安全的 (ln(1+x) ~ x for small x)
        x_global = torch.log1p(x_global)
        x_wl = torch.log1p(x_wl)
        
        # === A. 基础边特征嵌入 ===
        x_g = self.static_enc(self.global_norm(x_global))
        x_w = self.dynamic_enc(self.wl_norm(x_wl))
        # 融合静态和动态特征
        h_edge = F.leaky_relu(x_g + x_w, 0.1) # [B, hidden, N, N]
        
        # === B. 节点特征聚合 (Bottleneck Awareness) ===
        # 使用 LogSumExp (Soft Max) 替代 Hard Max，让梯度能流向非最大值的边
        # beta=10.0: 足够接近 Max 但保持平滑
        beta = 10.0
        
        # Node Out (Max over Cols) -> [B, hidden, N, 1]
        node_out_feat = torch.logsumexp(h_edge * beta, dim=3, keepdim=True) / beta
        
        # Node In (Max over Rows) -> [B, hidden, 1, N]
        node_in_feat = torch.logsumexp(h_edge * beta, dim=2, keepdim=True) / beta
        
        # 广播到边 (u, v):
        # node_in_feat: [B, hidden, 1, N] -> expand dim 2 to N
        node_in_expanded = node_in_feat.expand(-1, -1, N, -1)
        
        # node_out_feat: [B, hidden, N, 1] -> expand dim 3 to N
        node_out_expanded = node_out_feat.expand(-1, -1, -1, N)
        
        # === C. 记忆更新 ===
        # 同样使用 LogSumExp 聚合全局特征
        global_max = torch.logsumexp(h_edge.view(B, self.hidden_dim, -1) * beta, dim=2) / beta
        h_next = self.gru(global_max, h_prev)
        h_next_expanded = h_next.view(B, -1, 1, 1).expand(-1, -1, N, N)
        
        # === D. 请求意图编码 (Request Encoding) ===
        src_idx = context[:, 0].long() # [B]
        dst_idx = context[:, 1].long() # [B]
        
        # 生成 Source Mask: 标记源点所在的行 (u == src)
        # F.one_hot 生成 [B, N], view -> [B, 1, N, 1], expand -> [B, 1, N, N]
        src_mask = F.one_hot(src_idx, num_classes=N).view(B, 1, N, 1).expand(-1, -1, -1, N).float()
        
        # 生成 Dest Mask: 标记终点所在的列 (v == dst)
        dst_mask = F.one_hot(dst_idx, num_classes=N).view(B, 1, 1, N).expand(-1, -1, N, -1).float()
        
        # 标量特征 (Traffic, Protocol)
        traffic = context[:, 2].view(B, 1, 1, 1).expand(-1, -1, N, N).float()
        protocol = context[:, 3].view(B, 1, 1, 1).expand(-1, -1, N, N).float()
        
        # === E. 特征拼接与预测 ===
        # 拼接所有信息: 
        # [Edge本身, 到达u的能力, 离开v的能力, 历史记忆, 是不是从Src发出的, 是不是去Dst的, 流量, 协议]
        combined = torch.cat([
            h_edge, 
            node_in_expanded, 
            node_out_expanded, 
            h_next_expanded, 
            src_mask, 
            dst_mask, 
            traffic, 
            protocol
        ], dim=1)
        
        mu = self.scorer(combined).squeeze(1) # [B, N, N]
        
        # 数值稳定性处理
        # mu = torch.log1p(torch.exp(-mu)) + 1e-6 # Softplus
        tau = 0.2  # 0.1~1.0 之间试
        mu = F.softplus(-mu) + 1e-9
        
        # 兼容性 std
        std = torch.ones_like(mu) * 0.1
        
        return mu, std, h_next

class PolicyGradientAgent:
    def __init__(self, num_nodes=12, lr=1e-4, is_bypass=True, device='cuda', 
                 num_global_features=7, num_wl_features=3, num_wavelengths=10):
        self.num_nodes = num_nodes
        self.device = device
        self.is_bypass = is_bypass
        
        # 初始化双流 GNN 模型
        self.model = QKDGraphNet(
            num_global_features=num_global_features,
            num_wl_features=num_wl_features,
            num_wavelengths=num_wavelengths,
            actual_nodes=num_nodes,
            is_bypass=is_bypass
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 批量训练缓存
        self.batch_log_probs = [] # 存储多个 episode 的 log_probs
        self.batch_entropies = [] # 存储多个 episode 的 entropies
        self.batch_rewards = []   # 存储多个 episode 的总 reward
        
        self.reward_history = []
        
        # 运行时状态
        self.h_state = None
        self.last_action_t = None
        
        # 当前 episode 的临时缓存
        self.current_episode_log_probs = []
        self.current_episode_entropies = []

    def reset_episode(self):
        # hidden_dim 默认为 8，需与 QKDGraphNet 保持一致
        self.h_state = torch.zeros(1, 8).to(self.device)
        self.last_action_t = None
        # 清除当前 episode 的临时缓存，但不清除 batch 缓存
        del self.current_episode_log_probs[:]
        del self.current_episode_entropies[:]

    def select_action(self, state_matrices, context, train=True):
        self.model.train() if train else self.model.eval()
        
        # 解包双流输入
        # state_matrices 是 tuple: (x_global_np, x_wl_np)
        x_global_np, x_wl_np = state_matrices
        
        # 转换为 Tensor 并增加 batch 维度
        x_global_t = torch.FloatTensor(x_global_np).unsqueeze(0).to(self.device)
        x_wl_t = torch.FloatTensor(x_wl_np).unsqueeze(0).to(self.device)
        context_t = torch.FloatTensor(context).unsqueeze(0).to(self.device)
        
        mu, std, h_next = self.model(x_global_t, x_wl_t, context_t, self.last_action_t, self.h_state)
        self.h_state = h_next.detach()
        
        dist = torch.distributions.Normal(mu, std)
        if train:
            action = dist.sample()
            self.current_episode_log_probs.append(dist.log_prob(action).sum())
            self.current_episode_entropies.append(dist.entropy().sum())
            
            self.last_action_t = action.view(1, -1).detach()
            return action.squeeze().detach().cpu().numpy()
        else:
            return mu.squeeze().detach().cpu().numpy()

    def end_episode(self, total_reward):
        """记录当前 episode 的数据到 batch 缓存中"""
        self.batch_log_probs.append(list(self.current_episode_log_probs))
        self.batch_entropies.append(list(self.current_episode_entropies))
        self.batch_rewards.append(total_reward)
        self.reward_history.append(total_reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)

    def update(self, entropy_coef=0.005):
        """执行批量更新 (Batch-based Symmetric Rank)"""
        if not self.batch_rewards:
            return 0.0
            
        self.model.train()
        
        # --- 改动 A: Batch 内对称 Rank ---
        rewards = np.array(self.batch_rewards)
        if len(rewards) > 1:
            # 计算排名 (0 到 batch_size-1)
            ranks = np.argsort(np.argsort(rewards)) 
            # 映射到 [-0.5, 0.5]
            advantages = (ranks / (len(rewards) - 1)) - 0.5
        else:
            advantages = np.array([0.0])
            
        total_loss = []
        
        for ep_idx, advantage in enumerate(advantages):
            ep_log_probs = self.batch_log_probs[ep_idx]
            ep_entropies = self.batch_entropies[ep_idx]
            
            for log_prob, entropy in zip(ep_log_probs, ep_entropies):
                scaled_log_prob = log_prob / (self.num_nodes * self.num_nodes)
                # 使用减半后的 entropy_coef (改动 B)
                total_loss.append(-scaled_log_prob * advantage - entropy_coef * entropy / (self.num_nodes * self.num_nodes))
        
        self.optimizer.zero_grad()
        loss = torch.stack(total_loss).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # 清除 batch 缓存
        del self.batch_log_probs[:]
        del self.batch_entropies[:]
        del self.batch_rewards[:]
        
        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
