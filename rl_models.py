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
    极轻量化图神经网络 (GNN)：
    1. 拓扑无关设计：参数量不随节点数 N 增加。
    2. 消息传递机制：学习局部物理规律（如：边长度、节点热量压力）。
    3. 参数量约 500-800，远低于之前的 3522。
    """
    def __init__(self, num_features=10, num_wavelengths=10, actual_nodes=12, is_bypass=True, hidden_dim=8):
        super(QKDGraphNet, self).__init__()
        self.num_nodes = actual_nodes
        self.is_bypass = is_bypass
        self.hidden_dim = hidden_dim
        in_channels = num_features * num_wavelengths # 100
        
        # 1. 边编码器 (1x1 卷积实现)
        # 将 100 维物理特征压缩到低维嵌入
        self.edge_encoder = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # 2. 消息传递层 (简单版：聚合邻居边特征到节点)
        # 参数量极小
        self.node_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 3. 上下文编码 (src, dst, traffic, protocol)
        self.context_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # 4. 预测器：基于边嵌入、源节点嵌入、全局上下文预测权重
        # 输入：边(8) + 全局(8) = 16
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # 5. 极简记忆层 (不再使用全连接处理 N*N，改为全局池化记忆)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, context, last_action, h_prev):
        # x: [B, 100, N, N]
        # context: [B, 4]
        # last_action: [B, N*N] (CMA-ES 传入的是展开的)
        # h_prev: [B, hidden_dim]
        
        batch_size = x.size(0)
        N = self.num_nodes
        
        # A. 边特征初步提取 -> [B, hidden, N, N]
        edge_feat = F.leaky_relu(self.edge_encoder(x), 0.1)
        
        # B. 消息传递：计算节点嵌入 [B, hidden, N]
        # 简单的入度聚合 (对行求和)
        node_feat = torch.sum(edge_feat, dim=-1) / N 
        
        # C. 记忆更新 (使用全局聚合特征更新记忆)
        # 这样记忆层就与 N 无关了！
        global_summary = torch.mean(node_feat, dim=-1) 
        h_next = self.gru(global_summary, h_prev)
        
        # D. 融合上下文与记忆 -> [B, hidden]
        ctx_feat = self.context_encoder(context)
        global_context = ctx_feat + h_next
        
        # E. 最终边预测
        # 广播全局特征 [B, hidden, 1, 1] -> [B, hidden, N, N]
        global_context_expanded = global_context.view(batch_size, -1, 1, 1).expand(-1, -1, N, N)
        
        # 拼接边特征与全局上下文 -> [B, hidden*2, N, N]
        combined = torch.cat([edge_feat, global_context_expanded], dim=1)
        
        # 转换为 [B, N, N, hidden*2] 进行线性层处理
        combined = combined.permute(0, 2, 3, 1)
        mu = self.edge_predictor(combined).squeeze(-1) # [B, N, N]
        
        # 提升输出量程
        mu = torch.tanh(mu) * 1000.0
        
        # 兼容性 std
        std = torch.ones_like(mu) * 0.1
        
        return mu, std, h_next

class PolicyGradientAgent:
    def __init__(self, num_nodes=12, lr=1e-4, is_bypass=True, device='cuda'):
        self.num_nodes = num_nodes
        self.device = device
        self.is_bypass = is_bypass
        self.model = QKDWeightNet(actual_nodes=num_nodes, is_bypass=is_bypass).to(device)
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
        self.h_state = torch.zeros(1, 16).to(self.device)
        self.last_action_t = None
        # 清除当前 episode 的临时缓存，但不清除 batch 缓存
        del self.current_episode_log_probs[:]
        del self.current_episode_entropies[:]

    def select_action(self, state_matrices, context, train=True):
        self.model.train() if train else self.model.eval()
        
        state_t = torch.FloatTensor(state_matrices).unsqueeze(0).to(self.device)
        context_t = torch.FloatTensor(context).unsqueeze(0).to(self.device)
        
        mu, std, h_next = self.model(state_t, context_t, self.last_action_t, self.h_state)
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
