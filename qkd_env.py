import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import copy
import config
import networkx as nx
from utils.Network import Network
from utils.tools import (
    clear_path_cache, 
    build_auxiliary_graph_with_weights, 
    extract_feature_matrices_from_graph
)
from main import find_min_weight_path_with_relay, serve_traffic

class QKDEnv(gym.Env):
    def __init__(self, map_name="Paris", protocol="BB84", detector="SNSPD", traffic_mid="Low", wavelength_list=None, request_list=None, is_bypass=False):
        super(QKDEnv, self).__init__()
        self.map_name = map_name
        self.protocol = protocol
        self.detector = detector
        self.traffic_mid = traffic_mid
        self.is_bypass = is_bypass # 保存实例级配置
        self.wavelength_list = wavelength_list if wavelength_list is not None else list(range(1, 11))
        self.provided_request_list = request_list
        
        # 预先获取节点数以对齐 NN 维度
        self.network = Network(map_name=self.map_name, wavelength_list=self.wavelength_list, 
                               protocol=self.protocol, receiver=self.detector)
        self.num_nodes = len(self.network.topology.nodes())
        
        # === 预计算归一化统计量 ===
        max_dist = 0.0
        for u, v, d in self.network.topology.edges(data='distance'):
            if d > max_dist: max_dist = d
        if max_dist == 0: max_dist = 200.0
        
        # 预估最大流量 (Log scale base)
        # 假设最大可能流量是 5000 * 链路数? 或者直接给一个宽松上界
        # 这里设为 20.0 (对应 exp(20) ≈ 4.8亿，足够大)
        max_traffic_log = 20.0
        
        # 动态计算最大功耗 (基于 Detector 类型)
        # 估算公式：IceBox + Detector * N + Laser * N + Others
        # 假设一条链路上可能有 5 个 Detector
        if self.detector == 'SNSPD':
            # IceBox (3000) + Detector (200?) + Laser (100)
            base_max_power = 4000.0
        elif self.detector == 'APD':
            # No IceBox. Detector (500?) + Laser (100)
            base_max_power = 800.0
        elif self.detector == 'ThorlabsPDB': # CV-QKD
            # No IceBox. Detector (low power) + Laser + DSP (High)
            base_max_power = 1000.0
        else:
            base_max_power = 2000.0
            
        self.max_stats = {
            'max_dist': max_dist * 1.5,
            'max_power': base_max_power,
            'max_wl': float(self.network.num_wavelength),
            'max_traffic': max_traffic_log
        }
        
        # Observation: 100 feature planes
        self.observation_space = spaces.Box(low=0, high=1e9, shape=(100, self.num_nodes, self.num_nodes), dtype=np.float32)
        # Action: ONE weight plane (N x N)
        self.action_space = spaces.Box(low=0, high=1e9, shape=(self.num_nodes, self.num_nodes), dtype=np.float32)
        
        self.node_to_idx = None
        self.requests = None
        self.topology = None
        self.physical_topology = None
        self.served_request = {}
        
        # 实例级缓存 (Instance-level Cache)
        # 替代 utils.tools 的全局缓存，避免跨进程/跨实验污染
        self.path_cache = {}
        self.ld_pos_cache = {}
        
        # 不在 init 时自动 reset，推迟到显式调用时
        # self.reset() 

    def reset(self):
        # 清空实例级缓存
        self.path_cache.clear()
        self.ld_pos_cache.clear()
        
        # 兼容旧代码：虽然不再依赖全局缓存，但为了保险起见，
        # 如果有其他地方还在用 clear_path_cache，这里调用一下也没坏处（虽然它只清空空字典）
        clear_path_cache()
        
        config.protocol = self.protocol
        config.detector = self.detector
        config.bypass = self.is_bypass # 强制使用实例配置覆盖全局配置
        config.key_rate_list = {} 
        
        self.network = Network(map_name=self.map_name, wavelength_list=self.wavelength_list, 
                               protocol=self.protocol, receiver=self.detector)
        self.topology = self.network.topology
        self.physical_topology = self.network.physical_topology
        
        # Node mapping
        nodes = list(self.topology.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Requests
        if self.provided_request_list is not None:
            self.requests = copy.deepcopy(self.provided_request_list)
        else:
            from utils.traffic_generater import gen_traffic_matrix
            prinr("load new traffic")
            self.requests = gen_traffic_matrix(self.traffic_mid, self.map_name, self.wavelength_list, self.protocol, self.detector)
        
        self.current_req_idx = 0
        self.total_power = 0
        self.total_occupied_wls = 0
        self.total_path_cost = 0.0 # 新增：累计路径权重和
        self.success_count = 0
        self.total_component_power = {'source': 0, 'detector': 0, 'other': 0, 'ice_box': 0}
        self.served_request = {}
        
        return self._get_obs()

    def _get_obs(self):
        if self.current_req_idx >= len(self.requests):
            # 返回空状态，匹配新的双流结构
            # Global: [7, N, N], Wavelength: [50, N, N]
            empty_global = np.zeros((7, self.num_nodes, self.num_nodes), dtype=np.float32)
            empty_wl = np.zeros((50, self.num_nodes, self.num_nodes), dtype=np.float32)
            return (empty_global, empty_wl), np.zeros(4)
            
        req = self.requests[self.current_req_idx]
        traffic = req[3]
        
        # 1. 预先构建辅助图（不带权重），用于提取物理特征
        dummy_weights = np.zeros((self.num_nodes, self.num_nodes))
        
        self.current_aux_graph = build_auxiliary_graph_with_weights(
            self.topology, self.wavelength_list, traffic, self.physical_topology, 
            config.key_rate_list, self.served_request, len(self.requests) - self.current_req_idx,
            dummy_weights, self.node_to_idx,
            path_cache=self.path_cache, ld_pos_cache=self.ld_pos_cache
        )
        
        # 1.5 构建未来流量矩阵 (Future Traffic Matrix)
        # 统计剩余所有请求的流量分布，为 NN 提供前瞻性
        future_traffic_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        if self.current_req_idx + 1 < len(self.requests):
            remaining_requests = self.requests[self.current_req_idx + 1 :]
            for r in remaining_requests:
                # r: (id, src, dst, traffic)
                s_node, d_node, t_val = r[1], r[2], r[3]
                if s_node in self.node_to_idx and d_node in self.node_to_idx:
                    u, v = self.node_to_idx[s_node], self.node_to_idx[d_node]
                    future_traffic_matrix[u, v] += t_val
        
        # 2. 直接从辅助图中提取特征张量 (传入 future_traffic_matrix)
        # 返回 (global_tensor, wl_tensor)
        x_global, x_wl = extract_feature_matrices_from_graph(
            self.current_aux_graph, self.node_to_idx, self.num_nodes, self.wavelength_list,
            remain_request_matrix=future_traffic_matrix,
            max_stats=self.max_stats,
            topology=self.topology # 传入物理拓扑
        )
        
        # Context vector
        src_idx = self.node_to_idx[req[1]]
        dst_idx = self.node_to_idx[req[2]]
        prot_idx = 1 if self.protocol == "BB84" else 0
        context = np.array([
            src_idx / self.num_nodes, 
            dst_idx / self.num_nodes, 
            np.log1p(traffic) / 15.0,
            prot_idx
        ], dtype=np.float32)
        
        return (x_global, x_wl), context

    def step(self, action_weights):
        # action_weights: [N, N]
        if self.current_req_idx >= len(self.requests):
            return self._get_obs(), 0, True, {}
            
        req = self.requests[self.current_req_idx]
        src, dst, traffic = req[1], req[2], req[3]
        
        # 3. 使用 NN 输出的真实权重更新辅助图的权重属性
        # 注意：不需要重新 build 辅助图，只需修改已有图的边权重
        for u, v, k, data in self.current_aux_graph.edges(keys=True, data=True):
            u_idx, v_idx = self.node_to_idx[u], self.node_to_idx[v]
            nn_weight = float(action_weights[u_idx, v_idx])
            # 直接使用 NN 输出的权重 (0-1)
            data['weight'] = nn_weight
            
        # 2. 寻找路径
        # 使用概率采样 (Soft Sampling) 来增强探索
        # 这允许优化器通过微调权重来平滑地改变路径选择概率
        best_path = find_min_weight_path_with_relay(self.current_aux_graph, src, dst, probabilistic=False)
        
        if best_path:
            self.total_path_cost += best_path[4] # 修正索引：best_path[4] 才是 weight，[0] 是字符串
            
            # 3. Serve 并获取真实功耗和频谱变化
            occupied_wls, power_dict = serve_traffic(self.topology, self.current_aux_graph, best_path[1], traffic, None, self.served_request)
            self.total_power += sum(power_dict.values())
            self.total_occupied_wls += occupied_wls
            self.success_count += 1
            # 累加各分量功耗
            for k in self.total_component_power:
                self.total_component_power[k] += power_dict.get(k, 0)
        else:
            # 请求失败惩罚：暂时不加到 total_power，最后结算时统一加到 Avg 上
            # penalty_power = 10000.0
            # self.total_power += penalty_power
            pass
            
        self.current_req_idx += 1
        done = self.current_req_idx >= len(self.requests)
        
        # 最终奖励计算：
        if done:
            num_reqs = len(self.requests)
            num_fails = num_reqs - self.success_count
            
            # 基础平均功耗 (只算成功的，如果没有成功的则为0)
            if self.success_count > 0:
                base_avg_power = self.total_power / self.success_count # 注意：分母是 success_count 还是 num_reqs？
                # 如果分母是 num_reqs，那失败的就被算作 0 了，这会拉低基准。
                # 如果分母是 success_count，那这是“成功请求的平均功耗”。
                # 为了惩罚的连贯性，我们通常希望它是 Total Power / Num Reqs。
                # 但你要求的是 "直接在平均功耗上加"。
                
                # 方案 A: Avg = (Total Success Power) / N + (Fail Count * 10000)
                base_avg_power = self.total_power / num_reqs
            else:
                base_avg_power = 0.0
            
            # 加上超级惩罚
            avg_power = base_avg_power + num_fails * 10000.0
            
            spec_occ = self.total_occupied_wls / self.network.num_wavelength if self.network.num_wavelength > 0 else 0
            
            # 引入“潜在热能梯度” (Potential Heat Penalty)
            # ... (保持不变)
            # 即使功耗没变，如果热量分布更均匀，距离 3000W 跳变点更远，也给与奖励
            heat_risk_penalty = 0
            for node in self.topology.nodes():
                num_det = self.topology.nodes[node].get('num_detector', 0)
                # 计算该节点距离下一个冰机容量上限的剩余空间
                # 剩余空间越小，风险越高
                cap_used = num_det % config.ice_box_capacity
                if cap_used > 0:
                    # 越接近 capacity，惩罚越大，这能引导 NN 发现“热量均衡”的隐藏规律
                    heat_risk_penalty += (cap_used / config.ice_box_capacity) ** 2
            
            # 综合奖励：Power (主目标) + 频谱占用 (次要) + 热能风险 (引导信号)
            reward = -(avg_power / 5.0 + spec_occ * 0.5 + heat_risk_penalty * 2.0) 
            
            # 汇总各分量平均功耗
            info = {
                'success_rate': self.success_count / len(self.requests),
                'avg_power': avg_power, # 统一使用含惩罚的功耗
                'spec_occ': spec_occ, # 保持原样，虽然定义有点奇怪
                'total_wls': self.total_occupied_wls, # 新增：绝对波长占用数，用于日志对比
                'heat_risk': heat_risk_penalty,
                'path_cost': self.total_path_cost, # 传递给 Optimizer
                'source_p': self.total_component_power['source'] / len(self.requests),
                'detector_p': self.total_component_power['detector'] / len(self.requests),
                'other_p': self.total_component_power['other'] / len(self.requests),
                'ice_box_p': self.total_component_power['ice_box'] / len(self.requests)
            }
        else:
            reward = 0 # 中间步骤不给奖励
            info = {}
        
        return self._get_obs(), reward, done, info
