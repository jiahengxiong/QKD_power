import gym
from gym import spaces
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
    def __init__(self, map_name="Paris", protocol="BB84", detector="SNSPD", traffic_mid="Low", wavelength_list=None, request_list=None):
        super(QKDEnv, self).__init__()
        self.map_name = map_name
        self.protocol = protocol
        self.detector = detector
        self.traffic_mid = traffic_mid
        self.wavelength_list = wavelength_list if wavelength_list is not None else list(range(1, 11))
        self.provided_request_list = request_list
        
        # 预先获取节点数以对齐 NN 维度
        self.network = Network(map_name=self.map_name, wavelength_list=self.wavelength_list, 
                               protocol=self.protocol, receiver=self.detector)
        self.num_nodes = len(self.network.topology.nodes())
        
        # Observation: 100 feature planes
        self.observation_space = spaces.Box(low=0, high=1e9, shape=(100, self.num_nodes, self.num_nodes), dtype=np.float32)
        # Action: ONE weight plane (N x N)
        self.action_space = spaces.Box(low=0, high=1e9, shape=(self.num_nodes, self.num_nodes), dtype=np.float32)
        
        self.node_to_idx = None
        self.requests = None
        self.topology = None
        self.physical_topology = None
        self.served_request = {}
        
        self.reset()

    def reset(self):
        clear_path_cache()
        config.protocol = self.protocol
        config.detector = self.detector
        config.bypass = config.bypass 
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
            self.requests = gen_traffic_matrix(self.traffic_mid, self.map_name, self.wavelength_list, self.protocol, self.detector)
        
        self.current_req_idx = 0
        self.total_power = 0
        self.total_occupied_wls = 0
        self.total_component_power = {'source': 0, 'detector': 0, 'other': 0, 'ice_box': 0}
        self.served_request = {}
        
        return self._get_obs()

    def _get_obs(self):
        if self.current_req_idx >= len(self.requests):
            return np.zeros((100, self.num_nodes, self.num_nodes), dtype=np.float32), np.zeros(4)
            
        req = self.requests[self.current_req_idx]
        traffic = req[3]
        
        # 1. 预先构建辅助图（不带权重），用于提取物理特征
        dummy_weights = np.zeros((self.num_nodes, self.num_nodes))
        
        self.current_aux_graph = build_auxiliary_graph_with_weights(
            self.topology, self.wavelength_list, traffic, self.physical_topology, 
            config.key_rate_list, self.served_request, len(self.requests) - self.current_req_idx,
            dummy_weights, self.node_to_idx
        )
        
        # 2. 直接从辅助图中提取特征张量
        state_matrices = extract_feature_matrices_from_graph(
            self.current_aux_graph, self.node_to_idx, self.num_nodes, self.wavelength_list
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
        
        return state_matrices, context

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
            nn_virtual_cost = float(action_weights[u_idx, v_idx])
            # 这里的 Power 是在 build 时已经算好存进去的
            data['weight'] = max(1e-6, data['power'] + nn_virtual_cost)
            
        # 2. 寻找路径
        best_path = find_min_weight_path_with_relay(self.current_aux_graph, src, dst)
        
        if best_path:
            # 3. Serve 并获取真实功耗和频谱变化
            occupied_wls, power_dict = serve_traffic(self.topology, self.current_aux_graph, best_path[1], traffic, None, self.served_request)
            self.total_power += sum(power_dict.values())
            self.total_occupied_wls += occupied_wls
            # 累加各分量功耗
            for k in self.total_component_power:
                self.total_component_power[k] += power_dict.get(k, 0)
            
        self.current_req_idx += 1
        done = self.current_req_idx >= len(self.requests)
        
        # 最终奖励计算：
        if done:
            avg_power = self.total_power / len(self.requests)
            spec_occ = self.total_occupied_wls / self.network.num_wavelength if self.network.num_wavelength > 0 else 0
            
            # 引入“潜在热能梯度” (Potential Heat Penalty)
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
                'avg_power': avg_power, 
                'spec_occ': spec_occ, 
                'heat_risk': heat_risk_penalty,
                'source_p': self.total_component_power['source'] / len(self.requests),
                'detector_p': self.total_component_power['detector'] / len(self.requests),
                'other_p': self.total_component_power['other'] / len(self.requests),
                'ice_box_p': self.total_component_power['ice_box'] / len(self.requests)
            }
        else:
            reward = 0 # 中间步骤不给奖励
            info = {}
        
        return self._get_obs(), reward, done, info
