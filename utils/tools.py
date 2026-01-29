import copy
import gc
import itertools
import uuid
from itertools import product
from multiprocessing import Pool
import math

import config
from QEnergy.studies.key_rate_compute import compute_key_rate
from QEnergy.studies.power_compute import compute_power
from utils.Network import Network
from utils.util import can_connect_path, calculate_distance
import networkx as nx
import random
from utils.custom_algorithm import find_maximin_path
from tqdm import tqdm  # 确保安装了 tqdm: pip install tqdm
from itertools import combinations

# ==========================================
# 路径缓存与永久剪枝逻辑
# ==========================================
_PATH_CACHE = {}

def get_cached_paths(G, src, dst, is_bypass):
    key = (src, dst, is_bypass)
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]
    
    if is_bypass:
        try:
            # 预生成前 100 条最短物理路径，确保辅助图有足够的备选边
            gen = nx.shortest_simple_paths(G, src, dst, weight='distance')
            paths = list(itertools.islice(gen, 100))
            _PATH_CACHE[key] = paths
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            _PATH_CACHE[key] = []
            return []
    else:
        try:
            # 非 Bypass 模式只看直连
            paths = list(nx.all_simple_paths(G, source=src, target=dst, cutoff=1))
            _PATH_CACHE[key] = paths
            return paths
        except:
            _PATH_CACHE[key] = []
            return []

def clear_path_cache():
    """清空全局路径缓存"""
    global _PATH_CACHE
    _PATH_CACHE = {}

# ==========================================
# 核心构建与辅助函数 (保持不变或微调)
# ==========================================

def build_network_slice(wavelength_list, topology, traffic):
    network_slice = {}
    for wavelength in wavelength_list:
        virtual_graph = copy.deepcopy(topology)
        wavelength_slice = nx.Graph()
        edges = list(virtual_graph.edges(keys=True, data=True))
        for edge in edges:
            src = edge[0]
            dst = edge[1]
            key = edge[2]
            data = edge[3]
            if data['wavelength'] == wavelength and data['free_capacity'] > 0:
                wavelength_slice.add_edge(src, dst, key=key, **data)
        for node in wavelength_slice.nodes:
            data = topology.nodes[node]
            # print(node, data)
            wavelength_slice.nodes[node].update(data)
        network_slice[wavelength] = wavelength_slice
        del virtual_graph
    return network_slice


def combine_wavelength_slices(wavelength_list, topology, traffic, network_slice):
    # combined_capacity = {}
    # for u, v, _ in topology.edges(keys=True):
    #     combined_capacity[(u, v)] = 0
    #     combined_capacity[(v, u)] = 0

    # selected_wavelengths = []
    # last_selected_idx = None

    # for i, wavelength in enumerate(wavelength_list):
    #     network_wavelength = network_slice[wavelength]
    #     free_capacity_list = [network_wavelength.edges[edge]['free_capacity']
    #                           for edge in network_wavelength.edges]

    #     if max(free_capacity_list, default=0) <= 0:
    #         continue

    #     for edge in network_wavelength.edges:
    #         cap = network_wavelength.edges[edge].get('free_capacity', 0)
    #         combined_capacity[edge] += cap
    #         combined_capacity[(edge[1], edge[0])] += cap

    #     selected_wavelengths.append(wavelength)
    #     last_selected_idx = i

    #     if min(combined_capacity.values(), default=0) >= traffic:
    #         break

    # combination_wls = list(selected_wavelengths)
    # if last_selected_idx is not None and last_selected_idx + 2 < len(wavelength_list):
    #     extra_wl = wavelength_list[last_selected_idx + 2]
    #     combination_wls.append(extra_wl)

    all_combinations = []

    n = len(wavelength_list)
    for r in range(1, n + 1):
        for comb in combinations(wavelength_list, r):
            all_combinations.append(list(comb))
    return all_combinations


def find_min_free_capacity(wavelength_slice, path):
    try:
        free_capacities = [
            wavelength_slice.edges[path[i], path[i + 1]]['free_capacity']
            for i in range(len(path) - 1)
        ]
        return min(free_capacities)
    except KeyError:
        # 如果路径上的边在当前波长切片中不存在
        return 0


import copy

def find_laser_detector_position(wavelength_slice, path, wavelength):
    """
    优化版：
    1. 解决 unhashable type: 'list' 报错 (通过 to_tuple 转换)
    2. 使用 Set 替代 List 进行 O(1) 查找
    3. 移除 deepcopy 提升速度
    4. 预计算路径覆盖状态，避免重复切片
    """
    
    # --- 辅助函数：将 list 递归转为 tuple，使其可哈希 ---
    def to_tuple(obj):
        if isinstance(obj, list):
            return tuple(to_tuple(x) for x in obj)
        return obj

    # --- 1. 建立路径索引映射 (优化 path.index) ---
    # key 使用 tuple 格式，value 是索引
    path_map = {to_tuple(node): i for i, node in enumerate(path)}
    path_len = len(path)

    # --- 2. 识别现有的 Laser-Detector ---
    laser_detector = []
    
    # 预先获取 path 上的节点数据，减少 graph 访问
    # 注意：这里假设 graph 的 key 与 path 中的 node 是一致的
    # 如果 graph key 是 tuple 而 path 是 list，需要注意匹配。这里沿用原逻辑直接取。
    nodes_data = {}
    for node in path:
        # 为了防止 node 本身是 list 导致无法作为 key 访问 graph (如果 graph key 是 list 的话)
        # 通常 networkx 的 node 需要是 hashable 的，所以这里直接用 node 应该没问题
        # 如果你的 graph node 真的是 list，那 networkx 本身就会报错。
        # 假设 node 是 hashable 或者 graph 能处理。
        if node in wavelength_slice.nodes:
            nodes_data[node] = wavelength_slice.nodes[node]

    for i in range(path_len - 1):
        node = path[i]
        if node not in nodes_data: continue
        
        node_data = nodes_data[node]
        if 'laser' not in node_data or wavelength not in node_data['laser']:
            continue
            
        for detector_list in node_data['laser'][wavelength]:
            # 检查 detector_list 中的点是否都在 path 中，且顺序正确
            try:
                # 使用 path_map 快速查找索引
                # detector_list 里的元素如果是 list，必须转 tuple 才能查 path_map
                current_indices = []
                for item in detector_list:
                    item_key = to_tuple(item)
                    if item_key in path_map:
                        current_indices.append(path_map[item_key])
                    else:
                        raise KeyError # 只要有一个点不在 path 里，就跳过
                
                # 检查顺序是否递增
                if current_indices == sorted(current_indices):
                    laser_detector.append([node, detector_list[-1]])
            except KeyError:
                continue

    # 快速检查连接性
    if can_connect_path(path=path, laser_detector=laser_detector):
        return [[None, None]]

    # --- 3. 构建 Bypass Link 集合 (核心修复与优化) ---
    bypass_link_set = set()
    
    for node in path:
        if node not in nodes_data: continue
        node_data = nodes_data[node]

        # 定义一个内部函数来处理 link 添加
        def add_links(links):
            for i in range(len(links)-1):
                u = links[i]
                v = links[i+1]
                # 关键：转为 tuple 再存入 set
                bypass_link_set.add((to_tuple(u), to_tuple(v)))

        if 'laser' in node_data and wavelength in node_data['laser']:
            add_links(node_data['laser'][wavelength])
        
        if 'detector' in node_data and wavelength in node_data['detector']:
            add_links(node_data['detector'][wavelength])

    # --- 4. 预计算 Path 上的边是否被覆盖 ---
    # path_edges_covered[k] = True 表示 path[k]->path[k+1] 这段路在 bypass 中
    path_edges_covered = [False] * (path_len - 1)
    
    for k in range(path_len - 1):
        u = path[k]
        v = path[k+1]
        u_t = to_tuple(u)
        v_t = to_tuple(v)
        
        # 检查正向或反向是否在集合中
        if (u_t, v_t) in bypass_link_set or (v_t, u_t) in bypass_link_set:
            path_edges_covered[k] = True

    # --- 5. 生成并筛选 Pairs ---
    laser_detector_position = []
    
    for i in range(path_len):
        for j in range(i + 1, path_len):
            pair = [path[i], path[j]]
            
            # 如果 pair 已经在 laser_detector 中，跳过
            if pair in laser_detector:
                continue
            
            # 核心优化：检查 pair 中间的路径是否被 bypass 覆盖
            # 原逻辑：如果中间有任何一段 link 在 bypass_link 中，则剔除该 pair
            # 现逻辑：检查 path_edges_covered[i:j] 是否包含 True
            is_covered = False
            # 使用 any() 快速判断，不需要遍历切片
            # 切片范围：从节点 i 到 j，对应的边是 i 到 j-1
            if any(path_edges_covered[i:j]):
                is_covered = True
            
            # 只有没被覆盖的 pair 才是候选者
            if not is_covered:
                # --- 6. 最终测试 (移除 deepcopy) ---
                # 直接构造新列表，比 deepcopy 快得多
                possible_laser_detector = laser_detector + [pair]
                
                if can_connect_path(path=path, laser_detector=possible_laser_detector):
                    laser_detector_position.append(pair)
            
    return laser_detector_position


def calculate_keyrate(laser_detector_position, path, G):
    if laser_detector_position['laser'] is not None and laser_detector_position['detector'] is not None:
        laser_position = laser_detector_position['laser']
        detector_position = laser_detector_position['detector']
        laser_index = path.index(laser_position)
        detector_index = path.index(detector_position)
        cover_path = path[laser_index:detector_index + 1]
        distance = 0

        for i in range(len(cover_path) - 1):
            src = cover_path[i]
            dst = cover_path[i + 1]
            edges_data = G.get_edge_data(src, dst)

            if not edges_data:
                raise ValueError(f"No edges between {src} and {dst} in the path")

            if G.is_multigraph():
                first_key = next(iter(edges_data))
                edge_distance = edges_data[first_key].get('distance', 0)
            else:
                edge_distance = edges_data.get('distance', 0)

            distance += edge_distance

        bypass_number = len(cover_path) - 2
        if distance in config.key_rate_list.keys():
            key_rate = config.key_rate_list[distance]
            key_rate = (key_rate * (0.89 ** bypass_number))
        else:
            key_rate = compute_key_rate(protocol=config.protocol, receiver=config.detector, distance=distance)
            config.key_rate_list[distance] = key_rate
            key_rate = (key_rate * (0.89 ** bypass_number))
    else:
        distance = 0.00001
        if distance in config.key_rate_list.keys():
            key_rate = config.key_rate_list[distance]
        else:
            key_rate = compute_key_rate(protocol=config.protocol, receiver=config.detector, distance=0.00001)
            config.key_rate_list[distance] = key_rate

    return key_rate


def calculate_power(laser_detector_position, path, G):
    component_power = {'source': 0, 'detector':0, 'other':0, 'ice_box':0, 'total':0}
    if config.detector == 'SNSPD':
        ice_box_power = 3000
    else:
        ice_box_power = 0
    laser_position = laser_detector_position['laser']
    detector_position = laser_detector_position['detector']
    if laser_position is not None and detector_position is not None:
        distance = calculate_distance(G=G, path=path, start=laser_position, end=detector_position)
        power = compute_power(distance=distance, protocol=config.protocol, receiver=config.detector)
        component_power['source'] = power['source']
        component_power['detector'] = power['detector']
        component_power['other'] = power['other']
        total_power = power['total']
        num_detector = G.nodes[detector_position]['num_detector']
        if num_detector % config.ice_box_capacity == 0:
            total_power = ice_box_power + total_power
            component_power['ice_box'] = ice_box_power
    else:
        total_power = 0
    component_power['total'] = total_power
    return component_power


def Max_capacity(laser_detector, path, G, wavelength, network_slice):
    AG = nx.DiGraph()
    existing_laser_detector = []
    wavelength_slice = network_slice
    for i in range(len(path) - 1):
        node = path[i]
        for detector_list in wavelength_slice.nodes[node]['laser'][wavelength]:
            # 检查所有节点是否都在路径中
            if all(item in path for item in detector_list):
                # 检查方向性
                indices = [path.index(item) for item in detector_list]  # 获取节点在path中的索引
                if indices == sorted(indices):  # 判断索引是否是递增的
                    existing_laser_detector.append([node, detector_list[-1]])  # 添加符合条件的激光检测器
                    AG.add_edge(node, detector_list[-1], capacity=G.nodes[node]['laser_capacity'][wavelength][tuple(detector_list)], detector_list=detector_list)
    if laser_detector[0] is not None:
        laser_index = path.index(laser_detector[0])
        detector_index = path.index(laser_detector[1])
        if laser_index >= detector_index:
            print(f'ERROR !!! Laser is after Detector')
        cover_links = path[laser_index:detector_index + 1]
        AG.add_edge(laser_detector[0], laser_detector[1], capacity=calculate_keyrate(
            laser_detector_position={'laser': laser_detector[0], 'detector': laser_detector[1]}, path=path,
            G=network_slice), detector_list=cover_links, wavelength=wavelength)
    capacity, used_laser_detector_list = find_maximin_path(AG, path[0], path[-1])
    """if laser_detector in used_laser_detector:
        used_laser_detector.remove(laser_detector)"""
    recovery_detector_list = []
    for used_laser_detector in used_laser_detector_list:
        recovery_detector_list.append(AG.edges[used_laser_detector[0], used_laser_detector[1]]['detector_list'])
    del  AG
    

    return capacity, recovery_detector_list




import math
from itertools import product
try:
    import config
except ImportError:
    pass

def calculate_data_auxiliary_edge(G, path, wavelength_combination, wavelength_capacity, laser_detector_position,
                                  traffic, network_slice, remain_num_request, link_future_demand=None, topology=None, node_future_demand=None):
    data = []
    keys = laser_detector_position.keys()
    values = laser_detector_position.values()
    wavelength_laser_detector_list = [dict(zip(keys, combo)) for combo in product(*values)]
    
    # === 获取全局配置 ===
    # 默认为 SNSPD
    detector_type = getattr(config, 'detector', 'SNSPD')
    ice_box_capacity = getattr(config, 'ice_box_capacity', 8)
    unit_cooling_power = getattr(config, 'unit_cooling_power', 3000)

    for wavelength_laser_detector in wavelength_laser_detector_list:
        wavelength_traffic_limitation = {}
        max_traffic = 0
        wavelength_used_laser_detector = {}
        
        # 1. 计算容量
        for wavelength, laser_detector in wavelength_laser_detector.items():
            key_rate, used_laser_detector = Max_capacity(
                laser_detector=laser_detector,
                path=path,
                G=G, 
                wavelength=wavelength, 
                network_slice=network_slice[wavelength]
            )
            wavelength_traffic_limitation[wavelength] = min(wavelength_capacity[wavelength], key_rate)
            max_traffic += wavelength_traffic_limitation[wavelength]
            wavelength_used_laser_detector[wavelength] = used_laser_detector

        # 2. 只有满足流量才计算功耗
        if max_traffic >= traffic:
            source_power = 0
            detector_power = 0 # 仅指探测器自身运作功耗（不含制冷）
            other_power = 0
            
            spectrum = 0
            LD = 0
            used_LD = 0
            node_new_detectors_count = {}
            
            # === 3. 统计频谱与物理参数 ===
            spectrum = 0
            for wavelength, laser_detector in wavelength_laser_detector.items():
                # 调用 calculate_power 获取基础组件功耗
                component_power = calculate_power(
                    laser_detector_position={'laser': laser_detector[0], 'detector': laser_detector[1]}, 
                    path=path,
                    G=network_slice[wavelength]
                )
                
                source_power += component_power['source']
                detector_power += component_power['detector'] 
                other_power += component_power['other']
                
                # 统计 LD 距离
                if laser_detector[0] is not None:
                    LD += (path.index(laser_detector[1]) - path.index(laser_detector[0]) - 1)
                
                used_LD += len(wavelength_used_laser_detector[wavelength])

                # 统计新增探测器位置
                det_node = laser_detector[1]
                if det_node is not None:
                    node_new_detectors_count[det_node] = node_new_detectors_count.get(det_node, 0) + 1

                # 统计物理波长占用 (新增占用)
                for i in range(len(path) - 1):
                    u_p, v_p = path[i], path[i+1]
                    edges_data = G.get_edge_data(u_p, v_p)
                    if not edges_data: continue
                    for edge_key, edge_attrs in edges_data.items():
                        if edge_attrs.get('wavelength') == wavelength:
                            if not edge_attrs.get('occupied', False):
                                spectrum += 1

            # === 4. 计算频谱机会成本 (基于链路热度与占用波长数) ===
            # 大局观：占用波长越多，对未来的阻塞风险越大
            future_spectrum_cost = 0.0
            num_wls = max(1.0, traffic / max_traffic)
            if link_future_demand:
                for i in range(len(path) - 1):
                    u_p, v_p = path[i], path[i+1]
                    link_heat = link_future_demand.get((u_p, v_p), link_future_demand.get((v_p, u_p), 0))
                    # 频谱成本逻辑：占用波长数 * (链路热度^2) * 调节系数
                    # 使用平方项来对冲高热链路的阻塞风险
                    # link_heat 在 main.py 中是 num_wls 的累加，量级在 10~500 之间
                    future_spectrum_cost += num_wls * (link_heat ** 1.5) * 20.0 
            
            # === 4.1 大局观战略引导：惩罚跳过高价值节点的“新开旁路” ===
            strategic_bypass_tax = 0.0
            new_segments = sum(1 for w, ld in wavelength_laser_detector.items() if ld[0] is not None)
            if node_future_demand and new_segments > 0 and len(path) > 2:
                for skipped_node in path[1:-1]:
                    # 节点战略系数已在 main.py 归一化为 0~1
                    skipped_value = node_future_demand.get(skipped_node, 0)
                    # 惩罚量级设为 30W，代表一个普通中继节点的典型功耗
                    # 这样只有当 Bypass 带来的物理节省 > 30W * 节点价值时，才会执行旁路
                    strategic_bypass_tax += skipped_value * 30.0
            
            # === 5. 计算共享冰箱功耗与战略折扣 ===
            real_ice_box_power = 0 
            weighted_ice_box_power = 0 
            
            if detector_type == 'SNSPD':
                for node, new_count in node_new_detectors_count.items():
                    if node in G.nodes:
                        current_num = G.nodes[node].get('num_detector', 0)
                    else:
                        current_num = 0
                    
                    fridges_before = math.ceil(current_num / ice_box_capacity)
                    total_num_after = current_num + new_count
                    fridges_after = math.ceil(total_num_after / ice_box_capacity)
                    
                    marginal_fridges = fridges_after - fridges_before
                    if marginal_fridges > 0:
                        node_real_fridge_power = marginal_fridges * unit_cooling_power
                        real_ice_box_power += node_real_fridge_power
                        
                        # 战略引导：在枢纽节点开冰箱获得 70% 权重折扣 (鼓励在枢纽扩容)
                        node_s_value = node_future_demand.get(node, 0) if node_future_demand else 0
                        discount = 1.0 - (0.7 * node_s_value)
                        weighted_ice_box_power += node_real_fridge_power * discount
            
            # === 6. 汇总与权重计算 (资源消耗平衡模型) ===
            real_total_power = source_power + detector_power + other_power + real_ice_box_power
            
            # A. 基础功耗成本：激光器 + 探测器 + 加权冰箱
            power_cost = source_power + detector_power + other_power + weighted_ice_box_power
            
            # B. 战略折扣：如果在高价值节点安装新硬件，给予硬件成本折扣 (对冲 Bypass 倾向)
            if new_segments > 0 and node_future_demand:
                for node in node_new_detectors_count:
                    s_value = node_future_demand.get(node, 0)
                    # 给予一定的硬件战略返利 (5~10W 级别)
                    power_cost -= (5.0 * s_value)

            # C. 权重合成
            # 最终权重公式：功耗成本 + 频谱惩罚 + 旁路税
            # 确保所有项都在 Watts 级 (1~5000)
            raw_weight = (
                power_cost 
                + (future_spectrum_cost / 1e3) # 调整缩放系数，增强频谱保护
                + strategic_bypass_tax
            )
            weight = max(1.0, raw_weight)

            data.append({
                'power': real_total_power, # 返回真实功耗，修复统计 Bug
                'source_power': source_power,
                'detector_power': detector_power,
                'other_power': other_power,
                'ice_box_power': real_ice_box_power, 
                'path': path, 
                'laser_detector_position': wavelength_laser_detector,
                'wavelength_traffic': wavelength_traffic_limitation,
                'weight': weight,
                'wavelength_list': wavelength_combination,
                'transverse_laser_detector': wavelength_used_laser_detector
            })

    return data


def get_k_shortest_paths(graph, src, dst, k=5, weight="distance"):
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        G = nx.Graph() 
        for u, v, data in graph.edges(data=True):
            w = data.get(weight, 1.0)
            if G.has_edge(u, v):
                if w < G[u][v].get(weight, math.inf):
                    G[u][v][weight] = w
            else:
                G.add_edge(u, v, **{weight: w})
    else:
        G = graph

    try:
        gen = nx.shortest_simple_paths(G, source=src, target=dst, weight=weight)
        return list(itertools.islice(gen, k))
    except nx.NetworkXNoPath:
        return []

def check_path_coverage(path_links, wavelength_covered_links):
    norm = lambda e: frozenset(e)
    path_set = {norm(edge) for edge in path_links}
    wl_set   = {norm(edge) for edge in wavelength_covered_links}

    if path_set.isdisjoint(wl_set):
        return True
    if wl_set.issubset(path_set):
        return True
    return False

def check_path_validity_for_request(path, wavelength_combination, served_request):
    """
    检查单条路径是否满足当前波长组合的 served_request 约束
    替代原有的 filter_paths
    """
    path_links = list(zip(path, path[1:]))
    
    for wavelength in wavelength_combination:
        if wavelength not in list(served_request.keys()):
            continue

        wavelength_covered_links_list = served_request[wavelength]
        for wavelength_covered_links in wavelength_covered_links_list:
            if not check_path_coverage(path_links, wavelength_covered_links):
                return False
    return True

# ==========================================
# 新增：轻量级临时图构建
# ==========================================

def build_temp_graph_for_path(topology, path, wavelength_combinations):
    """
    只构建路径相关的临时图，用于 calculate_data_auxiliary_edge 计算。
    避免复制整个大图。
    """
    temp_G = nx.MultiDiGraph()
    
    # 只需要添加路径上的节点
    for node in path:
        if node in topology:
            temp_G.add_node(node, **topology.nodes[node])
            
    # 只需要添加路径上的边
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if topology.has_edge(u, v):
            # 获取两点间所有边
            edges = topology[u][v]
            for key, data in edges.items():
                # 只保留当前波长组合内的边
                if data.get('wavelength') in wavelength_combinations:
                    temp_G.add_edge(u, v, key=key, **data)
                    
    return temp_G

# ==========================================
# 核心重构：build_auxiliary_graph
# ==========================================

def build_auxiliary_graph(topology, wavelength_list, traffic, physical_topology, shared_key_rate_list, served_request,remain_num_request, link_future_demand=None, node_future_demand=None):
    """
    终极修正版 V8：DFS 回溯 + 智能剪枝 (Backtracking with Pruning)
    新增：link_future_demand (链路热力图), node_future_demand (节点热力图)
    """
    auxiliary_graph = nx.MultiDiGraph()
    
    # 1. 预处理
    network_slice = build_network_slice(wavelength_list=wavelength_list, topology=topology, traffic=traffic)
    config.key_rate_list = shared_key_rate_list 
    
    for node in topology.nodes():
        auxiliary_graph.add_node(node)

    # 2. 生成所有节点对
    nodes = list(topology.nodes())
    pairs = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                pairs.append((nodes[i], nodes[j]))
    
    # 3. 遍历处理
    for src, dst in pairs:
        # --- 3.1 获取路径缓存引用 ---
        cache_key = (src, dst, config.bypass)
        path_list = get_cached_paths(physical_topology, src, dst, config.bypass)
        if not path_list: continue
        
        # 使用副本进行迭代，以便在循环中安全地从原缓存列表中删除失效路径
        for path in list(path_list):
            # === 步骤 A: 收集并筛选候选波长 ===
            candidates = []
            for wavelength in wavelength_list:
                wl_slice = network_slice[wavelength]
                
                # 1. 容量筛选
                min_cap = find_min_free_capacity(wavelength_slice=wl_slice, path=path)
                if min_cap <= 0:
                    continue 

                # 2. 硬件筛选
                positions = find_laser_detector_position(
                    wavelength_slice=wl_slice, path=path, wavelength=wavelength
                )
                if len(positions) < 1:
                    continue 
                
                candidates.append({
                    'wl': wavelength,
                    'cap': min_cap, 
                    'pos': positions
                })
            
            # === 判定 1: 基础可行性探测 ===
            if not candidates:
                # 此路径连一个空闲波长都凑不齐，彻底不可行，永久剔除
                if path in _PATH_CACHE[cache_key]:
                    _PATH_CACHE[cache_key].remove(path)
                continue

            # === 步骤 B: 流量承载能力探测 ===
            total_possible_cap = sum(c['cap'] for c in candidates)
            if total_possible_cap < traffic:
                # 即使全选所有可用波长，容量也达不到 traffic 要求，永久剔除
                if path in _PATH_CACHE[cache_key]:
                    _PATH_CACHE[cache_key].remove(path)
                continue

            # === 步骤 C: DFS 深度组合探测 ===
            path_found_flag = False
            candidates.sort(key=lambda x: x['cap'], reverse=False)

            def dfs_find_valid_set(idx, current_theoretical_sum, current_wls, current_pos_dict, current_cap_dict):
                nonlocal path_found_flag
                if path_found_flag: return 
                
                if current_theoretical_sum >= traffic:
                    # 验证波长干扰约束 (served_request)
                    if not check_path_validity_for_request(path, current_wls, served_request):
                        return 
                    
                    temp_G = build_temp_graph_for_path(topology, path, current_wls)
                    data = calculate_data_auxiliary_edge(
                        G=temp_G, path=path, laser_detector_position=current_pos_dict, 
                        wavelength_combination=current_wls, wavelength_capacity=current_cap_dict, 
                        traffic=traffic, network_slice=network_slice, remain_num_request=remain_num_request, 
                        link_future_demand=link_future_demand, node_future_demand=node_future_demand, 
                        topology=topology 
                    )
                    del temp_G
                    
                    if data:
                        for edge_data in data:
                            auxiliary_graph.add_edge(src, dst, key=uuid.uuid4().hex, **edge_data)
                        path_found_flag = True 
                    return 

                if idx >= len(candidates): return
                
                # 选当前波长
                target = candidates[idx]
                new_pos = current_pos_dict.copy(); new_pos[target['wl']] = target['pos']
                new_cap = current_cap_dict.copy(); new_cap[target['wl']] = target['cap']
                dfs_find_valid_set(idx + 1, current_theoretical_sum + target['cap'], current_wls + [target['wl']], new_pos, new_cap)
                
                if path_found_flag: return
                
                # 不选当前波长
                remaining_potential = sum(c['cap'] for c in candidates[idx+1:])
                if current_theoretical_sum + remaining_potential >= traffic:
                    dfs_find_valid_set(idx + 1, current_theoretical_sum, current_wls, current_pos_dict, current_cap_dict)

            # 执行探测
            dfs_find_valid_set(0, 0, [], {}, {})
            
            # 注意：如果 DFS 探测失败，说明该物理路径即使存在空闲波长，也无法满足流量需求或干扰约束。
            # 根据“资源不回收”原则，这条路径对于该 traffic 级别已失效，永久剔除。
            if not path_found_flag:
                if path in _PATH_CACHE[cache_key]:
                    _PATH_CACHE[cache_key].remove(path)
            
            # 注意：此处不 break，继续寻找下一条物理路径，以增加辅助图的连通性。
            # 只有当彻底不可行时才剔除。

    del network_slice
    gc.collect()

    return auxiliary_graph





def find_first_valid_physical_path(topology, physical_topology, src, dst, traffic, wavelength_list, served_request):
    """
    基础版热力图探测函数：寻找第一条满足当前网络物理约束的可行路径。
    """
    try:
        path_generator = nx.shortest_simple_paths(physical_topology, src, dst, weight='distance')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    max_attempts = 5 # 限制尝试次数以保证热力图计算速度
    attempts = 0
    
    for path in path_generator:
        attempts += 1
        if attempts > max_attempts:
            break
            
        candidates = []
        for wavelength in wavelength_list:
            # 1. 检查容量
            path_min_cap = float('inf')
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_min_cap = 0
                edges = topology.get_edge_data(u, v)
                for edge_key, edge_attrs in edges.items():
                    if edge_attrs.get('wavelength') == wavelength:
                        edge_min_cap = max(edge_min_cap, edge_attrs['free_capacity'])
                path_min_cap = min(path_min_cap, edge_min_cap)
            
            if path_min_cap < 1e-6:
                continue

            # 2. 检查硬件位置
            temp_slice = nx.Graph()
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                temp_slice.add_edge(u, v)
                temp_slice.nodes[u].update(topology.nodes[u])
                temp_slice.nodes[v].update(topology.nodes[v])
            
            positions = find_laser_detector_position(temp_slice, path, wavelength)
            if len(positions) >= 1:
                candidates.append({'wl': wavelength, 'cap': path_min_cap})
        
        if not candidates:
            continue
            
        if sum(c['cap'] for c in candidates) < traffic:
            continue
            
        return path
        
    return None

# ==========================================
# 流量生成函数 (保持不变)
# ==========================================

def generate_traffic(mid, topology):
    node_list = list(topology.nodes)
    # random.shuffle(node_list)
    # random.shuffle(node_list)
    traffic = {}

    # 获取每个节点的度数
    degrees = dict(topology.degree())

    # 生成所有无向的节点对
    pairs = []
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            node_a = node_list[i]
            node_b = node_list[j]
            # 确定流量方向
            if degrees[node_b] > degrees[node_a]:
                sender, receiver = node_a, node_b
            else:
                sender, receiver = node_b, node_a
            pairs.append((sender, receiver))

    # 计算每对之间的最短路径长度和物理距离
    shortest_paths = {}
    physical_distances = {}
    for sender, receiver in pairs:
        # 计算跳数
        try:
            path_length = nx.shortest_path_length(topology, sender, receiver)
        except nx.NetworkXNoPath:
            path_length = float('inf')
        shortest_paths[(sender, receiver)] = path_length

        # 计算物理距离（使用'distance'属性）
        try:
            distance = nx.shortest_path_length(topology, sender, receiver, weight='distance')
        except nx.NetworkXNoPath:
            distance = float('inf')
        physical_distances[(sender, receiver)] = distance

    # 对有向对进行排序，新增物理距离作为排序条件
    pairs.sort(key=lambda x: (-degrees[x[1]],  shortest_paths[x], -degrees[x[0]], physical_distances[x]))

    # 生成流量值列表，按比例生成，不打乱顺序
    num_pairs = len(pairs)
    num_20_percent = int(0.2 * num_pairs)
    num_60_percent = int(0.6 * num_pairs)
    num_20_percent_2 = num_pairs - num_20_percent - num_60_percent

    traffic_values = ([mid - 1000] * num_20_percent +
                      [mid] * num_60_percent +
                      [mid + 1000] * num_20_percent_2)
    random.shuffle(traffic_values)

    # 按排序后的请求顺序分配流量值
    for idx, (sender, receiver) in enumerate(pairs):
        traffic_value = traffic_values[idx]
        traffic[(sender, receiver)] = traffic_value

    return traffic


def generate_and_sort_requests(map_name):
    network = Network(map_name=map_name, wavelength_list=[1], protocol='BB84', receiver='APD')
    topology = network.physical_topology
    node_list = list(topology.nodes)
    random.shuffle(node_list)
    pairs = []
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            pairs.append((node_list[i], node_list[j]))
    random.shuffle(pairs)

    return pairs


def assign_traffic_values(pairs, mid):
    """根据排序后的请求顺序和 mid 值分配流量"""
    traffic = []
    num_pairs = len(pairs)

    # 生成流量值列表（按比例生成，不打乱顺序）
    num_20_percent = int(0.2 * num_pairs)
    num_60_percent = int(0.6 * num_pairs)
    num_20_percent_2 = num_pairs - num_20_percent - num_60_percent
    # if mid % 10000 == 5000:
    #     step = 5000
    # else:
    #     step = 10000

    traffic_values = (
            [5000 - 0] * num_20_percent +
            [5000] * num_60_percent +
            [5000 + 0] * num_20_percent_2
    )
    random.shuffle(traffic_values)

    # 按排序后的顺序分配流量值
    current_mid = 0
    id = 0
    # while current_mid < mid:
    #     for i in range(len(pairs)):
    #         traffic.append((id, pairs[i][0], pairs[i][1],traffic_values[i]))
    #         id = id + 1
    #     current_mid = current_mid + 5000
    for i in range(len(pairs)):
        traffic.append((id, pairs[i][0], pairs[i][1], traffic_values[i]))
        id = id + 1

    return traffic