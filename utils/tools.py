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

def clear_path_cache():
    global _PATH_CACHE
    _PATH_CACHE = {}

def get_cached_paths(G, src, dst, is_bypass, traffic):
    """
    路径缓存引擎：
    1. 第一次访问时使用 nx.shortest_simple_paths 生成路径。
    2. 后续访问直接从 _PATH_CACHE 读取副本。
    """
    protocol = config.protocol
    detector = config.detector
    
    # 缓存键：源、宿、是否旁路、协议、检测器
    # 注意：我们不把 traffic 放在 key 里，因为物理路径是通用的。
    # 我们在 build_auxiliary_graph 中根据流量动态剔除。
    key = (src, dst, is_bypass, protocol, detector)
    
    if key in _PATH_CACHE:
        # 返回副本，防止外部修改影响缓存
        return list(_PATH_CACHE[key])
    
    paths = []
    if is_bypass:
        try:
            # 获取物理拓扑上的前 K 条路径
            gen = nx.shortest_simple_paths(G, src, dst, weight='distance')
            for path in itertools.islice(gen, 50): # 搜索空间限制为前 50 条
                d = calculate_distance(G, src, dst, path)
                # 预热物理过滤：只要物理上能通 (SKR > 0) 就加入缓存
                if compute_key_rate(d, protocol, detector) > 0:
                    paths.append(path)
                if len(paths) >= 15: # 最终保留 15 条
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
    else:
        # 非 bypass 模式：仅尝试 1-hop 物理边
        if G.has_edge(src, dst):
            paths = [[src, dst]]
            
    _PATH_CACHE[key] = paths
    return list(paths)

def build_network_slice(wavelength_list, topology, traffic):
    """
    为每个波长构建一个“切片”图。
    工程优化：
    1. 确保所有节点都在切片中，即使没有边。
    2. 避免多次 deepcopy。
    """
    network_slice = {}
    for wavelength in wavelength_list:
        # 创建一个包含所有节点的新图，但不包含边
        wavelength_slice = nx.DiGraph()
        wavelength_slice.add_nodes_from(topology.nodes(data=True))
        
        # 仅添加满足该波长且有剩余容量的边
        for u, v, key, data in topology.edges(keys=True, data=True):
            if data.get('wavelength') == wavelength and data.get('free_capacity', 0) > 0:
                wavelength_slice.add_edge(u, v, key=key, **data)
        
        network_slice[wavelength] = wavelength_slice
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
            for links in node_data['laser'][wavelength]:
                add_links(links)
        
        if 'detector' in node_data and wavelength in node_data['detector']:
            for links in node_data['detector'][wavelength]:
                add_links(links)

    # --- 4. 预计算 Path 上的边是否被覆盖 ---
    # path_edges_covered[k] = True 表示 path[k]->path[k+1] 这段路在 bypass 中
    path_edges_covered = [False] * (path_len - 1)
    
    for k in range(path_len - 1):
        u = path[k]
        v = path[k+1]
        u_t = to_tuple(u)
        v_t = to_tuple(v)
        
        # 严格检查正向是否在集合中（有向光纤）
        if (u_t, v_t) in bypass_link_set:
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
        # 安全检查：确保节点存在且具有 laser 属性
        if node not in wavelength_slice.nodes: continue
        node_data = wavelength_slice.nodes[node]
        if 'laser' not in node_data or wavelength not in node_data['laser']: continue
        
        for detector_list in node_data['laser'][wavelength]:
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
                                  traffic, network_slice, remain_num_request, link_future_demand=None, topology=None, node_future_demand=None, benchmark_dist=80.0):
    """
    V12 核心重构：自适应单向代价场。
    """
    data = []
    keys = laser_detector_position.keys()
    values = laser_detector_position.values()
    wavelength_laser_detector_list = [dict(zip(keys, combo)) for combo in product(*values)]
    
    detector_type = getattr(config, 'detector', 'SNSPD')
    ice_box_capacity = getattr(config, 'ice_box_capacity', 8)
    unit_cooling_power = getattr(config, 'unit_cooling_power', 3000)
    
    # 获取当前协议在“标杆距离”下的原生 SKR
    benchmark_skr = compute_key_rate(benchmark_dist, config.protocol, config.detector)

    for wavelength_laser_detector in wavelength_laser_detector_list:
        wavelength_traffic_limitation = {}
        max_traffic = 0
        wavelength_used_laser_detector = {}
        
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

        if max_traffic >= traffic:
            source_power = 0
            detector_power = 0 
            other_power = 0
            real_ice_box_power = 0
            
            marginal_weight = 0.0
            
            # 记录本逻辑边涉及的所有物理链路，用于功耗去重
            unique_links = set()
            
            for wavelength, laser_detector in wavelength_laser_detector.items():
                laser_node, det_node = laser_detector
                component_power = calculate_power(
                    laser_detector_position={'laser': laser_node, 'detector': det_node}, 
                    path=path,
                    G=network_slice[wavelength]
                )
                source_power += component_power['source']
                detector_power += component_power['detector'] 
                # other_power 和 ice_box_power 逻辑需要更精细的去重
                # 这里暂时累加，但在 marginal_weight 中体现共享
                real_ice_box_power += component_power['ice_box']
                
                is_already_active = False
                if laser_node is None:
                    is_already_active = True
                elif laser_node in G.nodes and wavelength in G.nodes[laser_node].get('laser', {}):
                    l_idx, d_idx = path.index(laser_node), path.index(det_node)
                    cover_links = path[l_idx:d_idx+1]
                    if cover_links in G.nodes[laser_node]['laser'][wavelength]:
                        is_already_active = True
                
                if is_already_active:
                    # 如果已经激活，仅增加 10% 的激光器维持功耗
                    marginal_weight += (component_power['source'] * 0.1)
                else:
                    # 如果是新激活，增加完整的硬件功耗 (170W base + source + detector)
                    marginal_weight += (component_power['source'] + component_power['detector'] + component_power['other'])
                    if detector_type == 'SNSPD':
                        marginal_weight += (unit_cooling_power / ice_box_capacity)

            # 修正统计信息中的 other_power：对于同一条逻辑路径，base power 只计一次
            # 假设 calculate_power 返回的 other 是 170W
            first_wl = next(iter(wavelength_laser_detector))
            first_laser, first_det = wavelength_laser_detector[first_wl]
            base_comp_power = calculate_power({'laser': first_laser, 'detector': first_det}, path, G)
            other_power = base_comp_power['other']

            # C. 频谱占用成本 (Spectrum Occupation Cost)
            num_links = len(path) - 1
            num_wls = len(wavelength_combination)
            spectrum_occupied = num_wls * num_links
            
            # 这里的代价要低，因为用户希望 Bypass 的频谱占用也低
            # 如果代价太高，Dijkstra 可能会为了省波长而放弃 Bypass
            spectrum_cost = spectrum_occupied * 1.0 

            # D. 节点固定成本 (Node Management Cost)
            # 核心：这是区分 Bypass 和 No-Bypass 的关键
            # 每多经过一个中继节点（即多一条逻辑边），就增加巨额成本
            node_management_cost = 500000.0 

            # E. 拥塞保护 (Congestion Guard)
            congestion_multiplier = 0.0
            
            # F. 最终权重计算
            # 权重必须直接反映我们的目标：最小化逻辑跳数 (Power) 和 频谱 (Spectrum)
            is_bypass_edge = len(path) > 2
            
            # 如果是 Bypass 边，我们给予一定的权重优惠，反映其透明转发的优势
            if is_bypass_edge:
                marginal_weight *= 0.5
                spectrum_cost *= 0.5
                node_management_cost = 1000.0 # Bypass 逻辑边本身的开销极低
            else:
                # OEO 边（1-hop）保留高昂的固定开销
                node_management_cost = 500000.0 
            
            weight = marginal_weight + spectrum_cost + node_management_cost
            weight = max(1.0, weight)

            data.append({
                'power': source_power + detector_power + other_power + real_ice_box_power,
                'source_power': source_power,
                'detector_power': detector_power,
                'other_power': other_power,
                'ice_box_power': real_ice_box_power, 
                'path': path, 
                'laser_detector_position': wavelength_laser_detector,
                'wavelength_traffic': wavelength_traffic_limitation,
                'weight': weight,
                'wavelength_list': wavelength_combination,
                'transverse_laser_detector': wavelength_used_laser_detector,
                'marginal_weight': marginal_weight,
                'spectrum_cost': spectrum_cost,
                'node_management_cost': node_management_cost,
                'efficiency_penalty': 1.0,
                'congestion_multiplier': 0.0,
                'is_bypass_edge': is_bypass_edge
            })

    return data


def get_k_shortest_paths(graph, src, dst, k=5, weight="distance"):
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        # 如果是多重图，转换为简单图以调用 shortest_simple_paths
        # 根据输入图的有向性选择 Graph 或 DiGraph
        G = nx.DiGraph() if graph.is_directed() else nx.Graph()
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
    # 使用 tuple 保持有向性，解决双向链路误判问题
    norm = lambda e: tuple(e)
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
    
    # 动态获取当前地图的物理链路平均长度，作为能效标杆的基准距离
    total_dist = 0
    num_edges = 0
    for u, v, d in physical_topology.edges(data='distance'):
        total_dist += d
        num_edges += 1
    avg_physical_dist = total_dist / num_edges if num_edges > 0 else 80.0

    # 3. 遍历处理
    for src, dst in pairs:
        # --- 3.1 从路径缓存中读取候选路径 ---
        path_list = get_cached_paths(physical_topology, src, dst, config.bypass, traffic)
        if not path_list: continue
        
        # 获取用于永久剔除的缓存键
        cache_key = (src, dst, config.bypass, config.protocol, config.detector)
        
        # --- 3.2 遍历探测路径 ---
        # 使用副本 list(path_list) 进行遍历，以便在循环中修改原始 path_list
        for path in list(path_list):
            path_found_flag = False  # 标志：该路径是否最终成功建立了辅助边

            # === 步骤 A: 收集并筛选候选波长 ===
            candidates = []
            for wavelength in wavelength_list:
                wl_slice = network_slice[wavelength]
                min_cap = find_min_free_capacity(wavelength_slice=wl_slice, path=path)
                if min_cap <= 0: continue 

                positions = find_laser_detector_position(
                    wavelength_slice=wl_slice, path=path, wavelength=wavelength
                )
                if len(positions) < 1: continue 
                
                candidates.append({'wl': wavelength, 'cap': min_cap, 'pos': positions})
            
            # === 步骤 B: 综合判定与 DFS 探测 ===
            if candidates:
                candidates.sort(key=lambda x: x['cap'], reverse=False)
                n_candidates = len(candidates)
                suffix_max_cap = [0.0] * n_candidates
                current_sum = 0.0
                for i in range(n_candidates - 1, -1, -1):
                    current_sum += candidates[i]['cap']
                    suffix_max_cap[i] = current_sum
                
                # 只有总容量可能达到 traffic 时才探测
                if suffix_max_cap[0] >= traffic:
                    def dfs_find_valid_set(idx, current_wls, current_pos_dict, current_cap_dict, current_theoretical_sum, remain_num_request, avg_dist):
                        nonlocal path_found_flag
                        if path_found_flag: return 
                        if current_theoretical_sum >= traffic:
                            if not check_path_validity_for_request(path, current_wls, served_request): return 
                            temp_G = build_temp_graph_for_path(topology, path, current_wls)
                            data = calculate_data_auxiliary_edge(
                                G=temp_G, path=path, laser_detector_position=current_pos_dict, 
                                wavelength_combination=current_wls, wavelength_capacity=current_cap_dict, 
                                traffic=traffic, network_slice=network_slice, remain_num_request=remain_num_request, 
                                link_future_demand=link_future_demand, node_future_demand=node_future_demand, 
                                topology=topology, benchmark_dist=avg_dist
                            )
                            if data:
                                for edge_data in data:
                                    auxiliary_graph.add_edge(path[0], path[-1], key=uuid.uuid4().hex, **edge_data)
                                path_found_flag = True 
                            return 

                        if idx >= n_candidates or current_theoretical_sum + suffix_max_cap[idx] < traffic: return

                        # 选
                        current_wls.append(candidates[idx]['wl'])
                        current_pos_dict[candidates[idx]['wl']] = candidates[idx]['pos']
                        current_cap_dict[candidates[idx]['wl']] = candidates[idx]['cap']
                        dfs_find_valid_set(idx + 1, current_wls, current_pos_dict, current_cap_dict, 
                                           current_theoretical_sum + candidates[idx]['cap'], remain_num_request, avg_dist)
                        if path_found_flag: return
                        # 不选
                        current_wls.pop()
                        del current_pos_dict[candidates[idx]['wl']]
                        del current_cap_dict[candidates[idx]['wl']]
                        dfs_find_valid_set(idx + 1, current_wls, current_pos_dict, current_cap_dict, 
                                           current_theoretical_sum, remain_num_request, avg_dist)

                    dfs_find_valid_set(0, [], {}, {}, 0, remain_num_request, avg_physical_dist)
            
            # === 步骤 C: 结果判定 ===
            if path_found_flag:
                break # 找到一条可行路径就收工

    del network_slice
    gc.collect()

    return auxiliary_graph





def find_first_valid_physical_path(topology, physical_topology, src, dst, traffic, wavelength_list, served_request):
    """
    专门为动态热力图设计的探测函数：寻找第一条满足当前网络物理约束的可行路径。
    采用与 build_auxiliary_graph 一致的缓存与流式探测逻辑。
    """
    # [工程优化]：使用路径缓存，避免在热力图计算中重复执行图搜索
    path_list = get_cached_paths(physical_topology, src, dst, config.bypass, traffic)
    if not path_list:
        return None

    # 获取用于永久剔除的缓存键
    cache_key = (src, dst, config.bypass, config.protocol, config.detector)
    
    for path in list(path_list):
        path_is_valid = False
        candidates = []
        for wavelength in wavelength_list:
            # 1. 检查容量 (直接从原拓扑检查)
            path_min_cap = float('inf')
            is_wavelength_available = True
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_found = False
                if topology.has_edge(u, v):
                    for key, data in topology[u][v].items():
                        if data.get('wavelength') == wavelength and data.get('free_capacity', 0) > 0:
                            path_min_cap = min(path_min_cap, data['free_capacity'])
                            edge_found = True
                            break
                if not edge_found:
                    is_wavelength_available = False
                    break
            
            if not is_wavelength_available:
                continue
                
            # 2. 检查硬件 (简化版：临时构建 slice 检查)
            temp_slice = nx.DiGraph()
            for k in range(len(path) - 1):
                u_n, v_n = path[k], path[k+1]
                temp_slice.add_edge(u_n, v_n)
                temp_slice.nodes[u_n].update(topology.nodes[u_n])
                temp_slice.nodes[v_n].update(topology.nodes[v_n])
            
            positions = find_laser_detector_position(temp_slice, path, wavelength)
            if len(positions) > 0:
                candidates.append(path_min_cap)
        
        # 3. 汇总检查是否能满足流量
        if sum(candidates) >= traffic:
            path_is_valid = True
            
        if path_is_valid:
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
