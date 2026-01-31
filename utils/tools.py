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

# 全局缓存，减少重复计算
_PATH_CACHE = {}
_LD_POS_CACHE = {}

def clear_path_cache():
    _PATH_CACHE.clear()
    _LD_POS_CACHE.clear()

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
    1. 增加缓存机制
    2. 解决 unhashable type: 'list' 报错
    """
    # 建立缓存键
    # 注意：wavelength_slice 的状态可能会变（虽然在 build_auxiliary_graph 内部是静态的）
    # 但为了保险，我们只在 build_auxiliary_graph 的单次运行中信任它
    # 或者包含 wavelength_slice 的 id 或其他标识
    cache_key = (id(wavelength_slice), tuple(path), wavelength)
    if cache_key in _LD_POS_CACHE:
        return _LD_POS_CACHE[cache_key]

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
            
    _LD_POS_CACHE[cache_key] = laser_detector_position
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
                                  traffic, network_slice, remain_num_request, topology=None, benchmark_dist=80.0):
    """
    V13：基于 Grooming 能力摊薄的精密权重模型。
    """
    data = []
    keys = laser_detector_position.keys()
    values = laser_detector_position.values()
    wavelength_laser_detector_list = [dict(zip(keys, combo)) for combo in product(*values)]
    
    # === 获取全局配置 ===
    detector_type = getattr(config, 'detector', 'SNSPD')
    ice_box_capacity = getattr(config, 'ice_box_capacity', 8)
    unit_cooling_power = getattr(config, 'unit_cooling_power', 3000)

    for wavelength_laser_detector in wavelength_laser_detector_list:
        # --- [方案 A]：精确化预估，只计算满足流量所需的最小波长子集 ---
        actual_needed_wls = []
        actual_max_traffic = 0
        actual_wavelength_traffic_limitation = {}
        actual_wavelength_used_laser_detector = {}
        
        # 1. 确定实际需要的波长子集及其容量
        # 按照波长列表顺序（通常 DFS 已经排好序或有其逻辑）进行累加
        for wavelength, laser_detector in wavelength_laser_detector.items():
            key_rate, used_laser_detector = Max_capacity(
                laser_detector=laser_detector,
                path=path,
                G=G, 
                wavelength=wavelength, 
                network_slice=network_slice[wavelength]
            )
            
            cap_contribution = min(wavelength_capacity[wavelength], key_rate)
            if cap_contribution <= 0: continue
            
            actual_needed_wls.append(wavelength)
            actual_max_traffic += cap_contribution
            actual_wavelength_traffic_limitation[wavelength] = cap_contribution
            actual_wavelength_used_laser_detector[wavelength] = used_laser_detector
            
            # 一旦凑够流量，就不再计入后续波长
            if actual_max_traffic >= traffic:
                break

        # 2. 只有满足流量才进行后续权重与功耗计算
        if actual_max_traffic >= traffic:
            source_power = 0
            detector_power = 0 
            other_power = 0
            
            spectrum = 0
            used_LD = 0
            
            node_new_detectors_count = {}

            # 只计算实际需要的波长对应的硬件开销
            for wavelength in actual_needed_wls:
                laser_detector = wavelength_laser_detector[wavelength]
                component_power = calculate_power(
                    laser_detector_position={'laser': laser_detector[0], 'detector': laser_detector[1]}, 
                    path=path,
                    G=network_slice[wavelength]
                )
                
                source_power += component_power['source']
                detector_power += component_power['detector'] 
                other_power += component_power['other']
                
                used_LD += len(actual_wavelength_used_laser_detector[wavelength])

                det_node = laser_detector[1]
                if det_node is not None:
                    node_new_detectors_count[det_node] = node_new_detectors_count.get(det_node, 0) + 1

                for i in range(len(path) - 1):
                    u_edge, v_edge = path[i], path[i+1]
                    if topology.has_edge(u_edge, v_edge):
                        for _, edge_data in topology[u_edge][v_edge].items():
                            if edge_data.get('wavelength') == wavelength and not edge_data.get('occupied', False):
                                spectrum += 1
                                break
            
            # === 3. 共享冰箱功耗 (只针对实际需要的探测器) ===
            ice_box_power = 0
            
            if detector_type == 'SNSPD':
                for node, new_count in node_new_detectors_count.items():
                    # 获取该节点当前已有的探测器数量
                    current_num = topology.nodes[node].get('num_detector', 0) if node in topology.nodes else 0
                    
                    # 之前的冰箱数 = ceil(当前数量 / 容量)
                    fridges_before = math.ceil(current_num / ice_box_capacity)
                    # 加上新增探测器后的总量
                    total_num_after = current_num + new_count
                    # 之后的冰箱数
                    fridges_after = math.ceil(total_num_after / ice_box_capacity)
                    
                    # 只有当冰箱数量确实增加时，才计入 3000W
                    marginal_fridges = fridges_after - fridges_before
                    if marginal_fridges > 0:
                        ice_box_power += marginal_fridges * unit_cooling_power
            
            # === 4. 汇总与权重计算 ===
            total_power_base = source_power + detector_power + other_power
            eps = 1e-12

            # 1) 主项：单位能力功耗（grooming 的“摊薄成本”核心）
            # 这里将冰箱功耗乘以 0.125 体现共享优势
            unit_power = ((total_power_base + ice_box_power) / max(actual_max_traffic, eps)) * 1e9

            # 2) 新增段数惩罚：鼓励复用已有的逻辑链路
            new_segments = sum(1 for w in actual_needed_wls if wavelength_laser_detector[w][0] is not None)
            seg_penalty = (new_segments / max(actual_max_traffic, eps)) * 1e9

            # 3) 波长数惩罚：鼓励波长聚合，减少碎片
            wl_penalty = (len(actual_needed_wls) / max(actual_max_traffic, eps))
            wl_penalty = wl_penalty * wl_penalty * 1e9

            # 4) 紧张度惩罚：避免“刚好够”，给未来留余量
            tightness = traffic / max(actual_max_traffic, eps)
            tight_penalty = tightness * tightness

            # 5) 弱频谱惩罚
            s_penalty = ((spectrum) / max(len(actual_needed_wls), 1))
            s_penalty = s_penalty * s_penalty * 1e-4

            weight = (
                unit_power 
                + 0.05 * seg_penalty 
                + 0.02 * wl_penalty 
                + 1e3 * tight_penalty 
                + s_penalty
            )

            data.append({
                'power': total_power_base + ice_box_power,
                'source_power': source_power,
                'detector_power': detector_power,
                'other_power': other_power,
                'ice_box_power': ice_box_power, 
                'path': path, 
                'laser_detector_position': {wl: wavelength_laser_detector[wl] for wl in actual_needed_wls},
                'wavelength_traffic': actual_wavelength_traffic_limitation,
                'weight': weight,
                'wavelength_list': actual_needed_wls,
                'transverse_laser_detector': actual_wavelength_used_laser_detector,
                'is_bypass_edge': len(path) > 2
            })

    return data

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

def build_auxiliary_graph(topology, wavelength_list, traffic, physical_topology, shared_key_rate_list, served_request, remain_num_request):
    """
    终极修正版 V8：DFS 回溯 + 智能剪枝 (Backtracking with Pruning)
    """
    # [关键修正]：每次构建辅助图前，必须清空硬件位置缓存
    # 因为随着业务上线，节点上的硬件状态（Laser/Detector）是动态变化的
    _LD_POS_CACHE.clear()
    
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

                # [核心优化]：提前校验单波长合法性
                if not check_path_validity_for_request(path, [wavelength], served_request):
                    continue
                
                candidates.append({'wl': wavelength, 'cap': min_cap})
            
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
                    def dfs_find_valid_set(idx, current_wls, current_cap_dict, current_theoretical_sum):
                        # 使用生成器模式：产出所有满足带宽需求的潜在波长组合
                        if current_theoretical_sum >= traffic:
                            yield current_wls[:], current_cap_dict.copy()
                            return

                        if idx >= n_candidates or current_theoretical_sum + suffix_max_cap[idx] < traffic:
                            return

                        # 选该波长
                        current_wls.append(candidates[idx]['wl'])
                        current_cap_dict[candidates[idx]['wl']] = candidates[idx]['cap']
                        yield from dfs_find_valid_set(idx + 1, current_wls, current_cap_dict, 
                                                     current_theoretical_sum + candidates[idx]['cap'])

                        # 不选该波长 (回溯)
                        current_wls.pop()
                        del current_cap_dict[candidates[idx]['wl']]
                        yield from dfs_find_valid_set(idx + 1, current_wls, current_cap_dict, current_theoretical_sum)

                    # --- 迭代尝试 DFS 产出的每一个波长组合 ---
                    for found_wls, found_caps in dfs_find_valid_set(0, [], {}, 0):
                        # 1. 决定位置 (find_laser_detector_position)
                        current_pos_dict = {}
                        pos_valid = True
                        for wl in found_wls:
                            pos = find_laser_detector_position(network_slice[wl], path, wl)
                            if not pos:
                                pos_valid = False
                                break
                            current_pos_dict[wl] = pos
                        
                        # 2. 如果位置有效，尝试建立辅助边
                        if pos_valid:
                            temp_G = build_temp_graph_for_path(topology, path, found_wls)
                            data = calculate_data_auxiliary_edge(
                                G=temp_G, path=path, laser_detector_position=current_pos_dict,
                                wavelength_combination=found_wls, wavelength_capacity=found_caps,
                                traffic=traffic, network_slice=network_slice, remain_num_request=remain_num_request,
                                topology=topology, benchmark_dist=avg_physical_dist
                            )
                            if data:
                                for edge_data in data:
                                    auxiliary_graph.add_edge(path[0], path[-1], key=uuid.uuid4().hex, **edge_data)
                                path_found_flag = True
                                break # 找到一个通的组合，该路径处理成功，退出组合迭代
                
            # === 步骤 C: 结果判定 ===
            if path_found_flag:
                break # 找到一条可行路径就收工
            else:
                # [核心判定]：如果该路径尝试了“第一组最佳波长组合”仍无法建立边，且满足资源不释放/Traffic恒定，则永久剔除
                if path in path_list:
                    path_list.remove(path)
                cache_key_full = (src, dst, config.bypass, config.protocol, config.detector)
                if path in _PATH_CACHE.get(cache_key_full, []):
                    _PATH_CACHE[cache_key_full].remove(path)

    del network_slice
    gc.collect()

    return auxiliary_graph

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
