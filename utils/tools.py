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

# 全局缓存已移除，改为参数传递
# _PATH_CACHE = {}
# _LD_POS_CACHE = {}

def clear_path_cache(path_cache=None, ld_pos_cache=None):
    if path_cache is not None:
        path_cache.clear()
    if ld_pos_cache is not None:
        ld_pos_cache.clear()

def get_cached_paths(G, src, dst, is_bypass, traffic, path_cache=None):
    """
    路径缓存引擎：
    1. 第一次访问时使用 nx.shortest_simple_paths 生成路径。
    2. 后续访问直接从 path_cache 读取副本。
    """
    if path_cache is None:
        path_cache = {}

    protocol = config.protocol
    detector = config.detector
    
    # 缓存键：源、宿
    key = (src, dst)
    
    if key in path_cache:
        # 返回副本，防止外部修改影响缓存
        return list(path_cache[key])
    
    paths = []
    if is_bypass:
        # print("Find bypass pair")
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
        # print("Find nobypass pair")
        if G.has_edge(src, dst):
            paths = [[src, dst]]
            
    _PATH_CACHE = {} # Deleted
    
    path_cache[key] = paths
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

def find_laser_detector_position(wavelength_slice, path, wavelength, ld_pos_cache=None):
    """
    优化版：
    1. 增加缓存机制
    2. 解决 unhashable type: 'list' 报错
    """
    if ld_pos_cache is None:
        ld_pos_cache = {}
        
    # 建立缓存键
    # 注意：wavelength_slice 的状态可能会变（虽然在 build_auxiliary_graph 内部是静态的）
    # 但为了保险，我们只在 build_auxiliary_graph 的单次运行中信任它
    # 或者包含 wavelength_slice 的 id 或其他标识
    cache_key = (id(wavelength_slice), tuple(path), wavelength)
    if cache_key in ld_pos_cache:
        return ld_pos_cache[cache_key]

    # --- 辅助函数：将 list/dict 递归转为 tuple，使其可哈希 ---
    def to_tuple(obj):
        if isinstance(obj, list):
            return tuple(to_tuple(x) for x in obj)
        if isinstance(obj, dict):
            # 将字典转为排序后的 items tuple: ((key1, val1), (key2, val2), ...)
            return tuple(sorted((k, to_tuple(v)) for k, v in obj.items()))
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
            
    ld_pos_cache[cache_key] = laser_detector_position
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
import numpy as np
from itertools import product
try:
    import config
except ImportError:
    pass

def calculate_data_auxiliary_edge(G, path, wavelength_combination, wavelength_capacity, laser_detector_position, 
                                   traffic, network_slice, remain_num_request, topology=None, benchmark_dist=80.0): 
    """ 
    V16：回归物理本质的特征提取逻辑。
    计算并返回原始特征（Raw Features），后续由 build_auxiliary_graph 进行动态标定和加权。
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
        actual_needed_wls = [] 
        actual_max_traffic = 0 
        actual_wavelength_traffic_limitation = {} 
        actual_wavelength_used_laser_detector = {} 
        min_free_cap_list = []

        # 1. 确定实际需要的波长子集
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
            min_free_cap_list.append(wavelength_capacity[wavelength])

            if actual_max_traffic >= traffic: 
                break 

        # 2. 只有满足流量才进行特征提取
        if actual_max_traffic >= traffic: 
            source_power = 0 
            detector_power = 0 
            other_power = 0 
            delta_spectrum = 0 
            node_new_detectors_count = {} 
            wavelength_power_info = {}
            wavelength_bypass_info = {} # 新增：记录每个波长的 LD Bypass 数量
            wavelength_dist_info = {}   # 新增：记录每个波长的物理距离

            # 遍历实际需要的波长
            for idx, wavelength in enumerate(actual_needed_wls): 
                laser_detector = wavelength_laser_detector[wavelength] 
                
                # [核心修正]：计算边际功耗。如果该波长下的硬件已存在，则边际功耗为 0。
                laser_pos, det_pos = laser_detector[0], laser_detector[1]
                is_new_hardware = True
                
                # 计算 Bypass 数量
                bypass_cnt = 0
                wl_dist = 0.0 # 新增：波长距离
                
                if laser_pos is not None and det_pos is not None:
                    bypass_cnt = max(0, path.index(det_pos) - path.index(laser_pos) - 1)
                    # 计算该波长覆盖的物理距离
                    wl_dist = calculate_distance(G=topology, path=path, start=laser_pos, end=det_pos)
                
                wavelength_dist_info[wavelength] = wl_dist
                
                if is_new_hardware: 
                    # 物理特征应该反映硬件能力，无论新旧。
                    # 如果复用了旧硬件，它的 Bypass 能力依然存在。
                    wavelength_bypass_info[wavelength] = bypass_cnt
                if laser_pos is not None and det_pos is not None:
                    cover_links_tuple = tuple(path[path.index(laser_pos):path.index(det_pos)+1])
                    if cover_links_tuple in G.nodes[laser_pos]['laser_capacity'][wavelength]:
                        is_new_hardware = False

                if is_new_hardware:
                    component_power = calculate_power( 
                        laser_detector_position={'laser': laser_pos, 'detector': det_pos}, 
                        path=path, 
                        G=network_slice[wavelength] 
                    ) 
                    source_power += component_power['source'] 
                    detector_power += component_power['detector'] 
                    other_power += component_power['other'] 
                    
                    wavelength_power_info[wavelength] = {
                        'source': component_power['source'],
                        'detector': component_power['detector'],
                        'other': component_power['other']
                    }
                    
                    if det_pos is not None: 
                        node_new_detectors_count[det_pos] = node_new_detectors_count.get(det_pos, 0) + 1 
                else:
                    wavelength_power_info[wavelength] = {'source': 0, 'detector': 0, 'other': 0}
                
                # 统计 Δspectrum: 真正会新增占用的物理边 (这点与硬件复用无关)
                for i in range(len(path) - 1): 
                    u_edge, v_edge = path[i], path[i+1] 
                    if topology.has_edge(u_edge, v_edge): 
                        for _, edge_data in topology[u_edge][v_edge].items(): 
                            if edge_data.get('wavelength') == wavelength and not edge_data.get('occupied', False): 
                                delta_spectrum += 1 
                                break 
            
            # === 3. 冰箱阶跃计算 === 
            marginal_fridges = 0 
            if detector_type == 'SNSPD': 
                for node, new_count in node_new_detectors_count.items(): 
                    current_num = topology.nodes[node].get('num_detector', 0) if node in topology.nodes else 0 
                    fridges_before = math.ceil(current_num / ice_box_capacity) 
                    total_num_after = current_num + new_count 
                    fridges_after = math.ceil(total_num_after / ice_box_capacity) 
                    marginal_fridges += max(0, fridges_after - fridges_before)
            
            ice_box_power = marginal_fridges * unit_cooling_power
            total_power_base = source_power + detector_power + other_power 
            eps = 1e-12 

            # === 4. 提取原始特征 (Raw Features) ===
            # [重要修正]：单位功耗应基于本次请求的流量 (traffic) 计算，而非波长总容量
            # 这样 Dijkstra 累加权重时才等价于物理总功耗的累加
            raw_unit = (total_power_base + ice_box_power * 0.125) / max(traffic, eps)
            num_new_LD = sum(1 for w in actual_needed_wls if wavelength_laser_detector[w][0] is not None)
            num_wls = len(actual_needed_wls)
            raw_invcap = traffic / (actual_max_traffic + eps)
            total_dist = calculate_distance(G=topology, path=path, start=path[0], end=path[-1])
            f_dist = total_dist / max(benchmark_dist, eps)
            min_free_cap = min(min_free_cap_list) if min_free_cap_list else 0

            # 5. 新增：资源占用与利用率特征
            # 计算物理路径上的平均波长占用率（衡量拥塞风险）
            total_occ = 0
            for i in range(len(path) - 1):
                u_edge, v_edge = path[i], path[i+1]
                if topology.has_edge(u_edge, v_edge):
                    # 遍历物理边上的所有波长
                    edge_dict = topology[u_edge][v_edge]
                    occ_count = sum(1 for _, d in edge_dict.items() if d.get('occupied'))
                    total_occ += occ_count / max(1, len(edge_dict))
            f_occ = total_occ / (len(path) - 1) if len(path) > 1 else 0
            
            # 计算波长容量利用率（衡量资源浪费，1 - 流量/总容量）
            total_cap = num_wls * actual_max_traffic
            f_waste = max(0.0, 1.0 - traffic / (total_cap + eps))
            
            # [新增]：Bypass 奖励项（按节省的中继节点数计算）
            # len(path) - 2 即为中间被跳过的节点数（潜在节省的硬件套数）
            f_bypass = max(0.0, float(len(path) - 2))

            # 最终特征字典
            raw_features = {
                'raw_unit': raw_unit,
                'num_new_LD': num_new_LD,
                'num_wls': num_wls,
                'delta_spectrum': delta_spectrum,
                'raw_invcap': raw_invcap,
                'f_dist': f_dist,
                'num_new_fridges': marginal_fridges,
                'min_free_cap': min_free_cap,
                'f_occ': f_occ,
                'f_waste': f_waste,
                'f_bypass': f_bypass
            }

            data.append({ 
                'distance': total_dist, # 新增：保存物理距离供特征提取使用
                'power': total_power_base + ice_box_power, 
                'source_power': source_power, 
                'detector_power': detector_power, 
                'other_power': other_power, 
                'ice_box_power': ice_box_power, 
                'path': path, 
                'laser_detector_position': {wl: wavelength_laser_detector[wl] for wl in actual_needed_wls}, 
                'wavelength_traffic': actual_wavelength_traffic_limitation, 
                'raw_features': raw_features,
                'wavelength_list': actual_needed_wls, 
                'wavelength_power_info': wavelength_power_info,
                'wavelength_bypass_info': wavelength_bypass_info,
                'wavelength_dist_info': wavelength_dist_info, # 新增
                'transverse_laser_detector': actual_wavelength_used_laser_detector, 
                'is_bypass_edge': len(path) > 2 
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
    # 使用 tuple 保持有向性
    norm = lambda e: tuple(e)

    path_seq = [norm(edge) for edge in path_links]
    wl_seq   = [norm(edge) for edge in wavelength_covered_links]

    path_set = set(path_seq)
    wl_set   = set(wl_seq)

    # 1) 完全不相交：OK（你说这个没问题）
    if path_set.isdisjoint(wl_set):
        return True

    # 2) 有交集时：要求 wl 覆盖必须贴着 path 的一端（前缀或后缀），且顺序一致
    Lp, Lw = len(path_seq), len(wl_seq)
    if Lw == 0:
        return True
    if Lw > Lp:
        return False

    # 前缀
    if path_seq[:Lw] == wl_seq:
        return True
    # 后缀
    if path_seq[-Lw:] == wl_seq:
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

# ==========================================
# 核心重构：build_auxiliary_graph
# ==========================================

def build_auxiliary_graph(topology, wavelength_list, traffic, physical_topology, shared_key_rate_list, served_request, remain_num_request, path_cache=None, ld_pos_cache=None):
    """
    终极修正版 V8：DFS 回溯 + 智能剪枝 (Backtracking with Pruning)
    """
    if path_cache is None: path_cache = {}
    if ld_pos_cache is None: ld_pos_cache = {}
    
    # [关键修正]：每次构建辅助图前，必须清空硬件位置缓存
    # 因为随着业务上线，节点上的硬件状态（Laser/Detector）是动态变化的
    ld_pos_cache.clear()
    
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

    # 3. 遍历处理并收集原始特征
    all_raw_entries = []
    for src, dst in pairs:
        # --- 3.1 从路径缓存中读取候选路径 ---
        path_list = get_cached_paths(physical_topology, src, dst, config.bypass, traffic, path_cache=path_cache)
        if not path_list: continue
        
        # 获取用于永久剔除的缓存键
        cache_key_full = (src, dst)
        
        path_found_count = 0
        # --- 3.2 遍历探测路径 ---
        for path in list(path_list):
            path_found_flag = False  # 标志：该路径是否最终成功建立了辅助边
            
            # --- 步骤 A: 波长组合搜索 (DFS) ---
            # ... (DFS 逻辑保持不变)
            
            # --- 迭代尝试 DFS 产出的每一个波长组合 ---
            # ... (组合尝试逻辑保持不变)
            # 在这里统计成功的路径数
            if path_found_flag:
                path_found_count += 1
        
        # print(f"DEBUG: Pair ({src}, {dst}) found {path_found_count} viable paths")

            # === 步骤 A: 收集并筛选候选波长 ===
            candidates = []
            for wavelength in wavelength_list:
                wl_slice = network_slice[wavelength]
                min_cap = find_min_free_capacity(wavelength_slice=wl_slice, path=path)
                if min_cap <= 0: continue 

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
                
                if suffix_max_cap[0] >= traffic:
                    def dfs_find_valid_set(idx, current_wls, current_cap_dict, current_theoretical_sum):
                        if current_theoretical_sum >= traffic:
                            yield current_wls[:], current_cap_dict.copy()
                            return

                        if idx >= n_candidates or current_theoretical_sum + suffix_max_cap[idx] < traffic:
                            return

                        current_wls.append(candidates[idx]['wl'])
                        current_cap_dict[candidates[idx]['wl']] = candidates[idx]['cap']
                        yield from dfs_find_valid_set(idx + 1, current_wls, current_cap_dict, 
                                                     current_theoretical_sum + candidates[idx]['cap'])

                        current_wls.pop()
                        del current_cap_dict[candidates[idx]['wl']]
                        yield from dfs_find_valid_set(idx + 1, current_wls, current_cap_dict, current_theoretical_sum)

                    for found_wls, found_caps in dfs_find_valid_set(0, [], {}, 0):
                        current_pos_dict = {}
                        pos_valid = True
                        for wl in found_wls:
                            pos = find_laser_detector_position(network_slice[wl], path, wl, ld_pos_cache=ld_pos_cache)
                            if not pos:
                                pos_valid = False
                                break
                            current_pos_dict[wl] = pos
                        
                        if pos_valid:
                            temp_G = build_temp_graph_for_path(topology, path, found_wls)
                            raw_data = calculate_data_auxiliary_edge(
                                G=temp_G, path=path, laser_detector_position=current_pos_dict,
                                wavelength_combination=found_wls, wavelength_capacity=found_caps,
                                traffic=traffic, network_slice=network_slice, remain_num_request=remain_num_request,
                                topology=topology, benchmark_dist=avg_physical_dist
                            )
                            if raw_data:
                                for entry in raw_data:
                                    entry['src'] = src
                                    entry['dst'] = dst
                                    all_raw_entries.append(entry)
                                path_found_flag = True
                                break # 找到一个通的组合，该路径处理成功
                
            # === 步骤 C: 结果判定 ===
            if not path_found_flag:
                # [核心判定]：如果该路径尝试了仍无法建立边，永久剔除
                if path in path_list:
                    path_list.remove(path)
                if path in path_cache.get(cache_key_full, []):
                    path_cache[cache_key_full].remove(path)

    # 4. 动态标定与权重应用 (回归物理本质)
    if all_raw_entries:
        # 计算 K_unit 标尺（基于非零项，避免全零干扰）
        nonzero_units = [e['raw_features']['raw_unit'] for e in all_raw_entries if e['raw_features']['raw_unit'] > 0]
        if nonzero_units:
            K_unit = np.percentile(nonzero_units, 60)
        else:
            K_unit = 1.0 # 默认值
        K_unit = max(K_unit, 1e-6)

        # 获取 BO 优化的系数
        w = getattr(config, 'weights', {})
        a0 = w.get('a0', 1.0)
        a1 = w.get('a1', 1.0) # unit
        a2 = w.get('a2', 1.0) # newLD
        a3 = w.get('a3', 1.0) # num_wls
        a4 = w.get('a4', 1.0) # Δspectrum
        a5 = w.get('a5', 1.0) # invcap
        a6 = w.get('a6', 1.0) # dist
        a7 = w.get('a7', 1.0) # fridge
        a8 = w.get('a8', 0.1) # f_occ
        a9 = w.get('a9', 0.1) # f_waste
        a10 = w.get('a10', 0.1) # f_bypass

        # 非线性指数参数 (默认为 1.0，即线性)
        # 增加 eps 防止负指数导致的除零错误
        eps = 1e-6
        p1 = w.get('p1', 1.0)  # unit power exponent
        p2 = w.get('p2', 1.0)  # num_new_LD exponent
        p3 = w.get('p3', 1.0)  # num_wls exponent
        p4 = w.get('p4', 1.0)  # delta_spectrum exponent
        p7 = w.get('p7', 1.0)  # num_new_fridges exponent
        p10 = w.get('p10', 1.0) # f_bypass exponent

        for e in all_raw_entries:
            rf = e['raw_features']
            
            # [物理回归]：引入幂函数，捕捉非线性边际代价
            f_unit = (rf['raw_unit'] / K_unit + eps) ** p1
            f_new_LD = (rf['num_new_LD'] + eps) ** p2
            f_wls = (rf['num_wls'] + eps) ** p3
            f_spectrum = (rf['delta_spectrum'] + eps) ** p4
            f_fridge = (rf['num_new_fridges'] + eps) ** p7
            f_bp_reward = (rf['f_bypass'] + eps) ** p10

            # 拥塞与容量风险 (保持 Hinge)
            f_invcap = max(0.0, rf['raw_invcap'] - 0.8)
            f_bottle = max(0.0, traffic / (rf['min_free_cap'] + 1e-12) - 1.0)
            
            # 最终权重组合
            weight = (
                a0
                + a1 * f_unit
                + a2 * f_new_LD
                + a3 * f_wls
                + a4 * f_spectrum
                + a5 * f_invcap
                + a6 * rf['f_dist']
                + a7 * f_fridge
                + a8 * rf['f_occ']
                + a9 * rf['f_waste']
                - a10 * f_bp_reward
                + 1.0 * f_bottle 
            )
            # 确保权重非负 (Dijkstra 要求)
            e['weight'] = max(1e-6, weight)
            auxiliary_graph.add_edge(e['src'], e['dst'], key=uuid.uuid4().hex, **e)

    del network_slice
    # gc.collect() 
    return auxiliary_graph

# ==========================================
# RL 扩展：特征矩阵提取与外部权重应用
# ==========================================

def _dfs_wavelength_combinations(idx, traffic, candidates, suffix_max_cap, current_wls, current_cap_dict, current_theoretical_sum):
    """内部辅助函数：DFS 寻找满足流量需求的波长组合"""
    if current_theoretical_sum >= traffic:
        yield current_wls[:], current_cap_dict.copy()
        return

    if idx >= len(candidates) or current_theoretical_sum + suffix_max_cap[idx] < traffic:
        return

    # 包含当前波长
    current_wls.append(candidates[idx]['wl'])
    current_cap_dict[candidates[idx]['wl']] = candidates[idx]['cap']
    yield from _dfs_wavelength_combinations(idx + 1, traffic, candidates, suffix_max_cap, current_wls, current_cap_dict, 
                                            current_theoretical_sum + candidates[idx]['cap'])

    # 不包含当前波长
    current_wls.pop()
    del current_cap_dict[candidates[idx]['wl']]
    yield from _dfs_wavelength_combinations(idx + 1, traffic, candidates, suffix_max_cap, current_wls, current_cap_dict, current_theoretical_sum)

def extract_feature_matrices_from_graph(auxiliary_graph, node_to_idx, num_nodes, wavelength_list, remain_request_matrix=None, max_stats=None, topology=None):
    """
    修复版 V5：完全自适应图内归一化 (Instance Normalization)
    不再依赖硬编码的 max_stats。对每个 Feature Channel 进行 Z-Score 归一化。
    [新增] topology: 物理拓扑，用于填入 Channel 5 的物理距离
    """
    num_global_feats = 7 # 移除 Bypass Count (冗余)，保留 Path Hops
    num_wl_feats = 5     # 新增 LD Bypass Count, WL Distance
    num_wavelengths = len(wavelength_list)
    
    # 1. 初始化 Tensor (使用 NaN 或 0，这里用 0 方便稀疏处理，但在归一化时需注意掩码)
    # 我们先填原始值，稍后统一归一化
    # 为了区分“无边”和“值为0”，建议初始化为 NaN，或者使用单独的 Adjacency Matrix
    # 但由于 GNN 通常只处理存在的边，这里我们直接填入值。
    # 这里的 Tensor 是 Dense 的 [C, N, N]。
    
    global_tensor = np.zeros((num_global_feats, num_nodes, num_nodes), dtype=np.float32)
    wl_tensor = np.zeros((num_wavelengths * num_wl_feats, num_nodes, num_nodes), dtype=np.float32)
    
    # 记录哪些位置有边 (Mask)
    edge_mask = np.zeros((num_nodes, num_nodes), dtype=bool)
    
    # [新增] 填入 Channel 4: 物理拓扑距离 (原 Ch 5)
    if topology is not None:
        for u, v, data in topology.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                u_i, v_i = node_to_idx[u], node_to_idx[v]
                dist = data.get('distance', 0)
                global_tensor[4, u_i, v_i] = dist
                global_tensor[4, v_i, u_i] = dist 
    
    # 辅助：去重
    min_power_map = np.full((num_nodes, num_nodes), 1e9, dtype=np.float32)
    wl_to_idx = {wl: i for i, wl in enumerate(wavelength_list)}
    
    # --- 第一步：批量收集特征 (Batch Collection) ---
    # 使用列表收集数据，最后一次性填入 Tensor，比逐个填入 Tensor 快得多
    u_list, v_list = [], []
    
    # Global Features Buffers
    dist_list, occ_list, nwls_list, fridge_list, hops_list = [], [], [], [], []
    
    # Wavelength Features Buffers: [List for ch0, List for ch1, ...]
    # 每个波长通道的数据需要单独维护，或者维护一个大列表 [(wl_idx, feature_val), ...]
    # 这里为了性能，我们维护 dict: {wl_idx: {'p': [], 'c': [], 'w': [], 'b': [], 'd': [], 'u': [], 'v': []}}
    wl_data_map = {} 
    
    for u, v, data in auxiliary_graph.edges(data=True):
        if u not in node_to_idx or v not in node_to_idx: continue
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        
        # [Strict Mode] 严格校验数据完整性
        rf = data.get('raw_features')
        if rf is None:
            raise KeyError(f"Edge {u}->{v} missing 'raw_features'")
            
        current_p = data.get('power', 0)
        
        # 仅当当前边是该节点对间功耗最低的边时，才记录 Global 特征
        # 注意：Global 特征是 per-link 的，而不是 per-edge (multi-graph)
        # 所以这里其实是在做一次 Min-Pooling
        if current_p < min_power_map[u_idx, v_idx]:
            min_power_map[u_idx, v_idx] = current_p
            edge_mask[u_idx, v_idx] = True
            
            u_list.append(u_idx)
            v_list.append(v_idx)
            
            # 收集 Global Features
            dist_list.append(data.get('distance', 0))
            occ_list.append(rf.get('f_occ', 0))
            nwls_list.append(rf.get('num_wls', 0))
            fridge_list.append(rf.get('num_new_fridges', 0) * 3000.0)
            path_hops = len(data.get('path', [])) - 1
            hops_list.append(float(max(0, path_hops)))
            
        # 收集 Wavelength Features (对每一条边都收集，不只是最小功耗边)
        # 实际上 GNN 通常也只关注最优边，或者聚合所有边。
        # 原逻辑是：只要 power < min_power，就填入 global；
        # 但 wl_tensor 是基于 u,v 索引的。
        # 如果 u,v 之间有多条边（多重图），wl_tensor 会被覆盖。
        # 原逻辑中：`wl_tensor[..., u, v] = ...` 也是在 if current_p < min_power 内部吗？
        # 让我们回看原代码... 是的！原代码所有赋值都在 if 块内。
        # 所以 wl features 也只取功耗最低的那条边的。
        
        if current_p <= min_power_map[u_idx, v_idx]: # 注意这里用 <= 或者是上面的 < 已经更新了 min_power
            # 由于上面更新了 min_power，这里需要判断是否就是刚才更新的那条
            # 简单起见，我们把 wl 收集也放在上面的 if 块里
            pass

    # --- 重新组织循环以匹配原逻辑 ---
    # 重新初始化
    min_power_map.fill(1e9)
    edge_mask.fill(False)
    u_list, v_list = [], []
    dist_list, occ_list, nwls_list, fridge_list, hops_list = [], [], [], [], []
    
    # 针对 WL 特征，我们使用 list of lists: feats[wl_idx][channel_idx] -> list of values
    # 并记录对应的 u, v
    wl_feats_u = [[] for _ in range(num_wavelengths)]
    wl_feats_v = [[] for _ in range(num_wavelengths)]
    wl_feats_val = [[[] for _ in range(num_wl_feats)] for _ in range(num_wavelengths)]
    
    for u, v, data in auxiliary_graph.edges(data=True):
        if u not in node_to_idx or v not in node_to_idx: continue
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        
        current_p = data.get('power', 0)
        
        # 仅处理最优边
        if current_p < min_power_map[u_idx, v_idx]:
            min_power_map[u_idx, v_idx] = current_p
            edge_mask[u_idx, v_idx] = True
            
            # 1. Global Features Collection
            u_list.append(u_idx)
            v_list.append(v_idx)
            rf = data['raw_features']
            
            dist_list.append(data.get('distance', 0))
            occ_list.append(rf.get('f_occ', 0))
            nwls_list.append(rf.get('num_wls', 0))
            fridge_list.append(rf.get('num_new_fridges', 0) * 3000.0)
            hops_list.append(float(max(0, len(data.get('path', [])) - 1)))
            
            # 2. Wavelength Features Collection
            wls = data.get('wavelength_list', [])
            wl_power_info = data.get('wavelength_power_info', {})
            wl_bypass_info = data.get('wavelength_bypass_info', {})
            wl_dist_info = data.get('wavelength_dist_info', {})
            wl_caps = data.get('wavelength_traffic', {})
            
            for wl in wls:
                if wl not in wl_to_idx: continue
                w_idx = wl_to_idx[wl]
                
                wl_feats_u[w_idx].append(u_idx)
                wl_feats_v[w_idx].append(v_idx)
                
                p_info = wl_power_info.get(wl, {'source':0, 'detector':0, 'other':0})
                ld_power = p_info['source'] + p_info['detector'] + p_info['other']
                
                # Append values to channels
                wl_feats_val[w_idx][0].append(ld_power)
                wl_feats_val[w_idx][1].append(wl_caps.get(wl, 0))
                wl_feats_val[w_idx][2].append(rf.get('f_waste', 0))
                wl_feats_val[w_idx][3].append(float(wl_bypass_info.get(wl, 0)))
                wl_feats_val[w_idx][4].append(float(wl_dist_info.get(wl, 0.0)))

    # --- 第二步：批量赋值 (Vectorized Assignment) ---
    if u_list:
        # Global Features Assignment
        global_tensor[0, u_list, v_list] = dist_list
        global_tensor[1, u_list, v_list] = occ_list
        global_tensor[2, u_list, v_list] = nwls_list
        global_tensor[3, u_list, v_list] = fridge_list
        global_tensor[5, u_list, v_list] = hops_list
        
        # Wavelength Features Assignment
        for w_idx in range(num_wavelengths):
            us = wl_feats_u[w_idx]
            if not us: continue
            vs = wl_feats_v[w_idx]
            vals = wl_feats_val[w_idx] # list of 5 channels
            
            base_idx = w_idx * num_wl_feats
            
            # 一次性对该波长的所有边进行赋值
            # vals[c] 是一个列表，长度等于 us 的长度
            wl_tensor[base_idx + 0, us, vs] = vals[0]
            wl_tensor[base_idx + 1, us, vs] = vals[1]
            wl_tensor[base_idx + 2, us, vs] = vals[2]
            wl_tensor[base_idx + 3, us, vs] = vals[3]
            wl_tensor[base_idx + 4, us, vs] = vals[4]

    # [Modification]: 移除所有归一化/Log处理。
    # Raw Data In, Norm Data Out. 所有的数值变换都交给 Neural Network 的输入层处理。
    if remain_request_matrix is not None:
        global_tensor[6] = remain_request_matrix
    
    return global_tensor, wl_tensor

def build_auxiliary_graph_with_weights(topology, wavelength_list, traffic, physical_topology, shared_key_rate_list, served_request, remain_num_request, action_weights, node_to_idx, path_cache=None, ld_pos_cache=None):
    """
    RL 专用：先建立所有辅助边，最后统一使用 NN 输出的权重矩阵更新
    action_weights: [N, N]
    """
    if path_cache is None: path_cache = {}
    if ld_pos_cache is None: ld_pos_cache = {}

    ld_pos_cache.clear()
    auxiliary_graph = nx.MultiDiGraph()
    for node in topology.nodes():
        auxiliary_graph.add_node(node)
        
    network_slice = build_network_slice(wavelength_list=wavelength_list, topology=topology, traffic=traffic)
    config.key_rate_list = shared_key_rate_list
    
    # 1. 第一阶段：寻找并建立所有物理可行的辅助边
    all_raw_entries = []
    nodes = list(topology.nodes())
    for src, dst in itertools.product(nodes, nodes):
        if src == dst: continue
        
        path_list = get_cached_paths(physical_topology, src, dst, config.bypass, traffic, path_cache=path_cache)
        for path in path_list:
            if isinstance(path, int): continue # 防御性跳过
            
            candidates = []
            for wavelength in wavelength_list:
                wl_slice = network_slice[wavelength]
                min_cap = find_min_free_capacity(wl_slice, path)
                if min_cap <= 0: continue 
                if not check_path_validity_for_request(path, [wavelength], served_request): continue
                candidates.append({'wl': wavelength, 'cap': min_cap})
            
            if not candidates: 
                # [Optimization] 没有任何波长可用，剔除该路径
                cache_key_full = (src, dst)
                try:
                    if cache_key_full in path_cache and path in path_cache[cache_key_full]:
                        path_cache[cache_key_full].remove(path)
                except Exception:
                    pass
                continue
            
            candidates.sort(key=lambda x: x['cap'])
            n_candidates = len(candidates)
            suffix_max_cap = [0.0] * n_candidates
            curr_sum = 0.0
            for i in range(n_candidates - 1, -1, -1):
                curr_sum += candidates[i]['cap']
                suffix_max_cap[i] = curr_sum
            
            if suffix_max_cap[0] >= traffic:
                path_found_for_pair = False
                for found_wls, found_caps in _dfs_wavelength_combinations(0, traffic, candidates, suffix_max_cap, [], {}, 0):
                    current_pos_dict = {}
                    pos_valid = True
                    for wl in found_wls:
                        pos = find_laser_detector_position(network_slice[wl], path, wl, ld_pos_cache=ld_pos_cache)
                        if not pos:
                            pos_valid = False
                            break
                        current_pos_dict[wl] = pos
                    
                    if pos_valid:
                        # 优化：直接传递 topology 避免构建临时图 temp_G
                        raw_data = calculate_data_auxiliary_edge(
                            G=topology, path=path, laser_detector_position=current_pos_dict,
                            wavelength_combination=found_wls, wavelength_capacity=found_caps,
                            traffic=traffic, network_slice=network_slice, remain_num_request=remain_num_request,
                            topology=topology
                        )
                        if raw_data:
                            for entry in raw_data:
                                entry['src'] = src
                                entry['dst'] = dst
                                all_raw_entries.append(entry)
                            path_found_for_pair = True
                            break 
                
                if path_found_for_pair:
                    break # 找到有效路径后跳出 path_list 循环
                else:
                    # [Optimization] 如果该路径尝试了所有波长组合仍无法满足流量，
                    # 应该从缓存中剔除，避免后续无意义的重试
                    cache_key_full = (src, dst)
                    try:
                        if cache_key_full in path_cache and path in path_cache[cache_key_full]:
                            path_cache[cache_key_full].remove(path)
                    except Exception:
                        pass
            else:
                # [Optimization] 如果所有可用波长的容量总和都不足以承载当前请求，剔除该路径
                cache_key_full = (src, dst)
                try:
                    if cache_key_full in path_cache and path in path_cache[cache_key_full]:
                        path_cache[cache_key_full].remove(path)
                except Exception:
                    pass

    # 2. 第二阶段：统一应用权重矩阵
    for entry in all_raw_entries:
        u_idx, v_idx = node_to_idx[entry['src']], node_to_idx[entry['dst']]
        # 直接使用 NN 输出的权重 (范围 0~1)
        # 此时 NN 充当了一个完全端到端的“打分器”，全权负责评估这条边的价值
        nn_weight = float(action_weights[u_idx, v_idx])
        entry['weight'] = nn_weight
        auxiliary_graph.add_edge(entry['src'], entry['dst'], key=uuid.uuid4().hex, **entry)

    return auxiliary_graph

def generate_traffic(mid, topology):
    # 固定随机种子
    random.seed(config.random_seed)
    
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
    # 固定随机种子
    random.seed(config.random_seed)

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
    # 固定随机种子
    random.seed(config.random_seed)

    traffic = []
    num_pairs = len(pairs)

    # 生成流量值列表（按比例生成，不打乱顺序）
    num_20_percent = int(0.2 * num_pairs)
    num_60_percent = int(0.6 * num_pairs)
    num_20_percent_2 = num_pairs - num_20_percent - num_60_percent

    traffic_values = (
            [5000 - 0] * num_20_percent +
            [5000] * num_60_percent +
            [5000 + 0] * num_20_percent_2
    )
    random.shuffle(traffic_values)

    # 按排序后的顺序分配流量值
    id = 0
    for i in range(len(pairs)):
        traffic.append((id, pairs[i][0], pairs[i][1], traffic_values[i]))
        id = id + 1

    return traffic
