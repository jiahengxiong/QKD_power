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
    return network_slice


def combine_wavelength_slices(wavelength_list, topology, traffic, network_slice):
    combined_capacity = {}
    for u, v, _ in topology.edges(keys=True):
        combined_capacity[(u, v)] = 0
        combined_capacity[(v, u)] = 0

    selected_wavelengths = []
    last_selected_idx = None

    for i, wavelength in enumerate(wavelength_list):
        network_wavelength = network_slice[wavelength]
        free_capacity_list = [network_wavelength.edges[edge]['free_capacity']
                              for edge in network_wavelength.edges]

        if max(free_capacity_list, default=0) <= 0:
            continue

        for edge in network_wavelength.edges:
            cap = network_wavelength.edges[edge].get('free_capacity', 0)
            combined_capacity[edge] += cap
            combined_capacity[(edge[1], edge[0])] += cap

        selected_wavelengths.append(wavelength)
        last_selected_idx = i

        if min(combined_capacity.values(), default=0) >= traffic:
            break

    combination_wls = list(selected_wavelengths)
    if last_selected_idx is not None and last_selected_idx + 2 < len(wavelength_list):
        extra_wl = wavelength_list[last_selected_idx + 2]
        combination_wls.append(extra_wl)

    all_combinations = []

    n = len(wavelength_list)
    for i in range(n):
        for j in range(i + 1, n + 1):
            all_combinations.append(wavelength_list[i:j])

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
            key_rate = (key_rate * (1 ** bypass_number))
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
            G=network_slice), detector_list=cover_links)
    capacity, used_laser_detector_list = find_maximin_path(AG, path[0], path[-1])
    """if laser_detector in used_laser_detector:
        used_laser_detector.remove(laser_detector)"""
    recovery_detector_list = []
    for used_laser_detector in used_laser_detector_list:
        recovery_detector_list.append(AG.edges[used_laser_detector[0], used_laser_detector[1]]['detector_list'])
    del  AG
    

    return capacity, recovery_detector_list


def calculate_data_auxiliary_edge(G, path, wavelength_combination, wavelength_capacity, laser_detector_position,
                                  traffic, network_slice):
    # print(path, laser_detector_position)
    data = []
    keys = laser_detector_position.keys()
    values = laser_detector_position.values()
    wavelength_laser_detector_list = [dict(zip(keys, combo)) for combo in product(*values)]
    # remove_possible_laser_detector_position(wavelength_laser_detector_list=wavelength_laser_detector_list, path=path, G=G, traffic=traffic, network_slice=network_slice)
    # print(wavelength_laser_detector_list)
    for wavelength_laser_detector in wavelength_laser_detector_list:
        wavelength_traffic_limitation = {}
        max_traffic = 0
        wavelength_used_laser_detector = {}
        for wavelength, laser_detector in wavelength_laser_detector.items():
            # distance = calculate_distance(G=G, path=path, start=laser_detector[0], end=laser_detector[1])
            key_rate, used_laser_detector = Max_capacity(laser_detector=laser_detector,path=path,G=G, wavelength=wavelength, network_slice=network_slice[wavelength])
            """key_rate = calculate_keyrate(
                laser_detector_position={'laser': laser_detector[0], 'detector': laser_detector[1]}, path=path,
                G=network_slice[wavelength])"""
            wavelength_traffic_limitation[wavelength] = min(wavelength_capacity[wavelength], key_rate)
            max_traffic = max_traffic + wavelength_traffic_limitation[wavelength]
            wavelength_used_laser_detector[wavelength] = used_laser_detector

        """for wavelength, laser_detector in wavelength_used_laser_detector.items():
            if len(laser_detector) != 0:
                print(wavelength, laser_detector)"""
        if max_traffic >= traffic:
            power = 0
            source_power = 0
            detector_power = 0
            other_power = 0
            ice_box_power = 0
            spectrum = 0
            LD = 0
            used_LD = 0
            for wavelength, laser_detector in wavelength_laser_detector.items():
                component_power = calculate_power(
                    laser_detector_position={'laser': laser_detector[0], 'detector': laser_detector[1]}, path=path,
                    G=network_slice[wavelength])
                power = power + component_power['total']
                source_power = source_power + component_power['source']
                detector_power = detector_power + component_power['detector']
                other_power = other_power + component_power['other']
                ice_box_power = ice_box_power + component_power['ice_box']
                if laser_detector[0] is not None:
                    LD = LD+(path.index(laser_detector[1]) - path.index(laser_detector[0]) - 1)
                used_LD = len(wavelength_used_laser_detector[wavelength]) + used_LD
                for i in range(len(path) - 1):
                    src = path[i]
                    dst = path[i + 1]
                    # 获取 src 和 dst 之间的所有边的 key 和数据
                    # 获取 src 和 dst 之间的所有边的 key 和数据
                    key_list = list(G.edges(src, dst, keys=True))

                    for _, _, key, _ in key_list:
                        # 确保边存在并且访问数据
                        if G.has_edge(src, dst, key):
                            edge_data = G.get_edge_data(src, dst, key)
                            # print(f"key: {key}, edge data: {edge_data}")  # 打印边的数据以便检查

                            # 检查'wavelength'和'occupied'条件
                            if edge_data.get('wavelength') == wavelength and not edge_data.get('occupied', False):
                                spectrum += 1
            distance = calculate_distance(G=G, path=path, start=path[0], end=path[-1])
            # bonus_score = 0
            # for bonus_wavelength in wavelength_combination:
            #     bonus_G = network_slice[bonus_wavelength]
            #     centrality = nx.betweenness_centrality(bonus_G, weight="distance")
            #     for bonus_node in path:
            #         bonus_score = bonus_score + centrality[bonus_node]
            if config.bypass:
                bypass = 1
            else:
                bypass = 0


            data.append({'power': power,
                         'source_power': source_power,
                         'detector_power': detector_power,
                         'other_power': other_power,
                         'ice_box_power': ice_box_power,
                         'path': path, 'laser_detector_position': wavelength_laser_detector,
                         'wavelength_traffic': wavelength_traffic_limitation,
                         'weight':
                             power
                             * (traffic/max_traffic)
                             # * (10 + (LD)/used_LD)
                             # * (10 + (1e-4)*(len(wavelength_combination)) ** 1.0 * (len(path) - 1) ** 1.0/(spectrum+1))
                             *(10 + (1e-2)*(len(wavelength_combination)) ** 1.0 * (len(path) - 1) ** 1.0)
                             # * bypass

                             # + spectrum
                             + (len(wavelength_combination)) ** 1.0 * (len(path) - 1) ** 1.0
                             # - 10**(-3)
                             # + 1e-2 * distance
                             # + 10 * len(wavelength_combination)
                             # + 1e-8 * sum(wavelength_combination)
                             # + 1e-4 * min(wavelength_combination) * (len(path) - 1)**1.01
                             # + 1e-6 * max(wavelength_combination)
                                 # + 1e-16 * max_traffic
                         ,
                         'wavelength_list': wavelength_combination,
                         'transverse_laser_detector':wavelength_used_laser_detector})

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

def build_auxiliary_graph(topology, wavelength_list, traffic, physical_topology, shared_key_rate_list, served_request):
    """
    重构后的主函数
    逻辑：Node Pairs -> K Paths -> Wavelength Combinations
    """
    auxiliary_graph = nx.MultiDiGraph()
    
    # 1. 预处理
    network_slice = build_network_slice(wavelength_list=wavelength_list, topology=topology, traffic=traffic)
    wavelength_combination_list = combine_wavelength_slices(wavelength_list=wavelength_list, topology=topology, traffic=traffic, network_slice=network_slice)
    
    # 全局配置更新
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
    # 使用 tqdm 显示进度
    # for src, dst in tqdm(pairs, desc="Building Auxiliary Graph"):
    for src, dst in pairs:
        # 3.1 生成 K 条路径 (基于物理拓扑)
        k_val = 5 # 可以根据需要调整 K 值
        if config.bypass is True:
             # 如果 bypass 开启，尝试找 K 条路
             raw_paths = get_k_shortest_paths(physical_topology, src, dst, k=k_val, weight='distance')
        else:
             # 如果 bypass 关闭，只找直连 (cutoff=1)
             try:
                 raw_paths = list(nx.all_simple_paths(source=src, target=dst, G=physical_topology, cutoff=1))
             except:
                 raw_paths = []

        path_success_flag = False # 标记当前节点对是否搞定
        
        # 3.2 遍历每一条路径
        for path in raw_paths:
            path_processed_in_any_wavelength = False
            
            # 3.3 遍历每个波长组合
            for wavelength_combinations in wavelength_combination_list:
                
                # --- A. 检查 Request 约束 ---
                if not check_path_validity_for_request(path, wavelength_combinations, served_request):
                    continue

                # --- B. 检查容量 ---
                wavelength_capacity = {}
                free_capacity_constraint = 0
                capacity_check_pass = True
                
                for wavelength in wavelength_combinations:
                    wl_slice = network_slice[wavelength]
                    min_cap = find_min_free_capacity(wavelength_slice=wl_slice, path=path)
                    if min_cap <= 0:
                        capacity_check_pass = False
                        break
                    wavelength_capacity[wavelength] = min_cap
                    free_capacity_constraint += min_cap
                
                if not capacity_check_pass:
                    continue
                if free_capacity_constraint < traffic:
                    continue
                
                # --- C. 检查硬件位置 ---
                laser_detector_position = {}
                hardware_check_pass = True
                for wavelength in wavelength_combinations:
                    wl_slice = network_slice[wavelength]
                    positions = find_laser_detector_position(
                        wavelength_slice=wl_slice, path=path, wavelength=wavelength
                    )
                    if len(positions) < 1:
                        hardware_check_pass = False
                        break
                    laser_detector_position[wavelength] = positions
                
                if not hardware_check_pass:
                    continue
                
                # --- D. 建立辅助边 ---
                # 构建临时小图用于计算
                temp_G = build_temp_graph_for_path(topology, path, wavelength_combinations)
                
                data = calculate_data_auxiliary_edge(
                    G=temp_G, 
                    path=path,
                    laser_detector_position=laser_detector_position,
                    wavelength_combination=wavelength_combinations,
                    wavelength_capacity=wavelength_capacity,
                    traffic=traffic,
                    network_slice=network_slice
                )
                
                if data:
                    for edge_data in data:
                        auxiliary_graph.add_edge(path[0], path[-1], key=uuid.uuid4().hex + str(wavelength_combinations),
                                                 **edge_data)
                    path_processed_in_any_wavelength = True
                    break
                if path_processed_in_any_wavelength:
                    break
            
            # 3.4 逻辑判断
            if path_processed_in_any_wavelength:
                path_success_flag = True
                break # 退出路径循环，处理下一个节点对

    # 清理内存
    del network_slice
    gc.collect()

    return auxiliary_graph


# ==========================================
# 流量生成函数 (保持不变)
# ==========================================

def generate_traffic(mid, topology):
    node_list = list(topology.nodes)
    random.shuffle(node_list)
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