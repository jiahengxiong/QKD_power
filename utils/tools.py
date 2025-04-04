import copy
import itertools
import uuid
from itertools import product
from multiprocessing import Pool

import config
from QEnergy.studies.key_rate_compute import compute_key_rate
from QEnergy.studies.power_compute import compute_power
from utils.util import can_connect_path, calculate_distance
import networkx as nx
import random
from utils.custom_algorithm import  find_maximin_path


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
            if data['free_capacity'] > 0 and data['wavelength'] == wavelength:
                wavelength_slice.add_edge(src, dst, key=key, **data)
        for node in wavelength_slice.nodes:
            data = topology.nodes[node]
            # print(node, data)
            wavelength_slice.nodes[node].update(data)
        network_slice[wavelength] = wavelength_slice
    return network_slice


def combine_wavelength_slices(wavelength_list):
    all_combinations = []
    for r in range(1, len(wavelength_list) + 1):
        combinations = itertools.combinations(wavelength_list, r)
        all_combinations.extend([list(comb) for comb in combinations])
    return all_combinations


def build_multiple_wavelength_slices(wavelength_combinations, topology, traffic):
    virtual_graph = copy.deepcopy(topology)
    edges_to_remove = []

    # Remove edges not in wavelength combinations
    for src, dst, key, data in virtual_graph.edges(keys=True, data=True):
        if data['wavelength'] not in wavelength_combinations:
            edges_to_remove.append((src, dst, key))
    for src, dst, key in edges_to_remove:
        virtual_graph.remove_edge(src, dst, key=key)

    # Remove edges with no free capacity
    edges_to_remove = []
    for src, dst, key, data in virtual_graph.edges(keys=True, data=True):
        if data['free_capacity'] <= 0:
            keys = list(virtual_graph.get_edge_data(src, dst).keys())
            for k in keys:
                if (src, dst, k) not in edges_to_remove:
                    edges_to_remove.append((src, dst, k))
    for src, dst, key in edges_to_remove:
        virtual_graph.remove_edge(src, dst, key=key)

    # Remove all links between src and dst if total capacity < traffic
    edges_to_remove = []
    for src, dst in virtual_graph.edges():
        total_capacity = 0
        keys = list(virtual_graph.get_edge_data(src, dst).keys())
        for k in keys:
            total_capacity += virtual_graph.get_edge_data(src, dst, k)['free_capacity']
        if total_capacity < traffic:
            for k in keys:
                if (src, dst, k) not in edges_to_remove:
                    edges_to_remove.append((src, dst, k))
    for src, dst, key in edges_to_remove:
        virtual_graph.remove_edge(src, dst, key=key)

    return virtual_graph


def find_min_free_capacity(wavelength_slice, path):
    # 获取路径上所有边的free_capacity
    free_capacities = [
        wavelength_slice.edges[path[i], path[i + 1]]['free_capacity']
        for i in range(len(path) - 1)
    ]

    # 返回最小值
    return min(free_capacities)


def find_laser_detector_position(wavelength_slice, path, wavelength):
    # todo: justify the remaining key rate in a pair of Laser-Detector if biiger than traffic
    path_edge = []
    for i in range(len(path) - 1):
        path_edge.append((path[i], path[i + 1]))

    laser_detector = []
    for i in range(len(path) - 1):
        node = path[i]
        for detector_list in wavelength_slice.nodes[node]['laser'][wavelength]:
            # 检查所有节点是否都在路径中
            if all(item in path for item in detector_list):
                # 检查方向性
                indices = [path.index(item) for item in detector_list]  # 获取节点在path中的索引
                if indices == sorted(indices):  # 判断索引是否是递增的
                    laser_detector.append([node, detector_list[-1]])  # 添加符合条件的激光检测器

    if can_connect_path(path=path, laser_detector=laser_detector):
        laser_detector_position = [[None, None]]

        return laser_detector_position
    else:
        laser_detector_position = []
        pairs = [
            [path[i], path[j]]
            for i in range(len(path))
            for j in range(i + 1, len(path))
            if [path[i], path[j]] not in laser_detector
        ]
        for pair in pairs:
            possible_laser_detector = copy.deepcopy(laser_detector)
            possible_laser_detector.append(list(pair))
            if can_connect_path(path=path, laser_detector=possible_laser_detector):
                # print(path, pair,possible_laser_detector)
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

            # 处理多图和普通图的兼容性
            if G.is_multigraph():
                # 多图：默认选择第一个边（可根据需求调整逻辑，例如选择最短边）
                first_key = next(iter(edges_data))  # 取第一个边的键
                edge_distance = edges_data[first_key].get('distance', 0)
            else:
                # 普通图：直接获取唯一边的属性
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
    if config.detector == 'SNSPD':
        ice_box_power = 3000
    else:
        ice_box_power = 0
    laser_position = laser_detector_position['laser']
    detector_position = laser_detector_position['detector']
    if laser_position is not None and detector_position is not None:
        distance = calculate_distance(G=G, path=path, start=laser_position, end=detector_position)
        power = compute_power(distance=distance, protocol=config.protocol, receiver=config.detector)
        num_detector = G.nodes[detector_position]['num_detector']
        if num_detector % config.ice_box_capacity == 0:
            power = ice_box_power + power
    else:
        power = 0

    return power


def remove_possible_laser_detector_position(wavelength_laser_detector_list, path, G, traffic, network_slice):
    """
    laser_detector = []
    for i in range(len(path) - 1):
        node = path[i]
        for detector_list in wavelength_slice.nodes[node]['laser'][wavelength]:
            # 检查所有节点是否都在路径中
            if all(item in path for item in detector_list):
                # 检查方向性
                indices = [path.index(item) for item in detector_list]  # 获取节点在path中的索引
                if indices == sorted(indices):  # 判断索引是否是递增的
                    laser_detector.append([node, detector_list[-1]])  # 添加符合条件的激光检测器
    """
    # print(wavelength_laser_detector_list)
    for wavelength_laser_detector in wavelength_laser_detector_list:
        for wavelength, laser_detector in wavelength_laser_detector.items():
            AG = nx.DiGraph()
            existing_laser_detector = []
            wavelength_slice = network_slice[wavelength]
            for i in range(len(path) - 1):
                node = path[i]
                for detector_list in wavelength_slice.nodes[node]['laser'][wavelength]:
                    # 检查所有节点是否都在路径中
                    if all(item in path for item in detector_list):
                        # 检查方向性
                        indices = [path.index(item) for item in detector_list]  # 获取节点在path中的索引
                        if indices == sorted(indices):  # 判断索引是否是递增的
                            existing_laser_detector.append([node, detector_list[-1]])  # 添加符合条件的激光检测器
                            AG.add_edge(node, detector_list[-1], capacity=G.nodes[node]['laser_capacity'][detector_list])
            if laser_detector[0] is not None:
                AG.add_edge(laser_detector[0], laser_detector[1], capacity= calculate_keyrate(
                laser_detector_position={'laser': laser_detector[0], 'detector': laser_detector[1]}, path=path,
                G=network_slice[wavelength]))



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
            for wavelength, laser_detector in wavelength_laser_detector.items():
                wavelength_power = calculate_power(
                    laser_detector_position={'laser': laser_detector[0], 'detector': laser_detector[1]}, path=path,
                    G=network_slice[wavelength])
                power = power + wavelength_power
            distance = calculate_distance(G=G, path=path, start=path[0], end=path[-1])
            data.append({'power': power, 'path': path, 'laser_detector_position': wavelength_laser_detector,
                         'wavelength_traffic': wavelength_traffic_limitation,
                         'weight': power + 0.00001 * ((len(path) - 1)**1.01) + 0.0000001 * len(
                             wavelength_combination) + 0.000000000001 * (distance**1.01),
                         'wavelength_list': wavelength_combination,
                         'transverse_laser_detector':wavelength_used_laser_detector})

    return data


def build_multi_wavelength_auxiliary_graph(multi_wavelength_slice, network_slice, traffic, physical_topology,
                                           wavelength_combinations, auxiliary_graph):
    virtual_physical_topology = copy.deepcopy(physical_topology)
    for (src, dst) in physical_topology.edges():
        if multi_wavelength_slice.has_edge(src, dst) is False:
            virtual_physical_topology.remove_edge(src, dst)
    node_list = virtual_physical_topology.nodes
    # cutoff = calculate_cutoff(G=multi_wavelength_slice, traffic=traffic)
    for src in node_list:
        for dst in node_list:
            if src != dst:
                # all_original_paths = list(nx.all_simple_paths(source=src, target=dst, G=virtual_physical_topology))
                if config.bypass is True:
                    if nx.has_path(source=src, target=dst, G=virtual_physical_topology):
                        all_paths = [
                            nx.dijkstra_path(virtual_physical_topology, source=src, target=dst, weight='distance')]
                    else:
                        all_paths = []
                else:
                    all_paths = list(nx.all_simple_paths(source=src, target=dst, G=virtual_physical_topology, cutoff=1))
                # print(f"Starting build virtual link between {src} and {dst}, num of path: {len(all_paths)}")
                for path in all_paths:
                    wavelength_capacity = {}
                    laser_detector_position = {}
                    wavelength_keyrate = {}
                    free_capacity_constraint = 0
                    for wavelength in wavelength_combinations:
                        wavelength_slice = network_slice[wavelength]
                        min_free_capacity = find_min_free_capacity(wavelength_slice=wavelength_slice, path=path)
                        wavelength_capacity[wavelength] = min_free_capacity
                        free_capacity_constraint = free_capacity_constraint + min_free_capacity
                    if free_capacity_constraint < traffic:
                        continue
                    for wavelength in wavelength_combinations:
                        wavelength_slice = network_slice[wavelength]
                        laser_detector_position[wavelength] = find_laser_detector_position(
                            wavelength_slice=wavelength_slice, path=path,
                            wavelength=wavelength)

                    # todo: add edge in auxiliary_graph
                    data = calculate_data_auxiliary_edge(G=virtual_physical_topology, path=path,
                                                         laser_detector_position=laser_detector_position,
                                                         wavelength_combination=wavelength_combinations,
                                                         wavelength_capacity=wavelength_capacity,
                                                         traffic=traffic, network_slice=network_slice)
                    for edge_data in data:
                        auxiliary_graph.add_edge(src, dst, key=uuid.uuid4().hex + str(wavelength_combinations),
                                                 **edge_data)
                # print(f"Finished build virtual link between {src} and {dst}")


"""def build_auxiliary_graph(topology, protocol, wavelength_list, traffic, physical_topology):
    auxiliary_graph = nx.MultiDiGraph()
    for node in topology.nodes():
        auxiliary_graph.add_node(node)
    network_slice = build_network_slice(wavelength_list=wavelength_list, topology=topology, traffic=traffic)
    wavelength_combination_list = combine_wavelength_slices(wavelength_list=wavelength_list)
    for wavelength_combinations in tqdm(wavelength_combination_list, desc=f"Processing wavelength slice"):
        multi_wavelength_slice = build_multiple_wavelength_slices(wavelength_combinations=wavelength_combinations,
                                                                  topology=topology, traffic=traffic)
        build_multi_wavelength_auxiliary_graph(multi_wavelength_slice=multi_wavelength_slice,
                                               network_slice=network_slice, traffic=traffic,
                                               physical_topology=physical_topology,
                                               wavelength_combinations=wavelength_combinations,
                                               auxiliary_graph=auxiliary_graph)

    return auxiliary_graph
"""


def build_auxiliary_graph(topology, wavelength_list, traffic, physical_topology,shared_key_rate_list):
    auxiliary_graph = nx.MultiDiGraph()
    network_slice = build_network_slice(wavelength_list=wavelength_list, topology=topology, traffic=traffic)
    physical_topology = copy.deepcopy(topology)
    wavelength_combination_list = combine_wavelength_slices(wavelength_list=wavelength_list)
    protocol = config.protocol
    detector = config.detector
    ice_box_capacity = config.ice_box_capacity
    bypass = config.bypass

    with Pool(processes=32) as pool:
        # 使用 tqdm 包装 starmap 的进度条
        results = pool.starmap(process_wavelength_combination,
                               [(wavelength_combinations, topology, traffic, network_slice, physical_topology, protocol, detector, ice_box_capacity, bypass, shared_key_rate_list)
                                for wavelength_combinations in wavelength_combination_list])

    # 合并所有局部图
    auxiliary_graph = nx.compose_all(results)

    return auxiliary_graph


def process_wavelength_combination(wavelength_combinations, topology, traffic, network_slice, physical_topology,
                                   protocol, detector, ice_box_capacity, bypass, shared_key_rate_list):
    # 在每个进程中创建自己的局部图
    multi_wavelength_slice = build_multiple_wavelength_slices(wavelength_combinations=wavelength_combinations,
                                                              topology=topology, traffic=traffic)
    local_auxiliary_graph = nx.MultiDiGraph()  # 创建局部图
    config.protocol = protocol
    config.detector = detector
    config.ice_box_capacity = ice_box_capacity
    config.bypass = bypass
    config.key_rate_list = shared_key_rate_list
    for node in topology.nodes():
        local_auxiliary_graph.add_node(node)

    build_multi_wavelength_auxiliary_graph(multi_wavelength_slice=multi_wavelength_slice,
                                           network_slice=network_slice, traffic=traffic,
                                           physical_topology=physical_topology,
                                           wavelength_combinations=wavelength_combinations,
                                           auxiliary_graph=local_auxiliary_graph)
    # print(f"Finished processing wavelength combination: {wavelength_combinations}")

    return local_auxiliary_graph  # 返回局部图





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


def generate_and_sort_requests(topology):
    """生成节点对请求，并按规则排序"""
    node_list = list(topology.nodes)
    # random.shuffle(node_list)
    degrees = dict(topology.degree())

    # 生成所有无向节点对，并确定方向
    pairs = []
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            node_a, node_b = node_list[i], node_list[j]
            # 根据度数确定方向：度数小的作为发送方，大的作为接收方
            sender, receiver = (node_a, node_b) if degrees[node_a] < degrees[node_b] else (node_b, node_a)
            pairs.append((sender, receiver))

    # 计算路径长度和物理距离
    shortest_paths = {}
    physical_distances = {}
    for sender, receiver in pairs:
        # 最短跳数
        try:
            path_length = nx.shortest_path_length(topology, sender, receiver)
        except nx.NetworkXNoPath:
            path_length = float('inf')
        shortest_paths[(sender, receiver)] = path_length

        # 物理距离（假设边有权重属性 'distance'）
        try:
            distance = nx.shortest_path_length(topology, sender, receiver, weight='distance')
        except nx.NetworkXNoPath:
            distance = float('inf')
        physical_distances[(sender, receiver)] = distance

    # 排序规则：按接收方度数降序 > 最短跳数升序 > 发送方度数降序 > 物理距离升序
    pairs.sort(
        key=lambda x: (-degrees[x[1]], shortest_paths[x], -degrees[x[0]], physical_distances[x])
    )

    return pairs


def assign_traffic_values(pairs, mid):
    """根据排序后的请求顺序和 mid 值分配流量"""
    traffic = {}
    num_pairs = len(pairs)

    # 生成流量值列表（按比例生成，不打乱顺序）
    num_20_percent = int(0.2 * num_pairs)
    num_60_percent = int(0.6 * num_pairs)
    num_20_percent_2 = num_pairs - num_20_percent - num_60_percent

    traffic_values = (
            [mid - 1000] * num_20_percent +
            [mid] * num_60_percent +
            [mid + 1000] * num_20_percent_2
    )
    random.shuffle(traffic_values)

    # 按排序后的顺序分配流量值
    for idx, (sender, receiver) in enumerate(pairs):
        traffic[(sender, receiver)] = traffic_values[idx]

    return traffic