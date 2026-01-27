import copy
import itertools
import random
import uuid
from itertools import product

import networkx as nx

import config
from QEnergy.studies.key_rate_compute import compute_key_rate
from QEnergy.studies.power_compute import compute_power
from utils.util import can_connect_path, calculate_distance, calculate_cutoff
from numba import jit
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Pool, Manager


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
        pairs = [[path[i], path[j]] for i in range(len(path)) for j in range(i + 1, len(path))]
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
        laser_index = path.index(laser_position)  # 获取 i 的索引
        detector_index = path.index(detector_position)  # 获取 j 的索引
        cover_path = path[laser_index:detector_index + 1]
        distance = 0
        for i in range(len(cover_path) - 1):
            src = cover_path[i]
            dst = cover_path[i + 1]
            distance += G.edges[src, dst]['distance']
        bypass_number = len(cover_path) - 2

        key_rate = compute_key_rate(protocol=config.protocol, receiver=config.detector, distance=distance) * (
                0.89 ** bypass_number)
    else:
        key_rate = compute_key_rate(protocol=config.protocol, receiver=config.detector, distance=0.00001)

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


def calculate_data_auxiliary_edge(G, path, wavelength_combination, wavelength_capacity, laser_detector_position,
                                  traffic, network_slice):
    # print(path, laser_detector_position)
    data = []
    keys = laser_detector_position.keys()
    values = laser_detector_position.values()
    wavelength_laser_detector_list = [dict(zip(keys, combo)) for combo in product(*values)]
    # print(laser_detector_position)
    for wavelength_laser_detector in wavelength_laser_detector_list:
        wavelength_traffic_limitation = {}
        max_traffic = 0
        for wavelength, laser_detector in wavelength_laser_detector.items():
            # distance = calculate_distance(G=G, path=path, start=laser_detector[0], end=laser_detector[1])
            key_rate = calculate_keyrate(
                laser_detector_position={'laser': laser_detector[0], 'detector': laser_detector[1]}, path=path,
                G=network_slice[wavelength])
            wavelength_traffic_limitation[wavelength] = min(wavelength_capacity[wavelength], key_rate)
            max_traffic = max_traffic + wavelength_traffic_limitation[wavelength]
        if max_traffic >= traffic:
            power = 0
            for wavelength, laser_detector in wavelength_laser_detector.items():
                wavelength_power = calculate_power(
                    laser_detector_position={'laser': laser_detector[0], 'detector': laser_detector[1]}, path=path,
                    G=network_slice[wavelength])
                power = power + wavelength_power
            data.append({'power': power, 'path': path, 'laser_detector_position': wavelength_laser_detector,
                         'wavelength_traffic': wavelength_traffic_limitation,
                         'weight': power + 0.001 * len(path) + 0.000001 * len(wavelength_combination),
                         'wavelength_list': wavelength_combination})

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
                    all_paths = [nx.dijkstra_path(virtual_physical_topology, source=src, target=dst, weight='distance')]
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
                        break
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
                        auxiliary_graph.add_edge(src, dst, key=uuid.uuid4().hex, **edge_data)
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

import networkx as nx
from tqdm.contrib.concurrent import thread_map  # 或使用 multiprocessing.dummy.Pool
import multiprocessing


def build_auxiliary_graph(topology, wavelength_list, traffic, physical_topology, shared_key_rate_list, served_request,remain_num_request, link_future_demand=None):
    """
    与 tools.py 保持一致
    """
    # 简单的转发到 .py 版本的逻辑或者直接复制逻辑
    # 考虑到用户环境可能有编译好的 .so 文件，最稳妥的方法是让 .pyx 的签名也对上
    pass


def process_wavelength_combination(wavelength_combinations, topology, traffic, network_slice, physical_topology):
    # 在每个进程中创建自己的局部图
    multi_wavelength_slice = build_multiple_wavelength_slices(wavelength_combinations=wavelength_combinations,
                                                              topology=topology, traffic=traffic)
    local_auxiliary_graph = nx.MultiDiGraph()  # 创建局部图
    config.protocol = 'BB84'
    config.detector = 'APD'
    config.ice_box_capacity = 8
    config.bypass = True
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
    traffic = {}

    num_pairs = len(node_list) * (len(node_list) - 1) // 2
    num_30_percent = int(0.3 * num_pairs)
    num_40_percent = int(0.4 * num_pairs)
    num_30_percent_2 = num_pairs - num_30_percent - num_40_percent

    traffic_values = ([mid - 1000] * num_30_percent +
                      [mid] * num_40_percent +
                      [mid + 1000] * num_30_percent_2)

    pairs = []
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            pairs.append((node_list[i], node_list[j]))

    random.shuffle(pairs)

    for idx, (node_a, node_b) in enumerate(pairs):
        traffic_value = traffic_values[idx]
        traffic[(node_a, node_b)] = traffic_value

    return traffic
