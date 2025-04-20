import networkx as nx

from utils.Network import Network
import config

import random

import random

import random
import numpy as np


def sort_traffic_matrix(topology, traffic_matrix):
    import networkx as nx

    # 原始度数
    raw_degrees = dict(topology.degree())

    # 计算增强度
    enhanced_degrees = {}
    for node in topology.nodes:
        neighbors = list(topology.neighbors(node))
        neighbor_degree_sum = sum(raw_degrees[n] for n in neighbors)
        enhanced_degrees[node] = raw_degrees[node] + 0.1 * neighbor_degree_sum

    shortest_paths = {}
    physical_distances = {}

    # 计算路径长度和物理距离
    for src, dst, _ in traffic_matrix:
        try:
            path_length = nx.shortest_path_length(topology, src, dst)
        except nx.NetworkXNoPath:
            path_length = float('inf')
        shortest_paths[(src, dst)] = path_length

        try:
            distance = nx.shortest_path_length(topology, src, dst, weight='distance')
        except nx.NetworkXNoPath:
            distance = float('inf')
        physical_distances[(src, dst)] = distance

    # 排序规则：接收方增强度降序 > 发送方增强度降序 > 最短跳数升序 > 物理距离升序 > traffic_value升序
    # traffic_matrix.sort(
    #     key=lambda x: (
    #         -enhanced_degrees.get(x[1], 0),     # 接收方增强度降序
    #         -enhanced_degrees.get(x[0], 0),     # 发送方增强度降序
    #         shortest_paths.get((x[0], x[1]), float('inf')),  # 最短跳数升序
    #         physical_distances.get((x[0], x[1]), float('inf')),  # 物理距离升序
    #         x[2]  # traffic_value升序
    #     )
    # )
    return traffic_matrix
def gen_traffic_matrix(mid, map_name, wavelength_list=None, protocol='BB84', detector='APD'):
    if wavelength_list is None:
        wavelength_list = [1]
    network = Network(map_name=map_name,
                      wavelength_list=wavelength_list,
                      protocol=protocol,
                      receiver=detector)
    G = network.physical_topology
    node_list = list(G.nodes())
    random.shuffle(node_list)
    if map_name == 'Test':
        traffic_matrix = []
        for i in range(999999):
            traffic_matrix.append((i, 0, 2, 100000))
            traffic_matrix.append((i, 0, 1, 100000))

        return traffic_matrix

    # 计算总流量与 traffic 候选值
    base_traffic = config.Traffic_cases[map_name][mid]
    total_traffic = base_traffic * (len(node_list) * (len(node_list) - 1)) / 2
    min_traffic = 1*base_traffic - 1000
    max_traffic = 1*base_traffic + 1000
    traffic_list = list(map(int, np.linspace(min_traffic, max_traffic, 3)))

    traffic_matrix = []
    remaining_traffic = total_traffic
    id = 0

    while remaining_traffic > 0:
        src = random.choice(node_list)
        dst = random.choice(node_list)
        while dst == src:
            dst = random.choice(node_list)
            src = random.choice(node_list)

        if remaining_traffic < min(traffic_list):
            traffic_matrix.append([id, src, dst, remaining_traffic])
            break

        possible_values = [t for t in traffic_list if t <= remaining_traffic]
        traffic_value = random.choice(possible_values)

        traffic_matrix.append((id, src, dst, traffic_value))
        id += 1
        remaining_traffic -= traffic_value

    random.shuffle(traffic_matrix)

    # ✅ 根据 src-dst 最短路径长度排序（距离短的排在前面）
    traffic_matrix.sort(key=lambda x: nx.shortest_path_length(G, x[1], x[2]))

    return traffic_matrix

if __name__ == '__main__':
    print(gen_traffic_matrix('Low', 'Large', [1], 'BB84', 'APD'))
    print(len(gen_traffic_matrix('Low', 'Large', [1], 'BB84', 'APD')))