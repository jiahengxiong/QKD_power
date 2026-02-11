import networkx as nx

from utils.Network import Network
import config

import random

import random
import math
import random
import numpy as np


def sort_traffic_matrix(topology, traffic_matrix, perturbation=0.0):
    import networkx as nx
    import random

    # 原始度数
    raw_degrees = dict(topology.degree())

    # 计算增强度
    enhanced_degrees = {}
    for node in topology.nodes:
        neighbors = list(topology.neighbors(node))
        neighbor_degree_sum = sum(raw_degrees[n] for n in neighbors)
        enhanced_degrees[node] = raw_degrees[node] + 0.1 * neighbor_degree_sum

    # 预处理 traffic_matrix：增强度高的作为 dst，低的作为 src
    processed_matrix = []
    for id, src, dst, traffic_value in traffic_matrix:
        if enhanced_degrees[src] > enhanced_degrees[dst]:
            src, dst = dst, src  # swap
        processed_matrix.append((id, src, dst, traffic_value))

    shortest_paths = {}
    physical_distances = {}

    # 计算路径长度和物理距离
    for id, src, dst, *_ in processed_matrix:
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

    # 排序规则：接收方增强度降序 > 发送方增强度降序 > 最短跳数升序 > 物理距离升序 > traffic_value降序
    processed_matrix.sort(
        key=lambda x: (
            # shortest_paths.get((x[1], x[2]), float('inf')),  # 最短跳数升序 (Shortest Path First)
            physical_distances.get((x[1], x[2]), float('inf')),  # 物理距离升序
            # -enhanced_degrees.get(x[2], 0),  # 接收方(dst)增强度降序 (x[2] is dst)
            # -enhanced_degrees.get(x[1], 0),  # 发送方(src)增强度降序 (x[1] is src)
            # -x[3]  # traffic_value降序 (High Traffic First)
        )
    )

    # # 引入随机扰动，跳出局部最优
    # if perturbation > 0:
    #     n = len(processed_matrix)
    #     # 交换次数与 perturbation 成正比
    #     num_swaps = int(n * perturbation)
    #     for _ in range(num_swaps):
    #         i = random.randint(0, n - 1)
    #         j = random.randint(0, n - 1)
    #         processed_matrix[i], processed_matrix[j] = processed_matrix[j], processed_matrix[i]

    return processed_matrix


import random
import numpy as np


def gen_traffic_matrix(mid, map_name, wavelength_list=None, protocol='BB84', detector='APD', seed=42):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    if wavelength_list is None:
        wavelength_list = [1]
    network = Network(map_name=map_name,
                      wavelength_list=wavelength_list,
                      protocol=protocol,
                      receiver=detector)
    G = network.physical_topology
    node_list = list(G.nodes())
    # random.shuffle(node_list)

    if map_name == 'Test':
        traffic_matrix = []
        for i in range(999999):
            traffic_matrix.append((i, 0, 2, 100000))
            traffic_matrix.append((i, 0, 1, 100000))
        return traffic_matrix

    base_traffic = config.Traffic_cases[map_name][mid]

    traffic_matrix = []
    id = 0
    pair_list = [(node_list[i], node_list[j]) for i in range(len(node_list)) for j in range(i + 1, len(node_list))]

    select_factor = 1.0
    pair_list = random.sample(pair_list, k=math.floor(len(pair_list)*select_factor))
    # random.shuffle(pair_list)
    total_pairs = len(pair_list)

    # 初始比例设置
    low_ratio = 0.2
    high_ratio = 0.2
    base_ratio = 1 - low_ratio - high_ratio

    # 计算数量（微调保证平均值为base_traffic）
    # 设 n_total 为总数，设 low, base, high 为三种类型数量
    # 求解使得 (low*(base-1000) + base*base + high*(base+1000)) / n_total = base
    # 解得 low = high，base = n_total - 2*low
    # 所以我们设 low = high = floor(0.2 * n_total)，base = n_total - 2*low

    low_count = int(low_ratio * total_pairs)
    high_count = int(high_ratio * total_pairs)
    base_count = total_pairs - low_count - high_count

    # 生成流量值列表
    traffic_values = (
            [base_traffic - 0] * low_count +
            [base_traffic + 0] * high_count +
            [base_traffic] * base_count
    )
    # random.shuffle(traffic_values)

    for (src, dst), traffic in zip(pair_list, traffic_values):
        traffic_matrix.append((id, src, dst, traffic/select_factor))
        id += 1

    # random.shuffle(traffic_matrix)
    # ✅ 根据 src-dst 最短路径长度排序（距离短的排在前面）
    # traffic_matrix.sort(key=lambda x: nx.shortest_path_length(G, x[1], x[2], weight='distance'))
    traffic_matrix = sort_traffic_matrix(G, traffic_matrix)

    # debug=[(0, 'Adachi', 'Ota', 1000)]
    # return debug

    return traffic_matrix

# if __name__ == '__main__':
#     print(gen_traffic_matrix('Low', 'Large', [1], 'BB84', 'APD'))
#     print(len(gen_traffic_matrix('Low', 'Large', [1], 'BB84', 'APD')))