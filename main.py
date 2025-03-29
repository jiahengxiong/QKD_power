# from utils.tools import build_auxiliary_graph, generate_traffic
import sys

import networkx as nx
import numpy as np
from tqdm import tqdm

import config
import utils.tools
from utils.Network import Network
from utils.custom_algorithm import Dijkstra_single_path, Dijkstra_double_path
from utils.tools import calculate_keyrate, generate_and_sort_requests, assign_traffic_values


def find_min_weight_path_with_relay(auxiliary_graph, src, dst):
    paths = []  # 存储所有可能路径和属性

    # 辅助函数：计算路径总功率
    def calculate_path_power(path_edges):
        total_power = 0
        for u, v, key in path_edges:
            total_power += auxiliary_graph[u][v][key]['power']
        return total_power

    # 辅助函数：查找路径及其边信息，选择每对节点之间权重最小的边
    def extract_path_edges(path):
        edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # 获取所有多重边
            min_edge = min(auxiliary_graph[u][v].items(), key=lambda x: x[1]['weight'])  # 选择权重最小的边
            edges.append((u, v, min_edge[0]))  # 只取最小权重的边
        return edges

    # 1. src -> dst
    if nx.has_path(auxiliary_graph, src, dst):
        # path = nx.dijkstra_path(auxiliary_graph, source=src, target=dst, weight='weight')
        path_edges = Dijkstra_single_path(src=src, dst=dst, graph=auxiliary_graph)
        if path_edges:
            power_sum = calculate_path_power(path_edges)
            paths.append(('src->dst', path_edges, None, power_sum))  # 无中继

    # 2. dst -> src
    if nx.has_path(auxiliary_graph, dst, src):
        # path = nx.dijkstra_path(auxiliary_graph, source=dst, target=src, weight='weight')
        path_edges = Dijkstra_single_path(src=dst, dst=src, graph=auxiliary_graph)
        if path_edges:
            power_sum = calculate_path_power(path_edges)
            paths.append(('dst->src', path_edges, None, power_sum))  # 无中继

    # 3. 中继路径: src->delay 和 dst->delay
    for delay in auxiliary_graph.nodes:
        if delay != src and delay != dst:  # 排除src和dst自身
            # src -> delay 和 dst -> delay
            if nx.has_path(auxiliary_graph, src, delay) and nx.has_path(auxiliary_graph, dst, delay):
                """path1 = nx.dijkstra_path(auxiliary_graph, source=src, target=delay, weight='weight')
                path2 = nx.dijkstra_path(auxiliary_graph, source=dst, target=delay, weight='weight')"""

                path1_edges, path2_edges = Dijkstra_double_path(graph=auxiliary_graph, src=src, dst=dst, delay=delay)
                path_edges = path1_edges + path2_edges
                if path_edges:
                    power_sum = calculate_path_power(path_edges)
                    paths.append(('src->delay,dst->delay', path_edges, delay, power_sum))

            # dst -> delay 和 src -> delay
            if nx.has_path(auxiliary_graph, dst, delay) and nx.has_path(auxiliary_graph, src, delay):
                """path1 = nx.dijkstra_path(auxiliary_graph, source=dst, target=delay, weight='weight')
                path2 = nx.dijkstra_path(auxiliary_graph, source=src, target=delay, weight='weight')"""

                path1_edges, path2_edges = Dijkstra_double_path(graph=auxiliary_graph, src=dst, dst=src, delay=delay)
                path_edges = path1_edges + path2_edges
                if path_edges:
                    # path_edges = extract_path_edges(path1) + extract_path_edges(path2)
                    power_sum = calculate_path_power(path_edges)
                    paths.append(('dst->delay,src->delay', path_edges, delay, power_sum))

    # 找到power最小的路径
    if paths:
        best_path_info = min(paths, key=lambda x: x[3])  # 按照power最小值排序
        return best_path_info
    else:
        return None


def serve_traffic(G, AG, path_edge_list, request_traffic, pbar):
    for (src, dst, key) in path_edge_list:
        edge_data = AG.get_edge_data(u=src, v=dst, key=key)
        # print(f' src: {src}, dst: {dst}, key: {key}, edge_data: {edge_data}')
        wavelength_list = edge_data['wavelength_list']
        path = edge_data['path']
        edge_traffic = request_traffic
        edge_laser_detector_list = edge_data['transverse_laser_detector']
        for wavelength in wavelength_list:
            laser_postion = edge_data['laser_detector_position'][wavelength][0]
            detector_postion = edge_data['laser_detector_position'][wavelength][1]
            traffic_limitation = edge_data['wavelength_traffic'][wavelength]
            trans_traffic = min(edge_traffic, traffic_limitation)

            if laser_postion is not None and detector_postion is not None:
                laser_index = path.index(laser_postion)
                detector_index = path.index(detector_postion)
                if laser_index >= detector_index:
                    tqdm.write(f'ERROR !!! Laser is after Detector')
                cover_links = path[laser_index:detector_index + 1]
                pbar.write(
                    f"{wavelength}: laser-detector: {laser_postion} -> {detector_postion}, cover links: {cover_links}")
                if cover_links not in G.nodes[laser_postion]['laser'][wavelength]:
                    G.nodes[laser_postion]['laser'][wavelength].append(cover_links)
                    # todo: add the key rate of new laser_detector (cover links)
                    G.nodes[laser_postion]['laser_capacity'][wavelength][tuple(cover_links)] = calculate_keyrate(
                        laser_detector_position={'laser': laser_postion, 'detector': detector_postion}, path=path, G=G)
                G.nodes[detector_postion]['num_detector'] += 1

            # add traffic in each wavelength slice
            for i in range(len(path) - 1):
                source = path[i]
                destination = path[i + 1]
                edges = G.get_edge_data(source, destination)
                for edge_key, edge_attrs in edges.items():
                    if edge_attrs.get('wavelength') == wavelength:
                        G.edges[source, destination, edge_key]['free_capacity'] -= trans_traffic
                        pbar.write(
                            f"{wavelength}: {source} -> {destination} with {trans_traffic}, {G.edges[source, destination, edge_key]['free_capacity']}, {G.edges[source, destination, edge_key]['capacity']}")
            for transverse_laser_detector in edge_laser_detector_list[wavelength]:
                # print("edge_laser_detector_list: ", edge_laser_detector_list)
                source = transverse_laser_detector[0]
                G.nodes[source]['laser_capacity'][wavelength][tuple(transverse_laser_detector)] -= trans_traffic
            edge_traffic -= trans_traffic
            # todo add the traffic consumption of laser detector
            # add laser-detector cover links, add detector
            """if cover_links not in G.nodes[laser_postion]['key_rate'].keys():
                G.nodes[laser_postion]['key_rate'][cover_links] = compute_key_rate(laser_postion, detector_postion)"""


def main():
    wavelength_list = np.linspace(1530, 1565, 10).tolist()
    print(
        f"----------CONFIG: map: {map_name}, Protocol: {config.protocol}, Detector: {config.detector}, Bypass: {config.bypass}----------")

    # 定义运行次数
    num_runs = 8
    # 用于存储每次运行的结果
    all_results = {}

    for run in range(num_runs):
        print(f"\n--- 第 {run + 1} 次运行 ---")
        flag = True
        mid = 0
        result_average_power = {}
        network = Network(map_name=map_name, wavelength_list=wavelength_list, protocol=config.protocol,
                          receiver=config.detector)
        topology = network.topology
        physical_topology = network.physical_topology
        pairs = generate_and_sort_requests(physical_topology)


        while flag:
            mid = mid + 5000
            result_average_power[mid] = 0.0
            network = Network(map_name=map_name, wavelength_list=wavelength_list, protocol=config.protocol,
                              receiver=config.detector)
            topology = network.topology
            physical_topology = network.physical_topology
            # traffic_matrix = utils.tools.generate_traffic(topology=physical_topology, mid=mid)
            traffic_matrix = assign_traffic_values(pairs=pairs, mid=mid)
            with tqdm(total=len(traffic_matrix), file=sys.stdout, colour="red") as pbar:
                for (src, dst), traffic in traffic_matrix.items():
                    auxiliary_graph = utils.tools.build_auxiliary_graph(topology=topology,
                                                                        wavelength_list=wavelength_list,
                                                                        traffic=traffic,
                                                                        physical_topology=physical_topology)

                    # 找出最优路径和最小功率
                    result = find_min_weight_path_with_relay(auxiliary_graph=auxiliary_graph, src=src, dst=dst)
                    if result:
                        direction, best_path_edges, relay, min_power = result
                        if relay:
                            pbar.write(
                                f"从 {src} 到 {dst} 最优路径(方向: {direction}, 中继: {relay}): {best_path_edges}, 最小功率: {min_power}")

                        else:
                            pbar.write(
                                f"从 {src} 到 {dst} 最优路径(方向: {direction}): {best_path_edges}, 最小功率: {min_power}")
                        serve_traffic(G=topology, AG=auxiliary_graph, path_edge_list=best_path_edges,
                                      request_traffic=traffic, pbar=pbar)
                        result_average_power[mid] += min_power / len(traffic_matrix)

                        pbar.update(1)
                    else:
                        pbar.write(f"从 {src} 到 {dst} 无可用路径")
                        pbar.update(1)
                        del result_average_power[mid]
                        flag = False
                        break

            if mid >= 105000:
                flag = False
        # 将当前运行的结果存储到 all_results 中
        for mid, power in result_average_power.items():
            if mid not in all_results:
                all_results[mid] = []
            all_results[mid].append(power)
        print(f"\n--- 第 {run + 1} 次运行结果 ---\n{all_results}")

    # 过滤掉在部分运行中缺失的 mid
    valid_results = {}
    for mid, powers in all_results.items():
        if len(powers) >= num_runs:  # 只有十次运行中都存在的 mid 才有效
            valid_results[mid] = powers

    # 计算平均值
    average_results = {}
    for mid, powers in valid_results.items():
        average_results[mid] = sum(powers) / len(powers)

    # 输出最终结果
    print(
        f'\n--- 最终结果（{num_runs} 次运行的平均值）---\n'
        f'Protocol: {config.protocol}, Bypass: {config.bypass}, Detector: {config.detector}, Map: {map_name}\n'
        f'{average_results}'
    )
    with open('result.txt', 'a') as file:
        file.write(f'\n--- 最终结果（{num_runs} 次运行的平均值）---\n')
        file.write(
            f'Protocol: {config.protocol}, Bypass: {config.bypass}, Detector: {config.detector}, Map: {map_name}\n')
        file.write(f'{average_results}\n')


if __name__ == '__main__':
    topology_type = ['Paris', 'Large']
    detector_list = ['APD', 'SNSPD']
    protocol_list = ['BB84', 'E91', 'CV-QKD']
    bypass_list = [True, False]
    for protocol in protocol_list:
        if protocol == 'CV-QKD':
            detector = 'ThorlabsPDB'
            for map_name in topology_type:
                for bypass in bypass_list:
                    config.protocol = protocol
                    config.detector = detector
                    config.bypass = bypass
                    config.ice_box_capacity = 8
                    main()
        else:
            for detector in detector_list:
                for map_name in topology_type:
                    for bypass in bypass_list:
                        config.protocol = protocol
                        config.detector = detector
                        config.bypass = bypass
                        config.ice_box_capacity = 8
                        main()
