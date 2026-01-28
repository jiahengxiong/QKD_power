# from utils.tools import build_auxiliary_graph, generate_traffic
import gc
import math
import random
import itertools
import networkx as nx
from utils.traffic_generater import gen_traffic_matrix
import config
import utils.tools
from utils.Network import Network
from utils.custom_algorithm import Dijkstra_single_path, Dijkstra_double_path
from utils.tools import calculate_keyrate, generate_and_sort_requests, assign_traffic_values
import copy

from multiprocessing import Process, Manager
from tqdm import tqdm
import sys
import numpy as np
import os




def average_component_power(component_power_run):
    if not component_power_run:
        return {}

    # 初始化所有 key 的累加器
    keys = component_power_run[0].keys()
    total = {key: 0 for key in keys}

    # 累加每个 key 的值
    for item in component_power_run:
        for key in keys:
            total[key] += item.get(key, 0)

    # 求平均
    count = len(component_power_run)
    average = {key: total[key] / count for key in keys}
    return average


def find_min_weight_path_with_relay(auxiliary_graph, src, dst):
    # 辅助函数：计算路径总功率
    def calculate_path_power(path_edges):
        total_power = 0
        for u, v, key in path_edges:
            total_power += auxiliary_graph[u][v][key]['power']
        return total_power

    paths = []  # 存储所有可能路径和属性

    # 1. 检查 src -> dst 的直接路径
    if nx.has_path(auxiliary_graph, src, dst):
        path_edges, weight = Dijkstra_single_path(src=src, dst=dst, graph=auxiliary_graph)
        if path_edges:
            power_sum = calculate_path_power(path_edges)
            paths.append(('src->dst', path_edges, None, power_sum, weight))

    # 2. 检查 dst -> src 的直接路径
    if nx.has_path(auxiliary_graph, dst, src):
        path_edges, weight = Dijkstra_single_path(src=dst, dst=src, graph=auxiliary_graph)
        if path_edges:
            power_sum = calculate_path_power(path_edges)
            paths.append(('dst->src', path_edges, None, power_sum, weight))

        # 3. 检查中继路径
        for delay in auxiliary_graph.nodes:
            if delay != src and delay != dst:
                # 情形1：src -> delay 和 dst -> delay
                if nx.has_path(auxiliary_graph, src, delay) and nx.has_path(auxiliary_graph, dst, delay):
                    path1_edges, path2_edges, weight = Dijkstra_double_path(graph=auxiliary_graph, src=src, dst=dst, delay=delay)
                    path_edges = path1_edges + path2_edges
                    if path_edges:
                        power_sum = calculate_path_power(path_edges)
                        paths.append(('src->delay,dst->delay', path_edges, delay, power_sum, weight))

                # 情形2：dst -> delay 和 src -> delay
                if nx.has_path(auxiliary_graph, dst, delay) and nx.has_path(auxiliary_graph, src, delay):
                    path1_edges, path2_edges, weight = Dijkstra_double_path(graph=auxiliary_graph, src=dst, dst=src, delay=delay)
                    path_edges = path1_edges + path2_edges
                    if path_edges:
                        power_sum = calculate_path_power(path_edges)
                        paths.append(('dst->delay,src->delay', path_edges, delay, power_sum, weight))

    if not paths:
        return False

    # 按权重排序 (index 4 是 weight)
    sorted_paths = sorted(paths, key=lambda x: x[4])
    best_path = sorted_paths[0]

    return best_path


def serve_traffic(G, AG, path_edge_list, request_traffic, pbar, served_request):
    occupied_wavelength = 0
    for (src, dst, key) in path_edge_list:
        edge_data = AG.get_edge_data(u=src, v=dst, key=key)
        # print(f' src: {src}, dst: {dst}, key: {key}, edge_data: {edge_data}')
        wavelength_list = edge_data['wavelength_list']
        path = edge_data['path']
        edge_traffic = request_traffic
        edge_laser_detector_list = edge_data['transverse_laser_detector']
        for wavelength in wavelength_list:
            if wavelength not in list(served_request.keys()):
                served_request[wavelength] = []
            laser_postion = edge_data['laser_detector_position'][wavelength][0]
            detector_postion = edge_data['laser_detector_position'][wavelength][1]
            traffic_limitation = edge_data['wavelength_traffic'][wavelength]
            trans_traffic = min(edge_traffic, traffic_limitation)
            
            if trans_traffic <= 0:
                print(f"ERROR!!! trans_traffic = {trans_traffic}")
                continue

            if laser_postion is not None and detector_postion is not None:
                laser_index = path.index(laser_postion)
                detector_index = path.index(detector_postion)
                if laser_index >= detector_index:
                    tqdm.write(f'ERROR !!! Laser is after Detector')
                cover_links = path[laser_index:detector_index + 1]
                pbar.write(
                    f"{wavelength}: laser-detector: {laser_postion} -> {detector_postion}, cover links: {cover_links}")
                if cover_links not in G.nodes[laser_postion]['laser'][wavelength]:
                    new_list = list(zip(cover_links, cover_links[1:]))
                    served_request[wavelength].append(new_list)
                    G.nodes[laser_postion]['laser'][wavelength].append(cover_links)
                    G.nodes[detector_postion]['detector'][wavelength].append(cover_links)
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
                        if G.edges[source, destination, edge_key]['occupied'] is False:
                            G.edges[source, destination, edge_key]['occupied'] = True
                            occupied_wavelength += 1
                        pbar.write(
                            f"{wavelength}: {source} -> {destination} with {trans_traffic}, {G.edges[source, destination, edge_key]['free_capacity']}, {G.edges[source, destination, edge_key]['capacity']}")
            for transverse_laser_detector in edge_laser_detector_list[wavelength]:
                # print("edge_laser_detector_list: ", edge_laser_detector_list)
                source = transverse_laser_detector[0]
                G.nodes[source]['laser_capacity'][wavelength][tuple(transverse_laser_detector)] -= trans_traffic
                if G.nodes[source]['laser_capacity'][wavelength][tuple(transverse_laser_detector)] < 0:
                    pbar.write('ERROR!!!')
            edge_traffic -= trans_traffic
            # todo add the traffic consumption of laser detector
            # add laser-detector cover links, add detector
            """if cover_links not in G.nodes[laser_postion]['key_rate'].keys():
                G.nodes[laser_postion]['key_rate'][cover_links] = compute_key_rate(laser_postion, detector_postion)"""
            for transverse_laser_detector in edge_laser_detector_list[wavelength]:
                for i in range(len(transverse_laser_detector) - 1):
                    edges = G.get_edge_data(transverse_laser_detector[i], transverse_laser_detector[i+1])
                    for edge_key, edge_attrs in edges.items():
                        if edge_attrs.get('wavelength') == wavelength:
                            G.edges[transverse_laser_detector[i], transverse_laser_detector[i+1], edge_key]['free_capacity'] = min(G.edges[transverse_laser_detector[i], transverse_laser_detector[i+1], edge_key]['free_capacity'],G.nodes[transverse_laser_detector[0]]['laser_capacity'][wavelength][tuple(transverse_laser_detector)])

    return occupied_wavelength


from multiprocessing import Pool, Manager
from tqdm import tqdm
import sys
import numpy as np
import os


# 请确保以下对象已正确导入：
# Network, assign_traffic_values, generate_and_sort_requests,
# utils.tools.build_auxiliary_graph, find_min_weight_path_with_relay, serve_traffic
# 同时，全局变量 map_name 和 config 模块需要在工程中预先定义

def calculate_dynamic_heatmap(topology, physical_topology, future_requests, wavelength_list, served_request, physical_betweenness):
    """
    自适应动态热力图：
    1. 统计路径上所有节点的未来需求。
    2. 结合物理拓扑的“介数中心性” (Topological Betweenness)，计算节点的综合战略系数。
    """
    link_demand = {}
    node_demand = {}
    total_weighted_traffic = 0.0
    
    decay_base = 1.0 # 目前设定为 1.0 以获得最长远视野
    
    # --- 构建“枢纽引力”寻路权重 ---
    # 在预测热力图时，强迫流量向枢纽节点汇聚
    strategic_weight_attr = 'strategic_weight'
    for u, v, d in physical_topology.edges(data=True):
        # 目标节点的介数中心性越高，这条边的虚拟权重就越小 (引力越大)
        target_node = v
        btwn = physical_betweenness.get(target_node, 0)
        # 虚拟权重 = 物理距离 / (1 + 10 * 战略地位)
        d[strategic_weight_attr] = d['distance'] / (1.0 + 10.0 * btwn)

    for step, req in enumerate(future_requests):
        r_id, r_src, r_dst, r_traffic = req
        weight = math.pow(decay_base, step)
        weighted_traffic = r_traffic * weight
        total_weighted_traffic += weighted_traffic
        
        # 关键改进：使用 'strategic_weight' 而非 'distance' 来探测路径
        # 这会让热力图预测未来流量高度聚集在枢纽节点
        paths = utils.tools.find_first_valid_physical_path(
            topology=topology,
            physical_topology=physical_topology,
            src=r_src,
            dst=r_dst,
            traffic=r_traffic,
            wavelength_list=wavelength_list,
            served_request=served_request,
            weight=strategic_weight_attr
        )
        
        if paths:
            if isinstance(paths[0], (int, str)):
                paths = [paths]
            for path in paths:
                # 1. 链路热度
                edges = list(zip(path[:-1], path[1:]))
                for u, v in edges:
                    link_demand[(u, v)] = link_demand.get((u, v), 0) + weighted_traffic
                    link_demand[(v, u)] = link_demand.get((v, u), 0) + weighted_traffic
                # 2. 节点热度
                for node in path:
                    node_demand[node] = node_demand.get(node, 0) + weighted_traffic
                    
    # 计算综合战略系数：拓扑天赋 (Betweenness) * 后天努力 (Dynamic Heat)
    node_strategic_coeffs = {}
    if total_weighted_traffic > 0:
        for node, demand in node_demand.items():
            # 获取该节点的介数中心性 (Topological Importance)
            btwn = physical_betweenness.get(node, 0.0)
            
            # 归一化动态需求
            dynamic_heat = demand / total_weighted_traffic
            
            # 综合系数 = 拓扑重要性与流量需求的乘积
            # 这里取 btwn 的平方根或适当缩放以平衡量级，通常 btwn 较小 (0~0.1)
            # 我们希望这个系数能反映“如果我不在这里中继，我错失了多大的全局复用潜力”
            node_strategic_coeffs[node] = btwn * dynamic_heat
            
    return link_demand, node_strategic_coeffs

def process_mid(traffic_type, map_name, protocol, detector, bypass, key_rate_list, wavelength_list, num_runs,
                ice_box_capacity, request_list):
    """
    对给定的 mid 内部执行 num_runs 次模拟，计算每次的结果后取平均，
    最终将该 mid 对应的平均值写入 shared_results。
    """

    # 在子进程中导入 config 并初始化所有需要的属性
    import config
    config.protocol = protocol
    config.detector = detector
    config.bypass = bypass
    config.key_rate_list = key_rate_list
    config.ice_box_capacity = ice_box_capacity
    flag = True

    # 打印当前进程ID及开始处理的信息

    total_power_run = []  # 用于保存每次 run 的结果
    shared_results = {}
    spectrum_occupied_run = []
    component_power_run = []


    network = Network(map_name=map_name,
                      wavelength_list=wavelength_list,
                      protocol=protocol,
                      receiver=detector)
    physical_topology = network.get_physical_topology(map_name=map_name)
    pairs = generate_and_sort_requests(physical_topology)
    mid = config.Traffic_cases[map_name][traffic_type]
    print(f"[PID {os.getpid()}] Starting processing Protocol: {protocol} Bypass:{bypass} Topology:{map_name} Traffic Type: {traffic_type}")

    # --- 预计算物理拓扑的介数中心性 (Topological Backbone) ---
    # 介数中心性反映了节点在网络中最短路径桥梁的重要性，是“战略枢纽”的最佳拓扑指标
    physical_betweenness = nx.betweenness_centrality(physical_topology, weight='distance')

    for run in range(num_runs):

        total_power_each_run = 0.0
        component_power = {'source': 0, 'detector': 0, 'other': 0, 'ice_box': 0}
        spectrum_occupied = 0.0

        # 每次 run 重新构造网络实例
        network = Network(map_name=map_name,
                          wavelength_list=wavelength_list,
                          protocol=protocol,
                          receiver=detector)
        topology = network.topology
        physical_topology = network.physical_topology

        # 根据当前 mid 生成流量矩阵（所有 run 使用相同的 pairs）
        # traffic_matrix = assign_traffic_values(pairs=pairs, mid=mid)
        traffic_matrix_base = request_list[run][traffic_type][map_name]
        
        # 不再调整服务顺序，直接使用原始流量矩阵
        traffic_matrix = traffic_matrix_base

        served_request = {}
        # print(traffic_matrix)
        max_traffic = mid + 1000

        # 使用 tqdm 显示当前 run 中该 mid 的进度，输出到 sys.stderr
        with tqdm(total=len(traffic_matrix), file=sys.stderr, colour="red",
                  desc=f"mid {mid} run {run + 1}/{num_runs}") as pbar:
            remain_num_request = len(traffic_matrix)
            
            # --- 1. 构建剩余需求热力图 (Future Demand Heatmap) ---
            # 预计算所有剩余请求的最短路径，统计每条物理链路被“未来”经过的次数
            # 这不需要非常精确（比如考虑波长），只需要基于物理距离的拓扑统计
            # link_future_demand = {(u, v): traffic_weighted_count}
            
            # --- 3. 顺序处理请求 (Greedy with Dynamic Heatmap) ---
            for i, request in enumerate(traffic_matrix):
                id = request[0]
                src = request[1]
                dst = request[2]
                traffic = request[3]
                
                # --- 动态重新计算后续所有请求的热力图 ---
                if i % 10 == 0:
                    future_requests = traffic_matrix[i+1:]
                    link_future_demand, node_future_demand = calculate_dynamic_heatmap(
                        topology=topology,
                        physical_topology=physical_topology,
                        future_requests=future_requests,
                        wavelength_list=wavelength_list,
                        served_request=served_request,
                        physical_betweenness=physical_betweenness
                    )

                auxiliary_graph = utils.tools.build_auxiliary_graph(
                    topology=topology,
                    wavelength_list=wavelength_list,
                    traffic=traffic,
                    physical_topology=physical_topology,
                    shared_key_rate_list=key_rate_list,
                    served_request=served_request,
                    remain_num_request=remain_num_request,
                    link_future_demand=link_future_demand,
                    node_future_demand=node_future_demand
                )
                result = find_min_weight_path_with_relay(auxiliary_graph=auxiliary_graph, src=src, dst=dst)

                if result:
                    direction, best_path_edges, relay, min_power, weight = result
                    if relay:
                        pbar.write(
                            f"[PID {os.getpid()}] 从 {src} 到 {dst} 最优路径(方向: {direction}, 中继: {relay}): {best_path_edges}, 最小功率: {min_power}")
                    else:
                        pbar.write(
                            f"[PID {os.getpid()}] 从 {src} 到 {dst} 最优路径(方向: {direction}): {best_path_edges}, 最小功率: {min_power}")
                    occupied_wavelength = serve_traffic(G=topology,
                                                        AG=auxiliary_graph,
                                                        path_edge_list=best_path_edges,
                                                        request_traffic=traffic,
                                                        pbar=pbar,
                                                        served_request = served_request)
                    
                    total_power_each_run += min_power / len(traffic_matrix)
                    spectrum_occupied += occupied_wavelength / network.num_wavelength
                    for (u, v, key) in best_path_edges:
                        edge_data = auxiliary_graph.get_edge_data(u=u, v=v, key=key)
                        component_power['source'] = component_power['source'] + edge_data['source_power'] / len(traffic_matrix)
                        component_power['detector'] = component_power['detector'] + edge_data['detector_power'] / len(traffic_matrix)
                        component_power['other'] = component_power['other'] + edge_data['other_power'] / len(traffic_matrix)
                        component_power['ice_box'] = component_power['ice_box'] + edge_data['ice_box_power'] / len(traffic_matrix)

                    pbar.update(1)
                else:
                    pbar.write(f"[PID {os.getpid()}] 从 {src} 到 {dst} 无可用路径")
                    pbar.update(1)
                    flag = False
                    del auxiliary_graph
                    gc.collect()
                    with open(f'result.txt', 'a') as file:
                        file.write(f'\n--- 最终结果 --- \n')
                        file.write(
                            f'Protocol: {config.protocol}, Map: {map_name}\n')
                        file.write(f'无可用路径\n')
                    return
                del auxiliary_graph
                gc.collect()
                remain_num_request = remain_num_request - 1

        if flag:
            total_power_run.append(total_power_each_run)
            spectrum_occupied_run.append(spectrum_occupied)
            component_power_run.append(component_power)

    if total_power_run:
        # 修改统计逻辑：取最小值而不是平均值 (Best of Runs)
        # 找到最小功耗对应的索引
        min_power_idx = total_power_run.index(min(total_power_run))
        
        avg_value = total_power_run[min_power_idx]
        avg_occupied_spectrum = spectrum_occupied_run[min_power_idx]
        avg_component_power = component_power_run[min_power_idx] # 这里不再取平均，而是取最佳 Run 的组件功耗
        
        shared_results['traffic'] = mid
        shared_results['total_avg_power'] = avg_value
        shared_results['avg_spectrum_occupied'] = avg_occupied_spectrum
        shared_results['avg_component_power'] = avg_component_power

        output_str = (
            f'\n--- 最终结果 (经过 {num_runs} 次运行后取最佳) ---\n'
            f'Protocol: {config.protocol}, Bypass: {config.bypass}, Detector: {config.detector}, '
            f'Map: {map_name}, Traffic: {traffic_type}\n'
            f'{shared_results}\n'
        )
        
        print(f"[PID {os.getpid()}] 任务完成: {output_str}")
        
        with open('result.txt', 'a') as file:
            file.write(output_str)
            file.flush() # 强制刷新到磁盘，确保结果不丢失
            os.fsync(file.fileno()) # 确保操作系统层面的同步


    print(f"[PID {os.getpid()}] Finished processing mid {mid}")



def main():
    import config
    # 生成波长列表
    wavelength_list = np.linspace(1530, 1565, 10).tolist()

    # 每个 mid 内部的运行次数
    num_runs = 1 

    manager = Manager()
    # 创建共享字典用于 key_rate（按原逻辑使用）
    shared_key_rate = {}
    config.key_rate_list = shared_key_rate
    topology_list = ['Large', 'Paris', 'Tokyo']
    protocol_list = ['BB84', 'CV-QKD']
    initial_traffic_list = ['Low']
    traffic_type_list = ['Low', 'High', 'Medium',]
    request_dic = {}
    # for run in range(num_runs):
    #     request_dic[run] = {}
    #     for traffic_type in traffic_type_list:
    #         request_dic[run][traffic_type] = {}
    #         for topology in topology_list:
    #             request_list = gen_traffic_matrix(traffic_type, topology)
    #             request_dic[run][traffic_type][topology] = request_list
    for run in range(num_runs):
        request_dic[run] = {}
        for traffic_type in traffic_type_list:
            request_dic[run][traffic_type] = {}
            for topology in topology_list:
                if traffic_type == 'Low':
                    request_list = gen_traffic_matrix(traffic_type, topology)
                    request_dic[run][traffic_type][topology] = request_list
                else:
                    request_list =  request_dic[run]['Low'][topology]
                    request_dic[run][traffic_type][topology] = []
                    for request in request_list:
                        gap = config.Traffic_cases[topology][traffic_type] - config.Traffic_cases[topology]['Low']
                        request_dic[run][traffic_type][topology].append((request[0], request[1], request[2], request[3]+gap))


    # print(request_dic[0]['Low']['Test'])
    # print(request_dic)


    # 根据当前网络配置生成用于计算请求对的列表 pairs（所有 mid 使用相同的 pairs）
    # network = Network(map_name=map_name,
    #                   wavelength_list=wavelength_list,
    #                   protocol=config.protocol,
    #                   receiver=config.detector)
    # physical_topology = network.physical_topology
    # pairs = generate_and_sort_requests(physical_topology)

    # 创建跨进程共享的 flag 和存放结果的共享字典
    # shared_results = manager.dict()

    # 定义所有待测试的 mid 值（5000 到 110000，步长5000）
    # mids = list(range(1200000, 1300000, 20000))
    # 构造任务参数列表，将 config.bypass 与 config.ice_box_capacity 一并传入
    config.ice_box_capacity = 8
    args_list = []
    for case in config.cases:
        map_name = case['Topology']
        config.detector = case['Detector']
        config.protocol = case['Protocol']
        config.bypass = case['Bypass']
        traffic = case['Traffic']
        # traffic_matrix = gen_traffic_matrix(mid=traffic,map_name=map_name,wavelength_list=wavelength_list,detector=config.detector)
        args_list.append((traffic, map_name, config.protocol, config.detector, config.bypass, shared_key_rate, wavelength_list,
         num_runs, config.ice_box_capacity, request_dic))



    with Pool(processes=16) as pool:
        pool.starmap(process_mid, args_list)

    # 将最终结果从共享字典转换为普通字典
    # average_results = dict(shared_results)
    #
    # print(f'\n--- 最终结果（每个 mid 经过 {num_runs} 次运行后取平均）---\n'
    #       f'Protocol: {config.protocol}, Bypass: {config.bypass}, Detector: {config.detector}, Map: {map_name}\n'
    #       f'{average_results}')
    # with open('result.txt', 'a') as file:
    #     file.write(f'\n--- 最终结果（每个 mid 经过 {num_runs} 次运行后取平均）---\n')
    #     file.write(f'Protocol: {config.protocol}, Bypass: {config.bypass}, Detector: {config.detector}, Map: {map_name}\n')
    #     file.write(f'{average_results}\n')


if __name__ == '__main__':
    main()
    # 根据 protocol、detector、map_name、bypass 进行多重循环配置
    # topology_type = ['Tokyo']
    # detector_list = ['APD']
    # protocol_list = ['BB84']
    # bypass_list = [False]
    #
    # import config  # 确保 config 模块能正确导入
    #
    # for protocol in protocol_list:
    #     if protocol == 'CV-QKD':
    #         detector = 'ThorlabsPDB'
    #         for map_name in topology_type:
    #             for bypass in bypass_list:
    #                 config.protocol = protocol
    #                 config.detector = detector
    #                 config.bypass = bypass
    #                 config.ice_box_capacity = 8  # 设置冰箱容量
    #                 main()
    #     else:
    #         for detector in detector_list:
    #             for map_name in topology_type:
    #                 for bypass in bypass_list:
    #                     config.protocol = protocol
    #                     config.detector = detector
    #                     config.bypass = bypass
    #                     config.ice_box_capacity = 8
    #                     main()
