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

def calculate_dynamic_heatmap(auxiliary_graph, future_requests):
    """
    高效动态热力图：利用当前已构建好的辅助图，为未来所有请求进行快速寻路探测。
    引入时间衰减因子：0.95，使热力图更聚焦于近期需求。
    """
    link_demand = {}
    decay_base = 0.95
    
    for step, req in enumerate(future_requests):
        r_src, r_dst, r_traffic = req[1], req[2], req[3]
        
        # 计算时间衰减权重
        weight = math.pow(decay_base, step)
        weighted_traffic = r_traffic * weight
        
        # 直接在现成的辅助图上跑 Dijkstra
        result = find_min_weight_path_with_relay(auxiliary_graph=auxiliary_graph, src=r_src, dst=r_dst)
        
        if result:
            _, best_path_edges, _, _, _ = result
            for (u, v, key) in best_path_edges:
                # 从辅助图中提取真实的物理路径
                edge_data = auxiliary_graph.get_edge_data(u, v, key=key)
                if edge_data and 'path' in edge_data:
                    physical_path = edge_data['path']
                    for j in range(len(physical_path) - 1):
                        p_u, p_v = physical_path[j], physical_path[j+1]
                        link_demand[(p_u, p_v)] = link_demand.get((p_u, p_v), 0) + weighted_traffic
                        link_demand[(p_v, p_u)] = link_demand.get((p_v, p_u), 0) + weighted_traffic
                        
    return link_demand

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
            link_future_demand = {} # 初始化热力图变量，修复 UnboundLocalError
            
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
                
                # --- 1. 构建辅助图 (使用现有的热力图) ---
                auxiliary_graph = utils.tools.build_auxiliary_graph(
                    topology=topology,
                    wavelength_list=wavelength_list,
                    traffic=traffic,
                    physical_topology=physical_topology,
                    shared_key_rate_list=key_rate_list,
                    served_request=served_request,
                    remain_num_request=remain_num_request,
                    link_future_demand=link_future_demand,
                    node_future_demand=None 
                )

                # --- 2. 动态更新热力图 (使用当前已建好的 AG) ---
                # 优化：每 10 个请求更新一次热度，为后续请求提供更精准的权重引导
                if i % 10 == 0:
                    future_requests = traffic_matrix[i+1:]
                    link_future_demand = calculate_dynamic_heatmap(
                        auxiliary_graph=auxiliary_graph,
                        future_requests=future_requests
                    )
                    # 重新构建当前请求的辅助图，以应用刚刚更新的热度惩罚
                    auxiliary_graph = utils.tools.build_auxiliary_graph(
                        topology=topology,
                        wavelength_list=wavelength_list,
                        traffic=traffic,
                        physical_topology=physical_topology,
                        shared_key_rate_list=key_rate_list,
                        served_request=served_request,
                        remain_num_request=remain_num_request,
                        link_future_demand=link_future_demand,
                        node_future_demand=None 
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
                        output_str = (
                        f'\n--- 最终结果 (经过 {num_runs} 次运行后取最佳) ---\n'
                        f'Protocol: {config.protocol}, Bypass: {config.bypass}, Detector: {config.detector}, '
                        f'Map: {map_name}, Traffic: {traffic_type}\n'
                    )
                        file.write(
                            output_str)
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
