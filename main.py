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
import ctypes




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


def find_min_weight_path_with_relay(auxiliary_graph, src, dst, top_k=3):
    # 辅助函数：计算路径总功率
    def calculate_path_power(path_edges_with_data):
        total_power = 0
        for u, v, data in path_edges_with_data:
            total_power += data['power']
        return total_power
    
    # 辅助函数：将 (u, v, key) 列表转换为 (u, v, data) 列表
    def attach_edge_data(path_keys):
        path_with_data = []
        for u, v, key in path_keys:
            data = auxiliary_graph.get_edge_data(u, v, key)
            path_with_data.append((u, v, data))
        return path_with_data

    paths = []  # 存储所有可能路径和属性

    # 1. 检查 src -> dst 的直接路径
    if nx.has_path(auxiliary_graph, src, dst):
        path_edges, weight = Dijkstra_single_path(src=src, dst=dst, graph=auxiliary_graph)
        if path_edges:
            path_with_data = attach_edge_data(path_edges)
            power_sum = calculate_path_power(path_with_data)
            paths.append(('src->dst', path_with_data, None, power_sum, weight))

    # 2. 检查 dst -> src 的直接路径
    if nx.has_path(auxiliary_graph, dst, src):
        path_edges, weight = Dijkstra_single_path(src=dst, dst=src, graph=auxiliary_graph)
        if path_edges:
            path_with_data = attach_edge_data(path_edges)
            power_sum = calculate_path_power(path_with_data)
            paths.append(('dst->src', path_with_data, None, power_sum, weight))

        # 3. 检查中继路径
        for delay in auxiliary_graph.nodes:
            if delay != src and delay != dst:
                # 情形1：src -> delay 和 dst -> delay
                if nx.has_path(auxiliary_graph, src, delay) and nx.has_path(auxiliary_graph, dst, delay):
                    path1_edges, path2_edges, weight = Dijkstra_double_path(graph=auxiliary_graph, src=src, dst=dst, delay=delay)
                    if path1_edges and path2_edges:
                        path_edges = path1_edges + path2_edges
                        path_with_data = attach_edge_data(path_edges)
                        power_sum = calculate_path_power(path_with_data)
                        paths.append(('src->delay,dst->delay', path_with_data, delay, power_sum, weight))

                # 情形2：dst -> delay 和 src -> delay
                if nx.has_path(auxiliary_graph, dst, delay) and nx.has_path(auxiliary_graph, src, delay):
                    path1_edges, path2_edges, weight = Dijkstra_double_path(graph=auxiliary_graph, src=dst, dst=src, delay=delay)
                    if path1_edges and path2_edges:
                        path_edges = path1_edges + path2_edges
                        path_with_data = attach_edge_data(path_edges)
                        power_sum = calculate_path_power(path_with_data)
                        paths.append(('dst->delay,src->delay', path_with_data, delay, power_sum, weight))

    if not paths:
        return []

    # 按权重 (weight) 排序，权重越小越优
    sorted_paths = sorted(paths, key=lambda x: x[4])
    
    # 返回前 top_k 条候选路径
    return sorted_paths[:top_k]


def serve_traffic(G, path_with_data, request_traffic, pbar, served_request):
    occupied_wavelength = 0
    for (src, dst, edge_data) in path_with_data:
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
        
        # 引入随机扰动：
        # Run 0: 纯贪心 (perturbation=0)
        # Run 1~N: 逐渐增加扰动 (perturbation=0.05 ~ 0.2)
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
            
            link_future_demand = {}
            node_future_demand = {} # 新增：宿节点热度统计
            
            # 初始统计所有请求
            request_paths_cache = {} # id -> list of (edges, weight)
            
            # 强化主路径权重，压缩尾部权重
            raw_weights = [0.8, 0.1, 0.05, 0.03, 0.02]
            
            for req in traffic_matrix:
                r_id, r_src, r_dst, r_traffic = req # 获取请求流量
                
                # 统计宿节点热度
                node_future_demand[r_dst] = node_future_demand.get(r_dst, 0) + r_traffic
                
                # 计算 K 条最短路径 (K=5)
                try:
                    k_paths_gen = nx.shortest_simple_paths(physical_topology, r_src, r_dst, weight='distance')
                    k_paths = list(itertools.islice(k_paths_gen, 5)) 
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    k_paths = []

                if not k_paths:
                    request_paths_cache[r_id] = []
                    continue

                # 动态归一化：确保实际存在的路径权重和为 1.0
                actual_raw = raw_weights[:len(k_paths)]
                total_w = sum(actual_raw)
                norm_weights = [w / total_w for w in actual_raw]

                path_data_list = []
                for idx, path in enumerate(k_paths):
                    edges = list(zip(path[:-1], path[1:]))
                    weight = norm_weights[idx]
                    path_data_list.append((edges, weight))
                    
                    for u, v in edges:
                        # 按权重 * 流量 增加热度
                        impact = weight * r_traffic
                        link_future_demand[(u, v)] = link_future_demand.get((u, v), 0) + impact
                        link_future_demand[(v, u)] = link_future_demand.get((v, u), 0) + impact
                
                request_paths_cache[r_id] = path_data_list

            # --- 3. 顺序处理请求 (Branch & Bound Backtracking) ---
            i = 0
            state_stack = []        # 存放状态快照的栈
            tried_path_indices = {}   # 记录每个请求索引当前尝试到第几条候选路径
            candidates_cache = {}     # 缓存每个请求在特定状态下的候选路径列表
            
            # 全局最优记录
            min_total_power_found = float('inf')
            best_solution_found = None # 存储最佳方案的所有路径和组件功耗
            
            # 最大尝试总步数，防止死循环或耗时过长
            total_steps = 0
            MAX_TOTAL_STEPS = 5000 
            
            pbar.write(f"[PID {os.getpid()}] 开始 Branch & Bound 搜索最低功耗方案...")

            while i < len(traffic_matrix) and total_steps < MAX_TOTAL_STEPS:
                total_steps += 1
                
                # 如果是第一次到达这个请求，或者由于回溯重新到达，初始化尝试索引
                if i not in tried_path_indices:
                    tried_path_indices[i] = 0
                    # 保存当前状态快照
                    state_stack.append({
                        'topology': topology.copy(),
                        'served_request': copy.deepcopy(served_request),
                        'total_power_each_run': total_power_each_run,
                        'spectrum_occupied': spectrum_occupied,
                        'component_power': copy.deepcopy(component_power),
                        'link_future_demand': link_future_demand.copy(),
                        'node_future_demand': node_future_demand.copy(),
                        'remain_num_request': remain_num_request
                    })

                # 获取快照作为基准
                snapshot = state_stack[-1]
                
                # 代价剪枝：如果当前累计功耗已经超过已找到的最优解，直接回溯
                if total_power_each_run >= min_total_power_found:
                    pbar.write(f"  [剪枝] 请求 {i}: 当前功耗 {total_power_each_run:.2f} >= 最优 {min_total_power_found:.2f}")
                    # 触发回溯逻辑（同下方的 else 分支）
                    if i in tried_path_indices: del tried_path_indices[i]
                    if i in candidates_cache: del candidates_cache[i]
                    state_stack.pop()
                    if not state_stack: break
                    i -= 1
                    tried_path_indices[i] += 1
                    # 恢复上一步状态
                    prev = state_stack[-1]
                    topology = prev['topology'].copy()
                    served_request = copy.deepcopy(prev['served_request'])
                    total_power_each_run = prev['total_power_each_run']
                    spectrum_occupied = prev['spectrum_occupied']
                    component_power = copy.deepcopy(prev['component_power'])
                    link_future_demand = prev['link_future_demand'].copy()
                    node_future_demand = prev['node_future_demand'].copy()
                    remain_num_request = prev['remain_num_request']
                    pbar.update(-1)
                    continue

                request = traffic_matrix[i]
                id, src, dst, traffic = request[0], request[1], request[2], request[3]

                # 恢复热力图并扣减当前请求
                link_future_demand = snapshot['link_future_demand'].copy()
                node_future_demand = snapshot['node_future_demand'].copy()
                node_future_demand[dst] = max(0, node_future_demand.get(dst, 0) - traffic)
                if id in request_paths_cache:
                    for edges, weight in request_paths_cache[id]:
                        for u, v in edges:
                            impact = weight * traffic
                            link_future_demand[(u, v)] = max(0, link_future_demand.get((u, v), 0) - impact)
                            link_future_demand[(v, u)] = max(0, link_future_demand.get((v, u), 0) - impact)

                # 获取或生成候选路径
                if i not in candidates_cache:
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
                    # 现在 find_min_weight_path_with_relay 返回的是 (direction, path_with_data, relay, min_power, weight)
                    # path_with_data 是 [(u, v, edge_data), ...]，不再依赖 AG 的 key
                    candidates_cache[i] = find_min_weight_path_with_relay(auxiliary_graph, src, dst, top_k=2)
                    del auxiliary_graph

                candidates = candidates_cache[i]

                # 尝试当前索引对应的候选路径
                if tried_path_indices[i] < len(candidates):
                    result = candidates[tried_path_indices[i]]
                    direction, path_with_data, relay, min_power, weight = result
                    
                    if relay:
                        pbar.write(f"[PID {os.getpid()}] 请求 {i}/{len(traffic_matrix)} (路径尝试 {tried_path_indices[i]+1}): {src}->{dst} (中继: {relay}), 功率: {min_power}")
                    else:
                        pbar.write(f"[PID {os.getpid()}] 请求 {i}/{len(traffic_matrix)} (路径尝试 {tried_path_indices[i]+1}): {src}->{dst}, 功率: {min_power}")
                    
                    # 执行分配 (不再需要重新构建辅助图)
                    occupied_wavelength = serve_traffic(topology, path_with_data, traffic, pbar, served_request)
                    
                    # 更新统计数据
                    total_power_each_run += min_power / len(traffic_matrix)
                    spectrum_occupied += occupied_wavelength / network.num_wavelength
                    for (u, v, edge_data) in path_with_data:
                        component_power['source'] += edge_data['source_power'] / len(traffic_matrix)
                        component_power['detector'] += edge_data['detector_power'] / len(traffic_matrix)
                        component_power['other'] += edge_data['other_power'] / len(traffic_matrix)
                        component_power['ice_box'] += edge_data['ice_box_power'] / len(traffic_matrix)

                    remain_num_request -= 1
                    i += 1
                    pbar.update(1)
                    
                    # 如果处理完了所有请求，记录一个可行方案
                    if i == len(traffic_matrix):
                        if total_power_each_run < min_total_power_found:
                            min_total_power_found = total_power_each_run
                            best_solution_found = {
                                'total_power': total_power_each_run,
                                'spectrum': spectrum_occupied,
                                'component': copy.deepcopy(component_power)
                            }
                            pbar.write(f"  [新记录] 找到完整方案，平均功耗: {min_total_power_found:.2f}")
                        
                        # 找到方案后也要回溯，以寻找是否有更优的
                        i -= 1
                        tried_path_indices[i] += 1
                        state_stack.pop()
                        prev = state_stack[-1]
                        topology = prev['topology'].copy()
                        served_request = copy.deepcopy(prev['served_request'])
                        total_power_each_run = prev['total_power_each_run']
                        spectrum_occupied = prev['spectrum_occupied']
                        component_power = copy.deepcopy(prev['component_power'])
                        link_future_demand = prev['link_future_demand'].copy()
                        node_future_demand = prev['node_future_demand'].copy()
                        remain_num_request = prev['remain_num_request']
                        pbar.update(-1)
                else:
                    # 回溯逻辑
                    if i in tried_path_indices: del tried_path_indices[i]
                    if i in candidates_cache: del candidates_cache[i]
                    state_stack.pop()
                    if not state_stack: break
                    i -= 1
                    if i < 0: break
                    tried_path_indices[i] += 1
                    
                    prev = state_stack[-1]
                    topology = prev['topology'].copy()
                    served_request = copy.deepcopy(prev['served_request'])
                    total_power_each_run = prev['total_power_each_run']
                    spectrum_occupied = prev['spectrum_occupied']
                    component_power = copy.deepcopy(prev['component_power'])
                    link_future_demand = prev['link_future_demand'].copy()
                    node_future_demand = prev['node_future_demand'].copy()
                    remain_num_request = prev['remain_num_request']
                    pbar.update(-1)
                
                gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)

            # 最终结算
            if best_solution_found:
                total_power_each_run = best_solution_found['total_power']
                spectrum_occupied = best_solution_found['spectrum']
                component_power = best_solution_found['component']
                pbar.write(f"[PID {os.getpid()}] 搜索结束。最佳平均功耗: {total_power_each_run:.2f}")
            else:
                pbar.write(f"[PID {os.getpid()}] 未能找到任何完整方案。")
                flag = False
                with open(f'result.txt', 'a') as file:
                    file.write(f'\n--- 结果: 搜索失败 (无解) ---\n')
                    file.write(f'Protocol: {config.protocol}, Map: {map_name}\n')
                return

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

        print(f'\n--- 最终结果（经过 {num_runs} 次运行后取最佳结果）---\n'
              f'Protocol: {config.protocol}, Bypass: {config.bypass}, Detector: {config.detector}, Map: {map_name}, Traffic ^:{traffic_type}\n'
              f'{shared_results}')
        with open('result.txt', 'a') as file:
            file.write(f'\n--- 最终结果（每个 mid 经过 {num_runs} 次运行后取平均）---\n')
            file.write(
                f'Protocol: {config.protocol}, Bypass: {config.bypass}, Detector: {config.detector}, Map: {map_name}, Traffic:{traffic_type}\n')
            file.write(f'{shared_results}\n')
            # file.write(f"{traffic_matrix}\n")


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
