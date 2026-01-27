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
    delta = [] # 用于回溯的增量记录: (type, target, key, old_value)
    
    for (src, dst, edge_data) in path_with_data:
        wavelength_list = edge_data['wavelength_list']
        path = edge_data['path']
        edge_traffic = request_traffic
        edge_laser_detector_list = edge_data['transverse_laser_detector']
        
        for wavelength in wavelength_list:
            if wavelength not in served_request:
                served_request[wavelength] = []
                delta.append(('served_request_new_key', wavelength, None, None))
                
            laser_postion = edge_data['laser_detector_position'][wavelength][0]
            detector_postion = edge_data['laser_detector_position'][wavelength][1]
            traffic_limitation = edge_data['wavelength_traffic'][wavelength]
            trans_traffic = min(edge_traffic, traffic_limitation)
            
            if trans_traffic <= 0:
                continue

            if laser_postion is not None and detector_postion is not None:
                laser_index = path.index(laser_postion)
                detector_index = path.index(detector_postion)
                cover_links = path[laser_index:detector_index + 1]
                
                if cover_links not in G.nodes[laser_postion]['laser'][wavelength]:
                    new_list = list(zip(cover_links, cover_links[1:]))
                    served_request[wavelength].append(new_list)
                    delta.append(('served_request_append', wavelength, None, None))
                    
                    G.nodes[laser_postion]['laser'][wavelength].append(cover_links)
                    delta.append(('node_attr_append', laser_postion, ('laser', wavelength), cover_links))
                    
                    G.nodes[detector_postion]['detector'][wavelength].append(cover_links)
                    delta.append(('node_attr_append', detector_postion, ('detector', wavelength), cover_links))
                    
                    # 记录之前的 laser_capacity (可能不存在)
                    old_cap = G.nodes[laser_postion]['laser_capacity'][wavelength].get(tuple(cover_links))
                    G.nodes[laser_postion]['laser_capacity'][wavelength][tuple(cover_links)] = calculate_keyrate(
                        laser_detector_position={'laser': laser_postion, 'detector': detector_postion}, path=path, G=G)
                    delta.append(('node_attr_dict_set', laser_postion, ('laser_capacity', wavelength, tuple(cover_links)), old_cap))
                    
                    G.nodes[detector_postion]['num_detector'] += 1
                    delta.append(('node_attr_inc', detector_postion, 'num_detector', -1))

            # 边容量扣减
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edges = G.get_edge_data(u, v)
                for edge_key, edge_attrs in edges.items():
                    if edge_attrs.get('wavelength') == wavelength:
                        # 记录边状态
                        delta.append(('edge_attr_inc', (u, v, edge_key), 'free_capacity', trans_traffic))
                        G.edges[u, v, edge_key]['free_capacity'] -= trans_traffic
                        
                        if G.edges[u, v, edge_key]['occupied'] is False:
                            G.edges[u, v, edge_key]['occupied'] = True
                            delta.append(('edge_attr_set', (u, v, edge_key), 'occupied', False))
                            occupied_wavelength += 1

            # 横向激光器容量扣减
            for transverse_laser_detector in edge_laser_detector_list[wavelength]:
                lp = transverse_laser_detector[0]
                tk = tuple(transverse_laser_detector)
                delta.append(('node_attr_dict_inc', lp, ('laser_capacity', wavelength, tk), trans_traffic))
                G.nodes[lp]['laser_capacity'][wavelength][tk] -= trans_traffic
                
                # 级联边容量更新
                for i in range(len(transverse_laser_detector) - 1):
                    eu, ev = transverse_laser_detector[i], transverse_laser_detector[i+1]
                    e_data = G.get_edge_data(eu, ev)
                    for ek, ea in e_data.items():
                        if ea.get('wavelength') == wavelength:
                            old_fc = G.edges[eu, ev, ek]['free_capacity']
                            new_fc = min(old_fc, G.nodes[lp]['laser_capacity'][wavelength][tk])
                            if old_fc != new_fc:
                                G.edges[eu, ev, ek]['free_capacity'] = new_fc
                                delta.append(('edge_attr_set', (eu, ev, ek), 'free_capacity', old_fc))
            
            edge_traffic -= trans_traffic

    return occupied_wavelength, delta

def undo_serve_traffic(G, delta, served_request):
    """
    根据 delta 记录反向撤销对 G 和 served_request 的修改
    """
    for action in reversed(delta):
        op, target, key, val = action
        
        if op == 'served_request_new_key':
            del served_request[target]
        elif op == 'served_request_append':
            served_request[target].pop()
        elif op == 'node_attr_append':
            # key is (attr_name, wavelength)
            G.nodes[target][key[0]][key[1]].pop()
        elif op == 'node_attr_dict_set':
            # key is (attr_name, wavelength, dict_key)
            if val is None:
                del G.nodes[target][key[0]][key[1]][key[2]]
            else:
                G.nodes[target][key[0]][key[1]][key[2]] = val
        elif op == 'node_attr_inc':
            G.nodes[target][key] += val
        elif op == 'node_attr_dict_inc':
            # key is (attr_name, wavelength, dict_key)
            G.nodes[target][key[0]][key[1]][key[2]] += val
        elif op == 'edge_attr_inc':
            u, v, k = target
            G.edges[u, v, k][key] += val
        elif op == 'edge_attr_set':
            u, v, k = target
            G.edges[u, v, k][key] = val


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

            # --- 3. 顺序处理请求 (Robust Beam Search) ---
            # 提升 Beam 宽度 B，以应对 Large 图的复杂性
            BEAM_WIDTH = 5 
            
            # 初始化 Beam
            current_beam = [{
                'topology': topology.copy(),
                'served_request': copy.deepcopy(served_request),
                'total_power': 0.0,
                'component_power': component_power.copy(),
                'spectrum_occupied': 0.0,
                'score': 0.0
            }]
            
            pbar.write(f"[PID {os.getpid()}] 开始 Robust Beam Search 搜索 (宽度: {BEAM_WIDTH})...")

            for i, request in enumerate(traffic_matrix):
                id, src, dst, traffic = request[0], request[1], request[2], request[3]
                
                # 更新热力图
                node_future_demand[dst] = max(0, node_future_demand.get(dst, 0) - traffic)
                if id in request_paths_cache:
                    for edges, weight in request_paths_cache[id]:
                        for u, v in edges:
                            impact = weight * traffic
                            link_future_demand[(u, v)] = max(0, link_future_demand.get((u, v), 0) - impact)
                            link_future_demand[(v, u)] = max(0, link_future_demand.get((v, u), 0) - impact)

                next_candidates = []
                
                for state_idx, state in enumerate(current_beam):
                    auxiliary_graph = utils.tools.build_auxiliary_graph(
                        topology=state['topology'],
                        wavelength_list=wavelength_list,
                        traffic=traffic,
                        physical_topology=physical_topology,
                        shared_key_rate_list=key_rate_list,
                        served_request=state['served_request'],
                        remain_num_request=remain_num_request,
                        link_future_demand=link_future_demand,
                        node_future_demand=node_future_demand
                    )
                    
                    # 提升分支因子 top_k 至 4，增加备选路径多样性
                    paths = find_min_weight_path_with_relay(auxiliary_graph, src, dst, top_k=4)
                    
                    for path_info in paths:
                        direction, path_with_data, relay, min_power, weight = path_info
                        
                        new_topo = state['topology'].copy()
                        new_served_req = copy.deepcopy(state['served_request'])
                        
                        occ_wl, _ = serve_traffic(new_topo, path_with_data, traffic, pbar, new_served_req)
                        
                        new_total_power = state['total_power'] + min_power / len(traffic_matrix)
                        new_spectrum = state['spectrum_occupied'] + occ_wl / network.num_wavelength
                        
                        new_comp_power = state['component_power'].copy()
                        for (u, v, edge_data) in path_with_data:
                            new_comp_power['source'] += edge_data['source_power'] / len(traffic_matrix)
                            new_comp_power['detector'] += edge_data['detector_power'] / len(traffic_matrix)
                            new_comp_power['other'] += edge_data['other_power'] / len(traffic_matrix)
                            new_comp_power['ice_box'] += edge_data['ice_box_power'] / len(traffic_matrix)
                        
                        # 改进评分逻辑：增加对未来冲突权重 (weight) 的敏感度，系数从 0.01 提升至 0.05
                        # 这样算法会更主动地避开未来可能需要的热门资源
                        score = new_total_power + (weight * 0.05) 
                        
                        next_candidates.append({
                            'topology': new_topo,
                            'served_request': new_served_req,
                            'total_power': new_total_power,
                            'component_power': new_comp_power,
                            'spectrum_occupied': new_spectrum,
                            'score': score
                        })
                    
                    del auxiliary_graph
                
                if not next_candidates:
                    # 如果常规搜索失败，尝试紧急恢复（使用更大的 top_k 和较轻的惩罚再次尝试）
                    pbar.write(f"[PID {os.getpid()}] 请求 {i} ({src}->{dst}) 常规分配失败，进入紧急恢复模式...")
                    # 这里暂不实现复杂的重试逻辑，先通过提升参数解决
                    flag = False
                    break
                
                next_candidates.sort(key=lambda x: x['score'])
                
                # 内存回收
                for discarded in current_beam:
                    discarded['topology'] = None
                    discarded['served_request'] = None
                for discarded in next_candidates[BEAM_WIDTH:]:
                    discarded['topology'] = None
                    discarded['served_request'] = None
                
                current_beam = next_candidates[:BEAM_WIDTH]
                remain_num_request -= 1
                pbar.update(1)
                
                if i % 5 == 0:
                    gc.collect()
                    ctypes.CDLL("libc.so.6").malloc_trim(0)

            if flag:
                # 从最终 Beam 中选出功耗最低的一个
                current_beam.sort(key=lambda x: x['total_power'])
                best_state = current_beam[0]
                total_power_each_run = best_state['total_power']
                spectrum_occupied = best_state['spectrum_occupied']
                component_power = best_state['component_power']
                pbar.write(f"[PID {os.getpid()}] Beam Search 结束。最佳方案平均功耗: {total_power_each_run:.4f}")
            else:
                # 仿真失败的处理
                with open(f'result.txt', 'a') as file:
                    file.write(f'\n--- 结果: 仿真失败 (Beam Search 无法满足所有请求) ---\n')
                    file.write(f'Protocol: {config.protocol}, Map: {map_name}\n')
                return

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
