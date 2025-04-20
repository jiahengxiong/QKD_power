import copy
import gc
import itertools
import uuid
from itertools import product
from multiprocessing import Pool


import config
from QEnergy.studies.key_rate_compute import compute_key_rate
from QEnergy.studies.power_compute import compute_power
from utils.Network import Network
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
        combined_capacity[(u, v)] = 0 # 累积每条边的容量
        combined_capacity[(v, u)] = 0
    selected_wavelengths = []  # 记录选中的λᵢ组合

    for wavelength in wavelength_list:
        network_wavelength = network_slice[wavelength]
        free_capacity_list = [0]
        for edge in network_wavelength.edges:
            free_capacity_list.append(network_wavelength.edges[edge]['free_capacity'])

        if max(free_capacity_list) <= 0:
            continue
        else:
            # 将该波长的容量加到总图上
            for edge in network_wavelength.edges:
                if network_wavelength.has_edge(*edge):
                    cap = network_wavelength.edges[edge].get('free_capacity', 0)
                    # print(combined_capacity)
                    combined_capacity[edge] += cap
                    combined_capacity[(edge[1], edge[0])] += cap

            selected_wavelengths.append(wavelength)

            # 判断是否当前组合可以满足traffic（取整图min）
            min_cap = min(combined_capacity.values())
            if min_cap >= traffic:
                break



    all_combinations = []
    for r in range(1, len(selected_wavelengths) + 1):
        combinations = itertools.combinations(selected_wavelengths, r)
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
                    LD = LD+(path.index(laser_detector[1]) - path.index(laser_detector[0]))
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
                                 * (10 + LD/(used_LD))
                                 * (10 + 1e-2*(len(wavelength_combination))**1.0 * ((len(path) - 1))**1.0)
                                 #* bypass

                                 # + spectrum
                                 + 1e-2*(len(wavelength_combination))**1.0 * ((len(path) - 1))**1.0
                                 # - 10**(-3)
                                 # + 1e-2 * distance
                                 # + 10 * len(wavelength_combination)
                                 # + 1e-8 * sum(wavelength_combination)
                                 # + 1e-4 * min(wavelength_combination) * (len(path) - 1)**1.01
                                   + 1e-6 * max(wavelength_combination)
                                 # + 1e-16 * max_traffic
                         ,
                         'wavelength_list': wavelength_combination,
                         'transverse_laser_detector':wavelength_used_laser_detector})

    return data

def get_k_shortest_paths(graph, src, dst, k=5, weight='distance'):
    virtual_physical_topology_simple = nx.Graph(graph)
    try:
        gen = nx.shortest_simple_paths(virtual_physical_topology_simple, source=src, target=dst)
        return list(itertools.islice(gen, k))  # 最多 k 条，少于也可以
    except nx.NetworkXNoPath:
        return []  # 找不到任何路径就返回空列表

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
                        # k = 5
                        # all_paths = get_k_shortest_paths(virtual_physical_topology, src, dst, k)
                        all_paths = [
                            nx.dijkstra_path(virtual_physical_topology, source=src, target=dst, weight='distance')]
                    else:
                        all_paths = []
                    # no_bypass_path = []
                    # for path in all_paths: # "Setagaya", "Ota"
                    #     path_edge = []
                    #     for i in range(len(path) - 1):
                    #         path_edge.append((path[i], path[i + 1]))
                    #         path_edge.append((path[i + 1], path[i]))
                    #     if (('TP', 'OG') in path_edge) or (('LIP6', 'OG') in path_edge) or ((19, 11) in path_edge) or (('Setagaya', 'Ota') in path_edge):
                    #         # all_paths = [[path[0], path[1]]]
                    #         for i in range(len(all_paths[0]) - 1):
                    #             no_bypass_path.append([path[i], path[i + 1]])
                    #         all_paths = no_bypass_path


                    # if min(wavelength_combinations) > 1530:
                    #     no_bypass_path = []
                    #     for path in all_paths:  # "Setagaya", "Ota"
                    #         path_edge = []
                    #         for i in range(len(path) - 1):
                    #             no_bypass_path.append([path[i], path[i + 1]])
                    #         all_paths = no_bypass_path
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
                        auxiliary_graph.add_edge(path[0], path[-1], key=uuid.uuid4().hex + str(wavelength_combinations),
                                                 **edge_data)
                        # print('Build virtual edge in ', path[0], path[-1])
    del  virtual_physical_topology
    
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
    wavelength_combination_list = combine_wavelength_slices(wavelength_list=wavelength_list, topology=physical_topology, traffic=traffic, network_slice=network_slice)
    protocol = config.protocol
    detector = config.detector
    ice_box_capacity = config.ice_box_capacity
    bypass = config.bypass
    results = []
    for node in topology.nodes():
        auxiliary_graph.add_node(node)
    # 遍历 wavelength_combination_list 中的每一项
    for wavelength_combinations in wavelength_combination_list:
        process_wavelength_combination(
            wavelength_combinations,
            topology,
            traffic,
            network_slice,
            physical_topology,
            protocol,
            detector,
            ice_box_capacity,
            bypass,
            shared_key_rate_list,
            auxiliary_graph
        )

        # for wavelength_combinations in wavelength_combination_list:
        #     bonus_G = nx.DiGraph()
        #
        #     # 获取包含所有边信息的生成器
        #     for src, dst, key, data in auxiliary_graph.edges(keys=True, data=True):
        #         if data['wavelength_list'] == wavelength_combinations:
        #             bonus_G.add_edge(src, dst, weight=data['weight'])
        #     centrality = nx.betweenness_centrality(bonus_G, weight="distance")
        #     for src, dst, key, data in auxiliary_graph.edges(keys=True, data=True):
        #         if data['wavelength_list'] == wavelength_combinations:
        #             path = data['path']
        #             bonus = 0
        #             for node in path:
        #                 bonus = centrality[node] + bonus
        #             # if bonus == 0:
        #             #     print("Bonus is 0", path)
        #             auxiliary_graph.edges[src, dst, key]['weight'] = auxiliary_graph.edges[src, dst, key]['weight'] / (1 + bonus/((len(path))*len(wavelength_combinations)))
        #     del bonus_G

    # 合并所有局部图，生成总图
    # auxiliary_graph = nx.compose_all(results)

    # with Pool(processes=8) as pool:
    #     # 使用 tqdm 包装 starmap 的进度条
    #     results = pool.starmap(process_wavelength_combination,
    #                            [(wavelength_combinations, topology, traffic, network_slice, physical_topology, protocol, detector, ice_box_capacity, bypass, shared_key_rate_list)
    #                             for wavelength_combinations in wavelength_combination_list])
    #
    # # 合并所有局部图
    # auxiliary_graph = nx.compose_all(results)
     # 清理临时图对象列表和切片以释放内存**
    for subgraph in results:
        subgraph.clear()  # 清空子图内容
    del  results, network_slice
    
    return auxiliary_graph



def process_wavelength_combination(wavelength_combinations, topology, traffic, network_slice, physical_topology,
                                   protocol, detector, ice_box_capacity, bypass, shared_key_rate_list, AG):
    # 在每个进程中创建自己的局部图
    multi_wavelength_slice = build_multiple_wavelength_slices(wavelength_combinations=wavelength_combinations,
                                                              topology=topology, traffic=traffic)
    # local_auxiliary_graph = nx.MultiDiGraph()  # 创建局部图
    config.protocol = protocol
    config.detector = detector
    config.ice_box_capacity = ice_box_capacity
    config.bypass = bypass
    config.key_rate_list = shared_key_rate_list
    # for node in topology.nodes():
    #     local_auxiliary_graph.add_node(node)

    build_multi_wavelength_auxiliary_graph(multi_wavelength_slice=multi_wavelength_slice,
                                           network_slice=network_slice, traffic=traffic,
                                           physical_topology=physical_topology,
                                           wavelength_combinations=wavelength_combinations,
                                           auxiliary_graph=AG)
    # print(f"Finished processing wavelength combination: {wavelength_combinations}")

    # return local_auxiliary_graph  # 返回局部图





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


# def generate_and_sort_requests(topology):
#
#     """生成节点对请求，并按增强度排序，增强度考虑所有节点影响，指数衰减"""
#
#     node_list = list(topology.nodes)
#     raw_degrees = dict(topology.degree())
#
#     # 计算增强度：对每个节点，计算所有可达节点的度数影响（指数衰减）
#     enhanced_degrees = {}
#
#     for node in node_list:
#         enhanced_degree = raw_degrees[node]  # 自身度数
#         lengths = nx.single_source_shortest_path_length(topology, node)
#
#         for other_node, distance in lengths.items():
#             if other_node == node:
#                 continue
#             decay_factor = 0.01 ** distance  # 第一圈0.01，第二圈0.0001，依此类推
#             enhanced_degree += raw_degrees[other_node] * decay_factor
#
#         enhanced_degrees[node] = enhanced_degree
#
#     # Sort enhanced_degrees by value in descending order
#     sorted_enhanced_degrees = sorted(enhanced_degrees.items(), key=lambda x: x[1], reverse=True)
#     print(sorted_enhanced_degrees)
#
#     # Identify the network center (node with the highest enhanced degree)
#     network_center = sorted_enhanced_degrees[0][0]
#     print(f"Network Center (Highest Enhanced Degree): {network_center}")
#
#     # 生成所有无向节点对，并确定方向（根据距离网络中心的距离决定receiver）
#     pairs = []
#     for i in range(len(node_list)):
#         for j in range(i + 1, len(node_list)):
#             node_a, node_b = node_list[i], node_list[j]
#             if node_a == network_center:
#                 pairs.append((node_b, node_a))
#                 continue
#             if node_b == network_center:
#                 pairs.append((node_a, node_b))
#                 continue
#
#             # Calculate the distance from each node to the network center
#             distance_a_to_center = nx.dijkstra_path_length(topology, node_a, network_center, weight='distance')
#             distance_b_to_center = nx.dijkstra_path_length(topology, node_b, network_center, weight='distance')
#             print(distance_a_to_center, distance_b_to_center)
#
#             if distance_a_to_center < distance_b_to_center:
#                 sender, receiver = node_a, node_b
#             elif distance_a_to_center > distance_b_to_center:
#                 sender, receiver = node_b, node_a
#             else:  # If both nodes are equally distant from the network center
#                 if enhanced_degrees[node_a] > enhanced_degrees[node_b]:
#                     sender, receiver = node_b, node_a
#                 else:
#                     sender, receiver = node_a, node_b
#
#             pairs.append((sender, receiver))
#
#     # 计算跳数和物理距离
#     shortest_paths = {}
#     physical_distances = {}
#     for sender, receiver in pairs:
#         try:
#             path_length = nx.shortest_path_length(topology, sender, receiver)
#         except nx.NetworkXNoPath:
#             path_length = float('inf')
#         shortest_paths[(sender, receiver)] = path_length
#
#         try:
#             distance = nx.shortest_path_length(topology, sender, receiver, weight='distance')
#         except nx.NetworkXNoPath:
#             distance = float('inf')
#         physical_distances[(sender, receiver)] = distance
#
#     # 排序：接收方增强度高 > 发送方增强度高 > 跳数短 > 物理距离短
#     pairs.sort(
#         key=lambda x: (
#             -enhanced_degrees[x[1]],
#             -enhanced_degrees[x[0]],
#             shortest_paths[x],
#             physical_distances[x]
#         )
#     )
#
#     return pairs

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