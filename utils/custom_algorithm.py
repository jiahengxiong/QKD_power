from os import write

import networkx as nx
import heapq
import gc

def find_maximin_path(G, src, dst):
    # 初始化每个节点从src出发可达到的最大最小容量（瓶颈容量），初值设为0
    max_cap = {node: 0 for node in G.nodes()}
    max_cap[src] = float('inf')  # 起点的初始容量无穷大

    # 用于记录路径信息，便于重构最终路径
    parent = {}

    # 使用最大堆，堆中元素为 (-容量, 节点)，因为 heapq 是最小堆
    heap = [(-max_cap[src], src)]

    while heap:
        cap, u = heapq.heappop(heap)
        cap = -cap  # 转回正数

        # 如果已到达目标，则可提前退出
        if u == dst:
            break

        # 如果当前容量小于已记录的容量，则跳过
        if cap < max_cap[u]:
            continue

        for v in G.successors(u):
            # 获取边(u, v)的capacity属性
            edge_cap = G[u][v].get('capacity', 0)
            # 新路径的瓶颈容量为当前路径的瓶颈和该边容量中的较小值
            new_cap = min(cap, edge_cap)
            # 如果新路径的瓶颈容量更大，则更新
            if new_cap > max_cap[v]:
                max_cap[v] = new_cap
                parent[v] = u
                heapq.heappush(heap, (-new_cap, v))

    # 如果目标点的容量还是0，则说明没有路径
    if max_cap[dst] == 0:
        return 0, []

    # 重构从src到dst的路径（以边的形式返回）
    path = []
    cur = dst
    while cur != src:
        prev = parent[cur]
        path.append([prev, cur])
        cur = prev
    path.reverse()

    return max_cap[dst], path


import networkx as nx
from networkx.exception import NodeNotFound, NetworkXNoPath


def Dijkstra_single_path(graph, src, dst):
    try:
        # 使用Dijkstra算法获取最短路径的节点序列
        min_cost_path = nx.dijkstra_path(graph, src, dst, weight='weight')
    except (NodeNotFound, NetworkXNoPath):
        # 处理节点不存在或无路径的情况
        return [], None

    path_key = []
    weight = 0
    for i in range(len(min_cost_path) - 1):
        u = min_cost_path[i]
        v = min_cost_path[i + 1]

        # 获取u到v之间所有边的数据和key
        edges_data = graph.get_edge_data(u, v)

        # 处理意外情况（理论上路径存在则边应该存在）
        if not edges_data:
            return [], None

        # 找到weight最小的边
        edges = [(key, data['weight']) for key, data in edges_data.items()]
        min_edge_key = min(edges, key=lambda x: x[1])[0]
        path_key.append((u, v, min_edge_key))
        weight += edges_data[min_edge_key]['weight']

    return path_key, weight

"""def Dijkstra_single_path(graph, src, dst):
    import heapq

    heap = []
    heapq.heappush(heap, (0, src))  # 状态仅保留节点和累计权重

    # 记录到达每个节点的最小权重和前驱信息
    visited = {src: 0}
    predecessors = {}  # Key: 当前节点, Value: (前驱节点, 边标识)

    while heap:
        current_weight, u = heapq.heappop(heap)

        if u == dst:
            # 回溯路径
            path_edges = []
            current_node = u
            while current_node in predecessors:
                prev_node, key = predecessors[current_node]
                path_edges.append((prev_node, current_node, key))
                current_node = prev_node
            path_edges.reverse()
            return path_edges

        # Step 1: 按目标节点分组，找到每个目标节点的最小成本边
        edges_by_v = {}
        for _, v, key, data in graph.out_edges(u, keys=True, data=True):
            if v not in edges_by_v:
                edges_by_v[v] = []
            edges_by_v[v].append((key, data))

        # Step 2: 对每个目标节点，仅处理最小成本边
        for v in edges_by_v:
            min_weight = min(data['weight'] for (_, data) in edges_by_v[v])
            min_edges = [(key, data) for (key, data) in edges_by_v[v] if data['weight'] == min_weight]

            for key, data in min_edges:
                new_weight = current_weight + data['weight']
                # 若新路径更优，则更新
                if v not in visited or new_weight < visited.get(v, float('inf')):
                    visited[v] = new_weight
                    predecessors[v] = (u, key)
                    heapq.heappush(heap, (new_weight, v))

    return []"""


def Dijkstra_double_path(graph, src, delay, dst):
    # 第一步，找到src到delay的最短路径
    path1_edges, path1_weight = Dijkstra_single_path(graph, src, delay)
    if not path1_edges:
        return [], [], None

    # 收集path1中所有物理节点，除了delay
    forbidden_physical_nodes = set()
    for u, v, key in path1_edges:
        path = graph[u][v][key].get('path', [])
        forbidden_physical_nodes.update(node for node in path if node != delay)

    # 创建图的拷贝
    graph_copy = graph.copy()

    # 遍历图的拷贝，移除包含forbidden_physical_nodes的边
    edges_to_remove = []
    for u, v, key, data in graph_copy.edges(data=True, keys=True):
        path = data.get('path', [])
        if any(node in forbidden_physical_nodes for node in path):
            edges_to_remove.append((u, v, key))

    graph_copy.remove_edges_from(edges_to_remove)

    # 第二步，在处理后的图上，找到dst到delay的最短路径
    path2_edges, path2_weight = Dijkstra_single_path(graph_copy, dst, delay)
    del graph_copy
    

    if path2_edges:
        weight = path1_weight + path2_weight
        return path1_edges, path2_edges, weight
    else:
        return [], [], None