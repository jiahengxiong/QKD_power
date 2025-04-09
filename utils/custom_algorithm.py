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


def Dijkstra_single_path(graph, src, dst):
    # 收集所有物理节点
    physical_nodes = set()
    for u, v, key, data in graph.edges(data=True, keys=True):
        path = data.get('path', [])
        for node in path:
            physical_nodes.add(node)

    # Dijkstra算法
    heap = []
    initial_used_nodes = frozenset()
    heapq.heappush(heap, (0, src, initial_used_nodes))
    visited = {}
    visited[(src, initial_used_nodes)] = 0
    predecessors = {}

    while heap:
        current_weight, u, used_nodes = heapq.heappop(heap)
        if u == dst:
            # 回溯路径
            path_edges = []
            current_node = u
            current_used_nodes = used_nodes
            while (current_node, current_used_nodes) in predecessors:
                prev_node, prev_used_nodes, edge = predecessors[(current_node, current_used_nodes)]
                path_edges.append(edge)
                current_node = prev_node
                current_used_nodes = prev_used_nodes
            path_edges.reverse()
            return path_edges
        for _, v, key, data in graph.out_edges(u, keys=True, data=True):
            edge = (u, v, key)
            path = data.get('path', [])
            internal_nodes = set(path[0:-1]) if len(path) >= 2 else set(path[0:1])
            if not (internal_nodes & set(used_nodes)):
                new_used_nodes = frozenset(set(used_nodes) | internal_nodes)
                new_weight = current_weight + data['weight']
                state = (v, new_used_nodes)
                if state not in visited or new_weight < visited[state]:
                    visited[state] = new_weight
                    predecessors[state] = (u, used_nodes, edge)
                    heapq.heappush(heap, (new_weight, v, new_used_nodes))

    return []


def Dijkstra_double_path(graph, src, delay, dst):
    # 第一步，找到src到delay的最短路径
    path1_edges = Dijkstra_single_path(graph, src, delay)
    if not path1_edges:
        return [], []

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
    path2_edges = Dijkstra_single_path(graph_copy, dst, delay)
    del graph_copy
    

    if path2_edges:
        return path1_edges, path2_edges
    else:
        return [], []