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


"""def Dijkstra_single_path(graph, src, dst):
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

    return path_key, weight"""

def Dijkstra_single_path(graph, src, dst, forbidden_nodes=None):
    """
    使用 NetworkX 原生 Dijkstra 算法获取最短路径。
    支持通过 forbidden_nodes 过滤物理路径冲突的边。
    """
    def weight_func(u, v, data):
        # 如果边经过了禁用的物理节点，则视为不可达
        if forbidden_nodes:
            # 兼容 MultiDiGraph 的 data 结构
            path = data.get('path', [])
            for node in path:
                if node in forbidden_nodes:
                    return None
        
        # 兼容 NetworkX 对 MultiDiGraph 可能传递整个 edge dict 的情况
        if 'weight' not in data and len(data) > 0:
            # 尝试从多边字典中找最小权重
            weights = [d.get('weight') for d in data.values() if isinstance(d, dict) and 'weight' in d]
            if weights:
                return float(min(weights))
        
        w = data.get('weight')
        if w is None:
            return 1.0
        return float(w)

    try:
        # NetworkX 的原生算法比纯 Python 循环快得多
        # 对于 MultiDiGraph，它会自动选择权重最小的边
        weight, path_nodes = nx.single_source_dijkstra(
            graph, src, target=dst, weight=weight_func
        )
        
        # 重构路径边（包含 key）
        path_edges = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            edges = graph[u][v]
            
            # 在多边中找到权重最小且符合条件的 key
            best_key = None
            min_w = float('inf')
            for key, data in edges.items():
                w = weight_func(u, v, data)
                if w is not None and w < min_w:
                    min_w = w
                    best_key = key
            
            if best_key is None:
                return [], float('inf')
            path_edges.append((u, v, best_key))
            
        return path_edges, weight
        
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], float('inf')


def Dijkstra_double_path(graph, src, dst, delay):
    """
    计算经过中继的最短路径 (src->delay 和 dst->delay)。
    要求两条路径在物理节点上不冲突（除了 delay 节点）。
    """
    # 1. 寻找 src -> delay 的最短路径
    path1_edges, path1_weight = Dijkstra_single_path(graph, src, delay)
    if not path1_edges:
        return [], [], float('inf')

    # 2. 收集 path1 中占用的所有物理节点
    forbidden_physical_nodes = set()
    for u, v, key in path1_edges:
        p_path = graph[u][v][key].get('path', [])
        forbidden_physical_nodes.update(p_path)
    
    # 中继节点本身是可以复用的，从禁用列表中移除
    forbidden_physical_nodes.discard(delay)

    # 3. 在排除掉冲突物理节点后，寻找 dst -> delay 的最短路径
    # 直接通过 forbidden_nodes 参数传递，无需复制整个图
    path2_edges, path2_weight = Dijkstra_single_path(
        graph, dst, delay, forbidden_nodes=forbidden_physical_nodes
    )

    if path2_edges:
        return path1_edges, path2_edges, path1_weight + path2_weight
    else:
        return [], [], float('inf')