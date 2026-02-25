import networkx as nx
import copy
from collections import deque
import math


def can_connect_path(path, laser_detector):
    """
    检查给定的 laser_detector 对是否能覆盖整个 path。
    优化版：基于 path 索引的快速连通性检查，避免构建 NetworkX 图。
    
    Args:
        path: 节点列表 [n0, n1, ..., nL]
        laser_detector: 边列表 [[u1, v1], [u2, v2], ...]，u,v 都在 path 上
    """
    if not path: return False
    
    # 1. 建立节点到索引的映射
    # O(V)
    node_to_idx = {node: i for i, node in enumerate(path)}
    target_idx = len(path) - 1
    
    # 2. 将边转换为索引邻接表
    # O(E)
    adj = [[] for _ in range(len(path))]
    for u, v in laser_detector:
        # 安全检查：确保 u, v 都在 path 上
        # 虽然调用方应该保证，但这里再次确认
        if u in node_to_idx and v in node_to_idx:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            # 只考虑前向边 (u -> v)，避免环路干扰
            if u_idx < v_idx:
                adj[u_idx].append(v_idx)
                
    # 3. 快速 BFS (在索引空间)
    # O(V + E)
    queue = deque([0])
    visited = {0} # 使用 set 记录访问过的索引
    
    while queue:
        curr = queue.popleft()
        if curr == target_idx:
            return True
            
        for nxt in adj[curr]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
                
    return False


def calculate_distance(G, start, end, path):
    if start is None or end is None:
        return 0.00001  # 默认极小值

    try:
        start_index = path.index(start)
        end_index = path.index(end)
    except ValueError:
        print(f"ERROR: Start or end node not in path")
        return 0.00001

    if start_index > end_index:
        print("ERROR: Start index is after end index in path")
        return 0.00001

    cover_link = path[start_index:end_index + 1]
    distance = 0

    for i in range(len(cover_link) - 1):
        src = cover_link[i]
        dst = cover_link[i + 1]

        # 获取所有边数据（兼容普通图和多图）
        edges_data = G.get_edge_data(src, dst)
        if not edges_data:
            print(f"No edges between {src} and {dst}")
            return 0.00001  # 或根据需求抛出异常

        # 多图处理逻辑 (兼容无向和有向多重图)
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            # 策略示例：选择第一条边（可替换为最短边、随机边等）
            first_edge_key = next(iter(edges_data))
            edge_distance = edges_data[first_edge_key].get('distance', 0)
        else:
            # 普通图直接获取唯一边的属性
            edge_distance = edges_data.get('distance', 0)

        distance += edge_distance

    return distance


def calculate_cutoff(G, traffic):
    max_traffic = 0  # 用于记录最大流量
    AG = copy.deepcopy(G)  # 创建图的副本

    # 遍历所有节点对
    for src, dst in G.edges():
        total_capacity = 0  # 记录src和dst之间的所有边的free_capacity之和

        # 遍历src和dst之间的所有边
        for key, edge_data in AG.get_edge_data(src, dst).items():
            # 获取每条边的free_capacity并加到总容量中
            total_capacity += edge_data['free_capacity']

        # 更新最大流量
        max_traffic = max(max_traffic, total_capacity)
        n = math.floor(math.log(traffic / max_traffic) / math.log(0.89))
        cutoff = n + 2
        print(cutoff, max_traffic, traffic)
        return cutoff
