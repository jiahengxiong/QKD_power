import networkx as nx
import copy
from collections import deque
import math


def can_connect_path(path, laser_detector):
    # 构建有向图的邻接表
    graph = {}
    # print(laser_detector)
    for i, j in laser_detector:
        if i not in graph:
            graph[i] = []
        graph[i].append(j)

    # 获取起点和终点
    start = path[0]
    end = path[-1]

    # 使用BFS检查从起点到终点的可达性
    queue = deque([start])
    visited = set()

    while queue:
        node = queue.popleft()
        if node == end:  # 找到终点
            return True
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:  # 将邻居加入队列
                    queue.append(neighbor)
    return False  # 无法到达终点


def calculate_distance(G, start, end, path):
    if start is None or end is None:
        return 0.00001
    if path.index(start) > path.index(end):
        print("ERROR!!! start_index is bigger than end_index!!!")
    start_index = path.index(start)
    end_index = path.index(end)
    cover_link = path[start_index:end_index + 1]
    distance = 0
    for i in range(len(cover_link) - 1):
        src = cover_link[i]
        dst = cover_link[i + 1]
        distance += G.edges[src, dst]['distance']

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
