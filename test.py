import heapq
import networkx as nx


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
        return None, []

    # 重构从src到dst的路径（以边的形式返回）
    path = []
    cur = dst
    while cur != src:
        prev = parent[cur]
        path.append((prev, cur))
        cur = prev
    path.reverse()

    return max_cap[dst], path


# 使用示例：
if __name__ == "__main__":
    G = nx.DiGraph()
    # 添加边及其容量属性
    G.add_edge('src', 'a', capacity=5)
    G.add_edge('a', 'dst', capacity=3)
    G.add_edge('src', 'b', capacity=4)
    G.add_edge('b', 'dst', capacity=6)

    capacity, edges = find_maximin_path(G, 'src', 'dst')
    print("最大瓶颈容量:", capacity)
    print("路径上的边:", edges)