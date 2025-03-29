from pyvis.network import Network
import networkx as nx

edge_list = [
                (1, 2), (1, 6),
                (2, 3), (2, 6),
                (3, 4), (3, 5), (3, 7),
                (4, 5), (4, 7),
                (5, 8),
                (6, 7), (6, 9), (6, 11),
                (7, 8), (7, 9),
                (8, 10),
                (9, 10), (9, 12), (9, 11),
                (10, 13), (10, 14),
                (11, 15), (11, 19), (11, 12),
                (12, 13), (12, 16),
                (13, 14), (13, 17),
                (14, 18),
                (15, 16), (15, 20),
                (16, 17), (16, 21), (16, 22),
                (17, 18), (17, 22), (17, 23),
                (18, 24),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24)
            ]
length_list = [800, 1000, 1100, 950, 250, 1000, 1000, 800, 850, 1200, 1000, 1200, 1900, 1150, 1000, 900,
               1000, 1000,
               1400, 950, 850, 1300, 2800, 900, 800, 1000, 650, 800, 1200, 800, 1300, 800, 1000, 800, 800,
               850, 1000,
               900, 1200, 700, 300, 600, 900]

# 缩放到 [30, 130]
min_val = 30
max_val = 130

# 数据缩放公式
scaled_list = [
    min_val + (x - min(length_list)) * (max_val - min_val) / (max(length_list) - min(length_list)) for x in
    length_list]

# 转换为整数
length_list = [int(round(x)) for x in scaled_list]

# 创建一个图
G = nx.Graph()
for i in range(len(edge_list)):
    edge = edge_list[i]
    distance = length_list[i]
    G.add_edge(edge[0], edge[1], distance=distance)

# 创建PyVis网络对象
net = Network()

# 添加节点
for node in G.nodes():
    net.add_node(node, label=str(node), font={'size': 25, 'color': 'black'}, shape='circle')

# 添加带有label的边，并根据distance调整边的宽度
for edge in G.edges():
    distance = G[edge[0]][edge[1]]['distance']
    # 将边的宽度与distance相关联
    width = max(1, distance / 200)  # 调整比例因子
    net.add_edge(edge[0], edge[1], label=f'{distance}km', color='blue',font={'size': 20, 'align': 'top'}, length=distance * 3.34)

# 启用物理布局，使得节点自动布局
net.set_options("""
{
  "edges": {
    "smooth": false,
    "font": {
      "size": 20,
      "align": "top"  
    },
    "labelHighlightBold": true
  },
  "physics": {
    "enabled": true,
    "barnesHut": {
      "gravitationalConstant": -5000,
      "centralGravity": 0.1,
      "springLength": 95,
      "springConstant": 0.05,
      "damping": 0.09,
      "avoidOverlap": 1.0
    },
    "minVelocity": 0.75
  }
}
""")


# 保存并显示图形
net.show("graph.html", notebook=False)
