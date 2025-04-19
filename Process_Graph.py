import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt

# 读取 GraphML 文件
tree = ET.parse("topology/Abilene.graphml")  # 替换成你的文件路径
root = tree.getroot()

# 命名空间定义
ns = {
    "g": "http://graphml.graphdrawing.org/xmlns",
    "y": "http://www.yworks.com/xml/graphml"
}

# 构建图
G = nx.Graph()

# 提取节点坐标
positions = {}
for node in root.findall(".//g:node", namespaces=ns):
    node_id = node.attrib['id']
    geometry = node.find(".//y:Geometry", namespaces=ns)
    if geometry is not None:
        x = float(geometry.attrib["x"])
        y = float(geometry.attrib["y"])
        positions[node_id] = (x, y)
        G.add_node(node_id, pos=(x, y))

# 提取边并计算距离
for edge in root.findall(".//g:edge", namespaces=ns):
    source = edge.attrib["source"]
    target = edge.attrib["target"]
    x1, y1 = positions[source]
    x2, y2 = positions[target]
    distance = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    G.add_edge(source, target, distance=distance)

# 构建 edge_list 和 length_list
edge_list = list(G.edges())
length_list = [G[u][v]['distance'] for u, v in edge_list]

# 打印结果
print("edge_list =", edge_list)
print("length_list =", length_list)
print(min(length_list), max(length_list))

# 可视化
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=500)
plt.title("Graph Visualization")
plt.show()