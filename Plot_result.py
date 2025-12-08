import matplotlib.pyplot as plt
import numpy as np

# Tokyo 三个 traffic level 的数据
Tokyo_Low = np.array([[0.4884057971014486, 3.66304347826087, 112.44322463768115, 0.0],
                       [0.5188405797101442, 3.8913043478260874, 119.45007246376811, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]])
Tokyo_Medium = np.zeros((6, 4))
Tokyo_High = np.zeros((4, 4))

groups = [Tokyo_Low, Tokyo_Medium, Tokyo_High]
Traffic_list = ['L', 'M', 'H']
cases = ['CV-B', 'CV-NB', 'BB-A-B', 'BB-A-NB', 'BB-S-B', 'BB-S-NB']
component_list = ['Source', 'Detector', 'Other', 'Ice box']
colors = ['#E91E63', '#4CAF50', '#2196F3', '#FFC107']

# 创建图形
fig, ax = plt.subplots(figsize=(10, 4))

x_positions = []
tick_positions = []
tick_labels = []
current_x = 0  # 初始 x 位置

# 计算柱子位置
for i, group in enumerate(groups):
    num_bars = group.shape[0]
    xs = np.arange(num_bars) + current_x
    x_positions.extend(xs)
    tick_positions.extend(xs)
    traffic = Traffic_list[i]
    tick_labels.extend([f"{traffic}-{cases[j % len(cases)]}" for j in range(num_bars)])
    current_x = xs[-1] + 1.5  # 控制不同组之间的空隙

x_positions = np.array(x_positions)
all_data = np.vstack(groups)

# 堆叠绘图
bottom = np.zeros(len(all_data))
bar_width = 0.8

for i in range(4):
    ax.bar(x_positions, all_data[:, i], bottom=bottom, width=bar_width, color=colors[i], label=component_list[i])
    bottom += all_data[:, i]

# 设置图例、标签等
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha='right')
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 140)
ax.tick_params(axis='y', labelsize=14)  # 设置 Y 轴字体大小
ax.set_xlim(-0.5, current_x)  # 👈 这里让第一个柱子贴紧 y 轴
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.show()