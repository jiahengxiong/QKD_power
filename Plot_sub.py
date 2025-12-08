import matplotlib.pyplot as plt
import numpy as np

# 每个柱子有 4 个部分（堆叠）
data = np.array([
    [0.3772048846675711, 2.8290366350067835, 86.8419945725916, 0.0],
    [0.39294436906377195, 2.9470827679782894, 90.46561736770694, 0.0]  # 第二个柱子
])

# 每部分的颜色
colors = ['#4CAF50', '#2196F3', '#FFC107', '#E91E63']
labels = ['Source', 'Detector', 'Other', 'Ice box']

# x 坐标（两个柱子）
x = np.arange(data.shape[0])


# 初始化底部
bottom = np.zeros(data.shape[0])

# 画堆叠柱子
for i in range(data.shape[1]):
    plt.bar(x, data[:, i], bottom=bottom, color=colors[i], label=labels[i], width = 0.1)
    bottom += data[:, i]

# 坐标设置
plt.xticks(x, ['Bar 1', 'Bar 2'], fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.title("Single Group with 2 Stacked Bars", fontsize=14)
# plt.legend()
plt.tight_layout()
plt.show()