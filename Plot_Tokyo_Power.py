import matplotlib.pyplot as plt
import numpy as np

Tokyo_Low = np.array([[0.3772048846675711, 2.8290366350067835, 86.8419945725916, 0.0],
                      [0.39294436906377195, 2.9470827679782894, 90.46561736770694, 0.0],
                      [3.6848484848484824, 593.2606060606057, 840.1454545454556, 0.0],
                      [3.890909090909088, 626.4363636363634, 887.1272727272741, 0.0],
                      [1.3939393939393931, 0, 317.81818181818176, 227.27272727272725],
                      [1.3575757575757568, 0, 309.5272727272726, 281.81818181818176]])
Tokyo_Medium = np.array([[0.4047942107643599, 3.0359565807326994, 93.1937471732248, 0.0],
                         [0.41085481682496594, 3.0814111261872448, 94.58905020352783, 0.0],
                         [5.406060606060602, 870.375757575757, 1232.581818181819, 0.0],
                         [6.399999999999995, 1030.3999999999992, 1459.1999999999998, 0.0],
                         [2.060606060606059, 0, 469.8181818181817, 272.7272727272727],
                         [2.1212121212121198, 0, 483.6363636363635, 409.090909090909]])
Tokyo_High = np.array([[0.5798281320669376, 4.348710990502035, 133.49093170511082, 0.0],
                       [0.5919493441881498, 4.4396200814111255, 136.2815377657169, 0.0],
                       [6.181818181818177, 0, 1409.4545454545457, 609.090909090909],
                       [7.054545454545449, 0, 1608.4363636363628, 772.7272727272729]])


Large_Low = np.array([[0.22753623188405805, 1.706521739130435, 52.38452898550727, 0.0],
                      [0.24057971014492763, 1.804347826086957, 55.387463768115985, 0.0],
                      [1.968115942028988, 316.8666666666666, 448.7304347826081, 0.0],
                      [2.00289855072464, 322.4666666666666, 456.6608695652167, 0.0],
                      [0.7478260869565211, 0, 170.50434782608704, 178.26086956521743],
                      [0.7623188405797094, 0, 173.80869565217398, 186.9565217391305]])
Large_Medium = np.array([[0.263768115942029, 1.978260869565218, 60.72601449275367, 0.0],
                         [0.26811594202898553, 2.0108695652173916, 61.72699275362325, 0.0],
                         [3.56231884057972, 573.5333333333326, 812.2086956521765, 0.0],
                         [3.6144927536231983, 581.9333333333327, 824.1043478260897, 0.0],
                         [1.1710144927536223, 0, 251.13043478260903, 213.04347826086965],
                         [1.1913043478260863, 0, 267.6521739130438, 228.2608695652175]])
Large_High = np.array([[0.4884057971014486, 3.66304347826087, 112.44322463768115, 0.0],
                       [0.5188405797101442, 3.8913043478260874, 119.45007246376811, 0.0],
                       [3.144927536231892, 0, 717.0434782608711, 402.1739130434785],
                       [3.2608695652173996, 0, 743.4782608695671, 413.0434782608698]])


# 模拟数据生成函数
def generate_data():
    group1 = np.random.randint(1, 10, size=(6, 4))
    group2 = np.random.randint(1, 10, size=(6, 4))
    group3 = np.random.randint(1, 10, size=(4, 4))
    return [Tokyo_Low, Tokyo_Medium, Tokyo_High]


# 数据准备：两个子图的数据
data_list = [[Tokyo_Low, Tokyo_Medium, Tokyo_High], [Large_Low, Large_Medium, Large_High]]
print(data_list[0])
group_labels = ['L', 'M', 'H']
colors = ['#E91E63', '#4CAF50', '#2196F3', '#FFC107']  # 堆叠颜色

# 创建子图（并排排列）
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

for ax_idx, ax in enumerate(axes):
    groups = data_list[ax_idx]
    x_positions = []
    current_x = 0
    tick_positions = []
    tick_labels = []
    Traffic_list = ['L', 'M', 'H']
    cases = ['CV-B',
             'CV-NB',
             'BB-A-B',
             'BB-A-NB',
             'BB-S-B',
             'BB-S-NB',
             'CV-B',
             'CV-NB',
             'BB-A-B',
             'BB-A-NB',
             'BB-S-B',
             'BB-S-NB',
             'CV-B',
             'CV-NB',
             'BB-S-B',
             'BB-S-NB'
             ]

    # 计算每组柱子的位置
    for i, group in enumerate(groups):
        num_bars = group.shape[0]
        xs = np.arange(num_bars) + current_x
        x_positions.extend(xs)
        tick_positions.extend(xs)
        traffic = Traffic_list[i]
        tick_labels.extend([f"{traffic}-{cases[j]}" for j in range(num_bars)])
        current_x = xs[-1] + 2  # 每组之间留空隙

    x_positions = np.array(x_positions)
    all_data = np.vstack(groups)

    # 初始化堆叠底部
    bottom = np.zeros(len(all_data))

    component_list = ['Source', 'Detector', 'Other', 'Cooling System']

    # 堆叠每一层
    Subplot_list = ['Tokyo', 'NSF']
    for i in range(4):
        component = component_list[i]
        ax.bar(x_positions, all_data[:, i], bottom=bottom, width=0.8, color=colors[i],
               label=f'{component}' if ax_idx == 0 else "")
        bottom += all_data[:, i]

    # 设置子图标题、字体大小
    # ax.set_title(f"{Subplot_list[ax_idx]}", fontsize=20)

    # 设置 Tokyo 子图的 y 轴最大值为 7000
    if ax_idx == 0:
        ax.set_ylim(0, 3200)
    else:
        ax.set_ylim(0, 1500)

    # 设置纵坐标和横坐标标签字体大小
    ax.set_xlabel(f'Traffic and Cases of {Subplot_list[ax_idx]}', fontsize=20)
    ax.set_ylabel('Average Power', fontsize=20)

    # 设置x轴标签的字体大小
    tick_labels = cases
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=18)

    # 设置y轴的字体大小
    ax.tick_params(axis='y', labelsize=20)
    # 设置x轴的字体大小
    ax.tick_params(axis='x', labelsize=16)
    if ax_idx == 1:
        ax.tick_params(axis='y', labelleft=True)  # 显示第二个子图的 y 轴数字
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# 添加图例到第一个子图
axes[0].legend(loc='upper center', ncol=4, fontsize=12)

# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=11)


fig.tight_layout()
plt.show()

# data = {'traffic': 1050000, 'total_avg_power': 81.64712347354137, 'avg_spectrum_occupied': 0.06095238095238094, 'avg_component_power': {'source': 0.41085481682496594, 'detector': 3.0814111261872448, 'other': 78.15485753052918, 'ice_box': 0.0}}
#
#
# total = 0
# for key, valeu in data['avg_component_power'].items():
#     total += valeu
# data['total_avg_power'] = total
# print(data)
