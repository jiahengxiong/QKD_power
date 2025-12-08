import matplotlib.pyplot as plt
import numpy as np

# Tokyo_Low = np.array([[0.3772048846675711, 2.8290366350067835, 71.75379918588875, 0.0],
#                       [0.39294436906377195, 2.9470827679782894, 74.74784260515605, 0.0],
#                       [4.181818181818182, 673.272727272727, 953.454545454546, 0.0],
#                       [4.927905924920851, 793.3928539122563, 1123.5625508819544, 0.0],
#                       [1.7953867028493882, 1346.5400271370424, 409.34816824966083, 262.14382632293075],
#                       [1.759384893713251, 1319.5386702849391, 401.1397557666215, 352.5101763907734]])
# Tokyo_Medium = np.array([[0.4047942107643599, 3.0359565807326994, 77.0019787426504, 0.0],
#                          [0.41085481682496594, 3.0814111261872448, 78.15485753052918, 0.0],
#                          [7.18552691090005, 1156.869832654907, 1638.3001356852105, 0.0],
#                          [8.145997286295799, 1311.5055630936229, 1857.2873812754417, 0.0],
#                          [2.750972410673903, 2063.2293080054287, 627.2217096336501, 351.56037991858886],
#                          [2.7992763455450023, 2099.4572591587525, 638.2350067842606, 486.2957937584803]])
# Tokyo_High = np.array([[0.5798281320669376, 4.348710990502035, 110.29780642243331, 0.0],
#                        [0.5919493441881498, 4.4396200814111255, 112.60356399819088, 0.0],
#                        [8.05011307100859, 6037.584803256443, 1835.4257801899607, 743.5549525101763],
#                        [8.763274536408863, 6572.455902306635, 1998.02659430122, 888.1953867028494]])

Tokyo_low = np.array([0.059047619047619036,0.059047619047619036,0.4866666666666667,0.38761904761904775,0.19142857142857145,0.13619047619047617])
Tokyo_medium = np.array([0.06095238095238094, 0.06095238095238094, 0.7714285714285715, 0.639047619047619, 0.3133333333333334, 0.2200000000000001])
Tokyo_high = np.array([0.09052380952380949, 0.08947619047619045, 0.8542857142857143, 0.6780952380952382])

Large_low = np.array([0.06586046511627907,0.0615813953488372,0.33209302325581325,0.3199999999999993,0.1279069767441859,0.11209302325581383])
Large_medium = np.array([0.08046511627906974, 0.06095238095238094, 0.6083720930232552, 0.5762790697674413, 0.21720930232558114, 0.1786046511627904])
Large_high = np.array([0.16744186046511605, 0.1576744186046509,0.55860465116279, 0.5280952380952382])
# 模拟数据生成函数
def generate_data():
    group1 = np.random.randint(1, 10, size=(6))
    group2 = np.random.randint(1, 10, size=(6))
    group3 = np.random.randint(1, 10, size=(4))
    # print(Tokyo_low, group2, group3)
    return [Tokyo_low, Tokyo_medium, Tokyo_high]


# 数据准备：两个子图的数据
data_list = [[Tokyo_low, Tokyo_medium, Tokyo_high], [Large_low, Large_medium, Large_high]]
group_labels = ['L', 'M', 'H']
colors = ['#E91E63', '#4CAF50', '#2196F3', '#FFC107']  # 堆叠颜色

# 创建子图（并排排列）
fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), sharey=True)

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
        tick_labels.extend([f"{traffic}-{cases[j % 6]}" for j in range(num_bars)])
        current_x = xs[-1] + 2  # 每组之间留空隙

    x_positions = np.array(x_positions)
    all_data = np.concatenate(groups)


    # 初始化堆叠底部
    bottom = np.zeros(len(all_data))

    component_list = ['Source', 'Detector', 'Other', 'Ice box']
    lables = []

    import matplotlib.pyplot as plt
    import numpy as np

    used_labels = set()  # 用于追踪已添加图例的标签

    # 预定义6个颜色（你可以根据需要调整颜色）
    colors = [
        '#8DB255',  # ¥1
        '#8DB255',
        '#5C8EA3',  # ¥10
        '#5C8EA3',
        '#C43E3E',  # ¥100
        '#C43E3E',
        '#8DB255',  # ¥1
        '#8DB255',
        '#5C8EA3',  # ¥10
        '#5C8EA3',
        '#C43E3E',  # ¥100
        '#C43E3E',
        '#8DB255',  # ¥1
        '#8DB255',
        '#C43E3E',  # ¥100
        '#C43E3E'
    ]

    hatches = [
        '', '///', '', '///', '', '///'  # 每对加阴影
    ]
    # 每组的柱子数
    group_size = 6

    for x_position_index in range(len(x_positions)):
        # 计算对应的标签
        label = cases[x_position_index]
        print(label)

        # 计算该柱子所属的组
        group_index = x_position_index // group_size

        # 获取该组对应的颜色，确保每组的颜色相同
        color_index = x_position_index # 每6个柱子就循环一次颜色
        base_color = colors[color_index]  # 从颜色列表中选择颜色

        # 如果该标签没有被使用过，则绘制柱子并添加到图例
        if label not in used_labels:
            if x_position_index%2 ==1:
                ax.bar(x_positions[x_position_index], height=all_data[x_position_index] * 100,
                       width=0.8, label=label, color=colors[color_index], hatch = '///')
            else:
                ax.bar(x_positions[x_position_index], height=all_data[x_position_index] * 100,
                       width=0.8, label=label, color=colors[color_index], hatch='')
            used_labels.add(label)  # 将标签加入集合，表示该标签已使用过
        else:
            # 如果标签已经存在，跳过图例的添加，仅绘制柱子
            if x_position_index%2 ==1:
                ax.bar(x_positions[x_position_index], height=all_data[x_position_index]* 100,
                       width=0.8, color=colors[color_index], hatch = '///')
            else:
                ax.bar(x_positions[x_position_index], height=all_data[x_position_index] * 100,
                       width=0.8, color=colors[color_index], hatch='')

    Subplot_list = ['Tokyo', 'NSF']
    # for i in range(1):
    #     component = component_list[i]
    #     ax.bar(x_positions, all_data[:, i], bottom=bottom, width=0.8, color=colors[i],
    #            label=f'{component}' if ax_idx == 0 else "")
    #     bottom += all_data[:, i]

    # 设置子图标题、字体大小
    # ax.set_title(f"{Subplot_list[ax_idx]}", fontsize=20)

    # 设置纵坐标和横坐标标签字体大小
    ax.set_xlabel('Traffic and Cases', fontsize=20)
    ax.set_ylabel('Spectrum Occupation (%)', fontsize=20)

    # 设置x轴标签的字体大小
    tick_positions = [2.5, 9.5, 15.5]
    tick_labels = ['Low', 'Medium', 'High']
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=24)

    # 设置y轴的字体大小
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', labelsize=20)
    # 设置x轴的字体大小
    ax.tick_params(axis='x', labelsize=20)
    if ax_idx == 1:
        ax.tick_params(axis='y', labelleft=True)  # 显示第二个子图的 y 轴数字
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# 添加图例到第一个子图
axes[0].legend(loc='upper center',
               bbox_to_anchor=(0.5, 1.03),  # y 值 > 1 表示上移
               ncol=3,
               fontsize=16,
               columnspacing=0.2)

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
