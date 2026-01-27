# QKD 网络算法改进方案 (集成模拟退火)

## 1. 热力图增强 (Heatmap Upgrade)
- **业务感知**：修改 [main.py](file:///Users/xiongjiaheng/QDK_power_journal/QKD_power/main.py)，在 `link_future_demand` 中引入 `traffic` 权重。
- **宿节点预判**：统计未来请求的终点分布 `node_future_demand`。
- **功耗奖励**：修改 [tools.py](file:///Users/xiongjiaheng/QDK_power_journal/QKD_power/utils/tools.py)，对高热度节点的新开冰箱功耗给予折扣奖励，引导资源共享。

## 2. 模拟退火路径探索 (Simulated Annealing)
- **温度控制**：根据运行次数 `run` 和请求进度设置温度 $T$。
- **接受准则**：
  - 计算 $\Delta W = W_{candidate} - W_{best}$。
  - 遵循 Metropolis 准则：若 $\Delta W > 0$，以 $P = \exp(-\Delta W / T)$ 的概率接受次优路径。
- **路径搜索**：修改 `find_min_weight_path_with_relay` 返回多个候选路径供 SA 选择。

## 3. 实施步骤
1. 修改 `main.py` 统计逻辑，生成双重热力图。
2. 更新 `tools.py` 中的 `calculate_data_auxiliary_edge` 权重计算逻辑。
3. 在 `main.py` 中实现模拟退火选择器并替换原有贪心选择逻辑。
