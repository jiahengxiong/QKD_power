import optuna
import config
import utils.tools
import main
from utils.Network import Network
from utils.traffic_generater import gen_traffic_matrix
import numpy as np
import random
import os
import sys
import json
from multiprocessing import Pool

# 抑制仿真过程中的冗余输出
class DummyPbar:
    def write(self, s): pass
    def update(self, n): pass

def run_simulation(case, weights):
    """执行单个 Case 的仿真并返回平均功耗"""
    import config
    import numpy as np
    import random
    
    # 保存当前的随机状态
    np_state = np.random.get_state()
    py_state = random.getstate()
    
    try:
        config.weights = weights
        
        map_name = case['Topology']
        traffic_type = case['Traffic']
        protocol = case['Protocol']
        detector = case['Detector']
        bypass = case['Bypass']
        
        config.protocol = protocol
        config.detector = detector
        config.bypass = bypass
        config.random_seed = 42
        config.ice_box_capacity = 8
        
        # 为仿真设置固定种子
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        utils.tools.clear_path_cache()
        
        wavelength_list = np.linspace(1530, 1565, 10).tolist()
        network = Network(map_name=map_name, wavelength_list=wavelength_list, 
                         protocol=protocol, receiver=detector)
        topology = network.topology
        physical_topology = network.physical_topology
        traffic_matrix = gen_traffic_matrix(traffic_type, map_name)
        
        served_request = {}
        case_power = 0.0
        shared_key_rate = {}
        dummy_pbar = DummyPbar()
        success_count = 0
        
        for i, request in enumerate(traffic_matrix):
            src, dst, traffic = request[1], request[2], request[3]
            auxiliary_graph = utils.tools.build_auxiliary_graph(
                topology=topology, wavelength_list=wavelength_list, traffic=traffic, 
                physical_topology=physical_topology, shared_key_rate_list=shared_key_rate, 
                served_request=served_request, remain_num_request=len(traffic_matrix) - i
            )
            result = main.find_min_weight_path_with_relay(auxiliary_graph, src, dst)
            if result:
                direction, best_path_edges, relay, min_power, weight = result
                _, actual_p = main.serve_traffic(
                    G=topology, AG=auxiliary_graph, path_edge_list=best_path_edges, 
                    request_traffic=traffic, pbar=dummy_pbar, served_request=served_request
                )
                case_power += sum(actual_p.values())
                success_count += 1
                
        blocking_rate = (len(traffic_matrix) - success_count) / len(traffic_matrix)
        avg_power = case_power / len(traffic_matrix) if len(traffic_matrix) > 0 else 0
        return avg_power + blocking_rate * 2000000
        
    finally:
        # 恢复随机状态，确保不干扰 Optuna 的采样器
        np.random.set_state(np_state)
        random.setstate(py_state)

def run_bo_for_single_case(case):
    """为一个具体的 Case 运行独立的贝叶斯优化"""
    case_name = f"{case['Protocol']}-{case['Detector']}-Bypass:{case['Bypass']}-{case['Traffic']}"
    
    def objective(trial):
        weights = {
            'a0': trial.suggest_float('a0', 0.0, 5.0),
            'a1': trial.suggest_float('a1', 0.1, 1000.0, log=True),
            'a2': trial.suggest_float('a2', 10.0, 5000.0, log=True),
            'a3': trial.suggest_float('a3', 10.0, 5000.0, log=True),
            'a4': trial.suggest_float('a4', 10.0, 5000.0, log=True),
            'a5': trial.suggest_float('a5', 0.1, 500.0, log=True),
            'a6': trial.suggest_float('a6', 0.1, 500.0, log=True),
            'a7': trial.suggest_float('a7', 100.0, 10000.0, log=True),
            'a8': trial.suggest_float('a8', 0.1, 500.0, log=True),   # f_occ
            'a9': trial.suggest_float('a9', 0.1, 500.0, log=True),   # f_waste
            'a10': trial.suggest_float('a10', 0.1, 1000.0, log=True) # f_bypass reward
        }
        val = run_simulation(case, weights)
        print(f"[Trial {trial.number}] Power: {val:.4f} W | a1={weights['a1']:.2f}, a10={weights['a10']:.2f}")
        return val

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=50) 
    
    # 提取纯物理功耗
    final_power = study.best_value if study.best_value < 1000000 else (study.best_value % 1000000)
    blocking_status = "SUCCESS" if study.best_value < 1000000 else "FAILED"

    result = {
        'case_name': case_name,
        'best_power': final_power,
        'status': blocking_status,
        'best_params': study.best_params
    }
    
    # 实时记录到文件
    with open('result.txt', 'a') as f:
        f.write(f"DONE: {case_name} | Best Avg Power: {final_power:.2f}W | {blocking_status}\n")
        f.write(f"Best Weights: {json.dumps(study.best_params)}\n\n")
        
    return result

if __name__ == '__main__':
    with open('result.txt', 'w') as f:
        f.write("--- Specific BO Optimization Test: Paris Topology (APD, Low, Bypass:True) ---\n\n")

    # 专门针对 Paris + APD + Low + Bypass 的场景
    test_case = {
        'Topology': 'Paris',
        'Traffic': 'Low',
        'Protocol': 'BB84',
        'Detector': 'APD',
        'Bypass': True
    }
    
    print(f"开始为 Paris 场景运行深度优化 (100 Trials)...")
    
    # 为了看到详细过程，我们不使用多进程，直接运行一个 Study
    result = run_bo_for_single_case(test_case)
    
    print("\n" + "="*60)
    print(f"优化完成！")
    print(f"Case: {result['case_name']}")
    print(f"最优平均功耗: {result['best_power']:.4f} W")
    print(f"最优参数: {json.dumps(result['best_params'], indent=4)}")
    print("="*60)
