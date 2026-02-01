
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
        
        # 返回物理功耗和阻塞率惩罚
        return avg_power + blocking_rate * 2000000
        
    finally:
        # 恢复随机状态
        np.random.set_state(np_state)
        random.setstate(py_state)

def run_bo_for_bypass(case, n_trials=30):
    """为 Bypass 场景寻找最优权重"""
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
            'a8': trial.suggest_float('a8', 0.1, 500.0, log=True),
            'a9': trial.suggest_float('a9', 0.1, 500.0, log=True),
            'a10': trial.suggest_float('a10', 0.1, 1000.0, log=True)
        }
        return run_simulation(case, weights)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    best_power = study.best_value if study.best_value < 1000000 else (study.best_value % 1000000)
    return best_power, study.best_params

def run_experiment_pair(protocol, detector, traffic):
    """运行一组 (Bypass vs No Bypass) 对比实验"""
    print(f"\n>>> Starting Independent BO Experiment: Protocol={protocol}, Detector={detector}, Traffic={traffic}")
    
    # 1. No Bypass Baseline (使用默认权重)
    nobypass_case = {
        'Topology': 'Paris', 'Protocol': protocol, 'Detector': detector, 
        'Traffic': traffic, 'Bypass': False
    }
    default_weights = {k: 1.0 for k in ['a0','a1','a2','a3','a4','a5','a6','a7','a8','a9','a10']}
    nobypass_power = run_simulation(nobypass_case, default_weights)
    if nobypass_power > 1000000:
        nobypass_power = nobypass_power % 1000000
        nobypass_status = "BLOCKED"
    else:
        nobypass_status = "SUCCESS"
    
    # 2. Bypass with BO (增加迭代次数至 100 次，确保充分收敛)
    bypass_case = {
        'Topology': 'Paris', 'Protocol': protocol, 'Detector': detector, 
        'Traffic': traffic, 'Bypass': True
    }
    bypass_power, best_params = run_bo_for_bypass(bypass_case, n_trials=100)
    
    print(f"  [DONE] {protocol}-{detector}-{traffic}: NoBypass={nobypass_power:.2f}W, Bypass={bypass_power:.2f}W")
    
    return {
        'protocol': protocol,
        'detector': detector,
        'traffic': traffic,
        'nobypass_power': nobypass_power,
        'bypass_power': bypass_power,
        'improvement': (nobypass_power - bypass_power) / nobypass_power * 100 if nobypass_power > 0 else 0,
        'best_params': best_params
    }

def run_experiment_pair_wrapper(args):
    """用于多进程调用的包装器"""
    return run_experiment_pair(*args)

if __name__ == '__main__':
    pairs = [
        ('BB84', 'APD', 'Low'),
        ('BB84', 'APD', 'Medium'),
        ('BB84', 'SNSPD', 'Low'),
        ('BB84', 'SNSPD', 'Medium'),
        ('BB84', 'SNSPD', 'High'),
        ('CV-QKD', 'ThorlabsPDB', 'Low'),
        ('CV-QKD', 'ThorlabsPDB', 'Medium'),
        ('CV-QKD', 'ThorlabsPDB', 'High')
    ]
    
    print(f"Starting experiments for {len(pairs)} pairs in parallel...")
    
    # 使用进程池并行执行 8 组实验
    with Pool(processes=min(len(pairs), os.cpu_count())) as pool:
        results = pool.map(run_experiment_pair_wrapper, pairs)
        
    # 打印最终表格
    print("\n\n" + "="*80)
    print(f"{'Protocol':<10} | {'Detector':<12} | {'Traffic':<8} | {'NoBypass(W)':<12} | {'Bypass(W)':<10} | {'Gain(%)':<8}")
    print("-"*80)
    for r in results:
        print(f"{r['protocol']:<10} | {r['detector']:<12} | {r['traffic']:<8} | {r['nobypass_power']:>12.2f} | {r['bypass_power']:>10.2f} | {r['improvement']:>7.2f}%")
    print("="*80)
    
    with open('paris_all_results.json', 'w') as f:
        json.dump(results, f, indent=4)
