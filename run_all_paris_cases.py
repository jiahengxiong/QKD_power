
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
    """执行单个 Case 的仿真并返回详细数据"""
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
        total_p_dict = {'source': 0.0, 'detector': 0.0, 'other': 0.0, 'ice_box': 0.0}
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
                for k in total_p_dict:
                    total_p_dict[k] += actual_p.get(k, 0.0)
                success_count += 1
                
        blocking_rate = (len(traffic_matrix) - success_count) / len(traffic_matrix)
        
        # 计算平均功耗
        avg_p_dict = {k: v / len(traffic_matrix) for k, v in total_p_dict.items()}
        total_avg_power = sum(avg_p_dict.values())
        
        # 计算平均频谱占用率
        total_links = 0
        occupied_links = 0
        for u, v in topology.edges():
            edge_dict = topology[u][v]
            total_links += len(edge_dict)
            occupied_links += sum(1 for _, d in edge_dict.items() if d.get('occupied'))
        spectrum_occ = occupied_links / total_links if total_links > 0 else 0.0
        
        return {
            'score': total_avg_power + blocking_rate * 2000000 + spectrum_occ,
            'avg_power': total_avg_power,
            'component_power': avg_p_dict,
            'spectrum_occ': spectrum_occ,
            'blocking_rate': blocking_rate
        }
        
    finally:
        # 恢复随机状态
        np.random.set_state(np_state)
        random.setstate(py_state)

def run_bo_for_case(case, n_trials=100, early_stopping_patience=100):
    """为指定 Case 寻找最优权重并返回完整结果，包含早停机制"""
    best_res = {'score': float('inf')}
    best_params = {}

    def objective(trial):
        nonlocal best_res, best_params
        # 14 维参数空间：8个系数 + 6个幂指数 (支持负指数)
        weights = {
            'a0': trial.suggest_float('a0', 0.01, 2.0),
            'a1': trial.suggest_float('a1', 0.1, 500.0, log=True),
            'p1': trial.suggest_float('p1', -2.5, 2.5),             # 功耗指数
            'a2': trial.suggest_float('a2', 10.0, 2000.0, log=True),
            'p2': trial.suggest_float('p2', -2.5, 2.5),             # 新硬件指数
            'a3': trial.suggest_float('a3', 1.0, 500.0, log=True),
            'p3': trial.suggest_float('p3', -2.5, 2.5),             # 波长指数
            'a4': trial.suggest_float('a4', 1.0, 500.0, log=True),
            'p4': trial.suggest_float('p4', -2.5, 2.5),             # 频谱指数
            'a5': 0.0,
            'a6': 0.0,
            'a7': trial.suggest_float('a7', 100.0, 5000.0, log=True),
            'p7': trial.suggest_float('p7', -2.5, 2.5),             # 冰箱指数
            'a8': trial.suggest_float('a8', 0.1, 200.0, log=True),
            'a9': 0.0,
            'a10': trial.suggest_float('a10', 0.0, 5.0),
            'p10': trial.suggest_float('p10', -2.5, 2.5)            # 奖励指数
        }
        res = run_simulation(case, weights)
        if res['score'] < best_res['score']:
            best_res = res
            best_params = weights
        return res['score']

    # 早停回调类
    class EarlyStoppingCallback:
        def __init__(self, patience):
            self.patience = patience
            self.best_score = None
            self.no_improvement_count = 0

        def __call__(self, study, trial):
            if self.best_score is None or trial.value < self.best_score:
                self.best_score = trial.value
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            if self.no_improvement_count >= self.patience:
                study.stop()

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    early_stopping = EarlyStoppingCallback(patience=early_stopping_patience)
    study.optimize(objective, n_trials=n_trials, callbacks=[early_stopping])
    
    return best_res, best_params

def run_single_experiment(args):
    """运行单个 (Bypass 或 No Bypass) 实验并进行 BO 优化"""
    protocol, detector, traffic, bypass = args
    strategy = "Bypass" if bypass else "NoBypass"
    print(f"\n>>> Starting Independent BO: Protocol={protocol}, Detector={detector}, Traffic={traffic}, Strategy={strategy}")
    
    case = {
        'Topology': 'Tokyo', 'Protocol': protocol, 'Detector': detector, 
        'Traffic': traffic, 'Bypass': bypass
    }
    res, best_params = run_bo_for_case(case, n_trials=200)
    
    print(f"  [DONE] {protocol}-{detector}-{traffic}-{strategy}: Power={res['avg_power']:.2f}W")
    
    return {
        'protocol': protocol,
        'detector': detector,
        'traffic': traffic,
        'bypass': bypass,
        'res': {
            'power': res['avg_power'],
            'component_power': res['component_power'],
            'spectrum_occ': res['spectrum_occ'],
            'best_params': best_params
        }
    }

if __name__ == '__main__':
    base_configs = [
        ('BB84', 'APD', 'Low'),
        ('BB84', 'APD', 'Medium'),
        ('BB84', 'SNSPD', 'Low'),
        ('BB84', 'SNSPD', 'Medium'),
        ('BB84', 'SNSPD', 'High'),
        ('CV-QKD', 'ThorlabsPDB', 'Low'),
        ('CV-QKD', 'ThorlabsPDB', 'Medium'),
        ('CV-QKD', 'ThorlabsPDB', 'High')
    ]
    
    # 构造 16 个独立任务 (8 组 * 2 种策略)
    tasks = []
    for protocol, detector, traffic in base_configs:
        tasks.append((protocol, detector, traffic, False)) # No Bypass
        tasks.append((protocol, detector, traffic, True))  # Bypass
    
    print(f"Starting {len(tasks)} independent experiments in parallel with 16 processes...")
    
    # 使用 16 个进程并行执行
    with Pool(processes=16) as pool:
        raw_results = pool.map(run_single_experiment, tasks)
        
    # 将结果重新组合成对比格式
    results_map = {}
    for r in raw_results:
        key = (r['protocol'], r['detector'], r['traffic'])
        if key not in results_map:
            results_map[key] = {'protocol': r['protocol'], 'detector': r['detector'], 'traffic': r['traffic']}
        
        if r['bypass']:
            results_map[key]['bypass'] = r['res']
        else:
            results_map[key]['nobypass'] = r['res']
            
    final_results = []
    for key in results_map:
        r = results_map[key]
        nb_p = r['nobypass']['power']
        b_p = r['bypass']['power']
        r['improvement'] = (nb_p - b_p) / nb_p * 100 if nb_p > 0 else 0
        final_results.append(r)
    
    # 打印最终表格
    print("\n\n" + "="*80)
    print(f"{'Protocol':<10} | {'Detector':<12} | {'Traffic':<8} | {'NoBypass(W)':<12} | {'Bypass(W)':<10} | {'Gain(%)':<8}")
    print("-"*80)
    for r in final_results:
        print(f"{r['protocol']:<10} | {r['detector']:<12} | {r['traffic']:<8} | {r['nobypass']['power']:>12.2f} | {r['bypass']['power']:>10.2f} | {r['improvement']:>7.2f}%")
    print("="*80)
    
    with open('tokyo.json', 'w') as f:
        json.dump(final_results, f, indent=4)
