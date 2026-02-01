
import config
import utils.tools
import main
from utils.Network import Network
from utils.traffic_generater import gen_traffic_matrix
import numpy as np
import random
import os

def run_single_test(case, weights):
    print(f"\nTesting with weights: {weights}")
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
    success_count = 0
    
    # 增加测试请求数量
    test_limit = 100 
    traffic_subset = traffic_matrix[:test_limit]
    
    for i, request in enumerate(traffic_subset):
        src, dst, traffic = request[1], request[2], request[3]
        auxiliary_graph = utils.tools.build_auxiliary_graph(
            topology=topology, wavelength_list=wavelength_list, traffic=traffic, 
            physical_topology=physical_topology, shared_key_rate_list=shared_key_rate, 
            served_request=served_request, remain_num_request=len(traffic_subset) - i
        )
        
        result = main.find_min_weight_path_with_relay(auxiliary_graph, src, dst)
        if result:
            direction, best_path_edges, relay, min_power, weight = result
            _, actual_p = main.serve_traffic(
                G=topology, AG=auxiliary_graph, path_edge_list=best_path_edges, 
                request_traffic=traffic, pbar=None, served_request=served_request
            )
            case_power += sum(actual_p.values())
            success_count += 1
            
    avg_power = case_power / len(traffic_subset) if len(traffic_subset) > 0 else 0
    print(f"Result -> Avg Power: {avg_power:.4f}W, Success: {success_count}/{len(traffic_subset)}")
    return avg_power

if __name__ == '__main__':
    # 选取用户指定的 Paris Case
    test_case = {
        'Topology': 'Paris', 
        'Protocol': 'BB84', 
        'Bypass': True, 
        'Detector': 'APD', 
        'Traffic': 'Low'
    }
    
    # 权重集合 1: BO 找到的最优权重 (Paris/APD/Low)
    w1 = {
        "a0": 3.7697949995850895,
        "a1": 0.4933275380266298,
        "a2": 361.2361926232809,
        "a3": 47.85048775475712,
        "a4": 3064.2958154862154,
        "a5": 23.312869200484027,
        "a6": 277.3366097415615,
        "a7": 1266.9131730459962,
        "a8": 17.476897878074944,
        "a9": 0.5470785903767319,
        "a10": 0.1908263182995043
    }
    
    # 权重集合 2: 极度关注距离 (a6)
    w2 = dict(w1)
    w2['a6'] = 1000.0
    
    # 权重集合 3: 默认/均衡
    w3 = {'a0': 1.0, 'a1': 1.0, 'a2': 1.0, 'a3': 1.0, 'a4': 1.0, 'a5': 1.0, 'a6': 1.0, 'a7': 1.0, 'a8': 1.0, 'a9': 1.0, 'a10': 1.0}

    p1 = run_single_test(test_case, w1)
    p2 = run_single_test(test_case, w2)
    p3 = run_single_test(test_case, w3)

    # 权重集合 4: Bypass=False (对比组)
    print("\n--- Testing Bypass=False (Baseline) ---")
    test_case_no_bypass = dict(test_case)
    test_case_no_bypass['Bypass'] = False
    p4 = run_single_test(test_case_no_bypass, w3) # w3 is default
    
    print("\n" + "="*30)
    print(f"Power 1 (Power-focused): {p1:.4f}W")
    print(f"Power 2 (Dist-focused):  {p2:.4f}W")
    print(f"Power 3 (Default):       {p3:.4f}W")
    print(f"Power 4 (No Bypass):     {p4:.4f}W")
    
    if p1 < p4:
        print("\nSUCCESS: Bypass Power < No Bypass Power!")
    else:
        print("\nFAILURE: Bypass Power >= No Bypass Power.")
    
    if p1 == p2 == p3:
        print("\nWARNING: RESULTS ARE IDENTICAL! Optimization is likely insensitive.")
    else:
        print("\nSUCCESS: RESULTS DIFFER. Weights are working.")
