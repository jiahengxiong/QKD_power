#!/usr/bin/env python3
"""
Test script to verify computer_power feature integration with GNN policy network
"""

import numpy as np
import torch
import networkx as nx
from utils.tools import extract_feature_matrices_from_graph, calculate_data_auxiliary_edge
from rl_models import QKDGraphNet

def test_computer_power_integration():
    """Test that per-node computer power planes are correctly extracted and used by GNN"""
    
    print("=== Testing Computer Power Integration ===")
    
    # Create a simple test graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    
    # Add distances
    for u, v in G.edges():
        G[u][v]['distance'] = 100.0
    
    # Create auxiliary graph with computer_power data
    auxiliary_graph = nx.MultiDiGraph()
    
    # Add some test edges with per-node computer power map
    test_data = {
        'distance': 100.0,
        'power': 500.0,
        'source_power': 200.0,
        'detector_power': 150.0,
        'other_power': 100.0,
        'computer_node_power_map': {0: 150.0, 1: 150.0},
        'fridge_node_power_map': {},
        'path': [0, 1, 2],
        'wavelength_list': ['wl1'],
        'wavelength_power_info': {'wl1': {'source': 200.0, 'detector': 150.0, 'other': 100.0}},
        'wavelength_bypass_info': {'wl1': 1},
        'wavelength_dist_info': {'wl1': 100.0},
        'wavelength_traffic': {'wl1': 50.0},
        'raw_features': {
            'raw_unit': 10.0,
            'num_new_LD': 1,
            'num_wls': 1,
            'delta_spectrum': 0.5,
            'raw_invcap': 0.8,
            'f_dist': 1.25,
            'num_new_fridges': 0,
            'min_free_cap': 100.0,
            'f_occ': 0.3,
            'f_waste': 0.2,
            'f_bypass': 1.0
        }
    }
    
    # Add test edges
    auxiliary_graph.add_edge(0, 1, **test_data)
    auxiliary_graph.add_edge(1, 2, **test_data)
    
    # Create node mapping
    node_to_idx = {0: 0, 1: 1, 2: 2}
    num_nodes = 3
    wavelength_list = ['wl1']
    
    print("1. Testing feature extraction...")
    
    # Extract features
    global_tensor, wl_tensor = extract_feature_matrices_from_graph(
        auxiliary_graph, node_to_idx, num_nodes, wavelength_list, topology=G
    )
    
    print(f"   Global tensor shape: {global_tensor.shape}")
    print(f"   Wavelength tensor shape: {wl_tensor.shape}")
    
    base = 6
    ch_node0 = base + node_to_idx[0]
    ch_node1 = base + node_to_idx[1]
    print(f"   Per-node computer power planes:")
    print(f"   Edge (0,1) node0 (ch {ch_node0}): {global_tensor[ch_node0, 0, 1]}")
    print(f"   Edge (0,1) node1 (ch {ch_node1}): {global_tensor[ch_node1, 0, 1]}")
    expected_val = 150.0
    if abs(global_tensor[ch_node0, 0, 1] - expected_val) < 1e-6 and abs(global_tensor[ch_node1, 0, 1] - expected_val) < 1e-6:
        print("   ✓ Per-node computer power correctly extracted for edge (0,1)")
    else:
        print("   ✗ Per-node computer power mismatch for edge (0,1)")
    
    print("\n2. Testing GNN processing...")
    
    # Test with GNN model
    model = QKDGraphNet(
        num_global_features=6 + 2 * num_nodes,
        num_wl_features=5,
        num_wavelengths=1,
        actual_nodes=3,
        is_bypass=True,
        hidden_dim=8
    )
    
    # Convert to torch tensors
    global_tensor_torch = torch.FloatTensor(global_tensor).unsqueeze(0)  # Add batch dimension
    wl_tensor_torch = torch.FloatTensor(wl_tensor).unsqueeze(0)  # Add batch dimension
    
    # Create context (src, dst, traffic, protocol)
    context = torch.FloatTensor([[0.0, 2.0, np.log1p(50.0)/15.0, 1.0]])
    
    # Create dummy last_action and h_prev
    last_action = None
    h_prev = torch.zeros(1, 8)  # hidden_dim=8
    
    try:
        mu, std, h_next = model(global_tensor_torch, wl_tensor_torch, context, last_action, h_prev)
        print(f"   ✓ GNN forward pass successful")
        print(f"   Output mu shape: {mu.shape}")
        print(f"   Output std shape: {std.shape}")
        print(f"   Hidden state shape: {h_next.shape}")
        
        # Check that outputs are reasonable
        if not torch.isnan(mu).any() and not torch.isinf(mu).any():
            print("   ✓ Output mu values are valid")
        else:
            print("   ✗ Output mu contains NaN or Inf")
            
        if not torch.isnan(std).any() and not torch.isinf(std).any():
            print("   ✓ Output std values are valid")
        else:
            print("   ✗ Output std contains NaN or Inf")
            
    except Exception as e:
        print(f"   ✗ GNN forward pass failed: {e}")
        return False
    
    print("\n3. Testing with different per-node computer power maps...")
    
    test_cases = [
        {'computer_node_power_map': {}},
        {'computer_node_power_map': {0: 150.0}},
        {'computer_node_power_map': {0: 150.0, 1: 150.0}},
    ]
    
    for i, test_case in enumerate(test_cases):
        # Update test data
        test_data_copy = test_data.copy()
        test_data_copy.update(test_case)
        
        # Create new auxiliary graph
        aux_graph_test = nx.MultiDiGraph()
        aux_graph_test.add_edge(0, 1, **test_data_copy)
        
        # Extract features
        global_test, _ = extract_feature_matrices_from_graph(
            aux_graph_test, node_to_idx, num_nodes, wavelength_list, topology=G
        )
        
        v0 = global_test[base + node_to_idx[0], 0, 1]
        v1 = global_test[base + node_to_idx[1], 0, 1]
        exp0 = 150.0 if 0 in test_case['computer_node_power_map'] else 0.0
        exp1 = 150.0 if 1 in test_case['computer_node_power_map'] else 0.0
        if abs(v0 - exp0) < 1e-6 and abs(v1 - exp1) < 1e-6:
            print(f"   ✓ Test case {i+1}: per-node planes correctly extracted")
        else:
            print(f"   ✗ Test case {i+1}: expected ({exp0},{exp1}), got ({v0},{v1})")
    
    print("\n=== All tests completed ===")
    return True

if __name__ == "__main__":
    test_computer_power_integration()
