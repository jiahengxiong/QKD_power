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
    """Test that computer_power is correctly extracted and used by GNN"""
    
    print("=== Testing Computer Power Integration ===")
    
    # Create a simple test graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    
    # Add distances
    for u, v in G.edges():
        G[u][v]['distance'] = 100.0
    
    # Create auxiliary graph with computer_power data
    auxiliary_graph = nx.MultiDiGraph()
    
    # Add some test edges with computer_power information
    test_data = {
        'distance': 100.0,
        'power': 500.0,
        'source_power': 200.0,
        'detector_power': 150.0,
        'other_power': 100.0,
        'computer_power': 150.0,  # This is what we want to test
        'computer_nodes': [0, 1],  # Nodes using computer power
        'ice_box_power': 0.0,
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
    
    # Check if computer_power is in channel 7
    computer_power_channel = global_tensor[7]  # Channel 7 is computer_power
    print(f"   Computer power channel (7) values:")
    print(f"   Edge (0,1): {computer_power_channel[0, 1]}")
    print(f"   Edge (1,2): {computer_power_channel[1, 2]}")
    
    # Verify values
    expected_computer_power = 150.0
    if abs(computer_power_channel[0, 1] - expected_computer_power) < 1e-6:
        print("   ✓ Computer power correctly extracted for edge (0,1)")
    else:
        print(f"   ✗ Computer power mismatch for edge (0,1): expected {expected_computer_power}, got {computer_power_channel[0, 1]}")
    
    if abs(computer_power_channel[1, 2] - expected_computer_power) < 1e-6:
        print("   ✓ Computer power correctly extracted for edge (1,2)")
    else:
        print(f"   ✗ Computer power mismatch for edge (1,2): expected {expected_computer_power}, got {computer_power_channel[1, 2]}")
    
    print("\n2. Testing GNN processing...")
    
    # Test with GNN model
    model = QKDGraphNet(
        num_global_features=8 + 2 * num_nodes,
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
    context = torch.FloatTensor([[0.0, 2.0/3.0, np.log1p(50.0)/15.0, 1.0]])  # Normalized
    
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
    
    print("\n3. Testing with different computer_power values...")
    
    # Test with different computer power values
    test_cases = [
        {'computer_power': 0.0, 'computer_nodes': []},
        {'computer_power': 150.0, 'computer_nodes': [0]},
        {'computer_power': 300.0, 'computer_nodes': [0, 1]},
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
        
        computer_power_val = global_test[7, 0, 1]
        expected_val = test_case['computer_power']
        
        if abs(computer_power_val - expected_val) < 1e-6:
            print(f"   ✓ Test case {i+1}: computer_power={expected_val} correctly extracted")
        else:
            print(f"   ✗ Test case {i+1}: expected {expected_val}, got {computer_power_val}")
    
    print("\n=== All tests completed ===")
    return True

if __name__ == "__main__":
    test_computer_power_integration()
