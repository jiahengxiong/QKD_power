
import sys
import os

# 确保能找到 QEnergy 和 utils
sys.path.append(os.getcwd())

import config
from QEnergy.studies.key_rate_compute import compute_key_rate
from QEnergy.studies.power_compute import compute_power

def analyze_protocol(protocol, detector, distances):
    print(f"\n--- Analyzing {protocol} with {detector} ---")
    for d in distances:
        skr = compute_key_rate(d, protocol, detector)
        p = compute_power(d, protocol, detector)
        print(f"Dist: {d:3}km | SKR: {skr/1e6:6.3f} Mbps | Power: {p['total']:6.2f}W (Source: {p['source']:5.2f}, Det: {p['detector']:5.2f}, Other: {p['other']:5.2f})")

if __name__ == "__main__":
    # 模拟 Large 拓扑的常见跨度
    distances = [30, 50, 80, 100, 130]
    
    # CV-QKD
    analyze_protocol('CV-QKD', 'ThorlabsPDB', distances)
    
    # BB84 + SNSPD
    analyze_protocol('BB84', 'SNSPD', distances)
    
    # BB84 + APD
    analyze_protocol('BB84', 'APD', distances)
