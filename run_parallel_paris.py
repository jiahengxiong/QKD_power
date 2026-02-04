import subprocess
import os
import json
import time
from multiprocessing import Pool

def run_case(config):
    protocol = config['protocol']
    detector = config['detector']
    traffic = config['traffic']
    bypass = config['bypass']
    
    log_file = f"log_Paris_{protocol}_{detector}_{traffic}_Bypass_{bypass}.txt"
    cmd = [
        "python3", "-u", "train_cma.py",
        "--protocol", protocol,
        "--detector", detector,
        "--traffic", traffic,
        "--bypass", str(bypass),
        "--max_iter", "150",  # 并行跑，迭代数稍微降低一点点以保证时间，或者根据需要调整
        "--pop_size", "32"   # 并行跑，种群稍微减小一点以减少内存压力，或者根据服务器性能调整
    ]
    
    print(f"🚀 Starting: {protocol}, {detector}, {traffic}, Bypass={bypass}")
    
    with open(log_file, "w") as f:
        env = os.environ.copy()
        env["MKL_THREADING_LAYER"] = "GNU"
        env["OMP_NUM_THREADS"] = "1"
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        process.wait()
    
    print(f"✅ Finished: {protocol}, {detector}, {traffic}, Bypass={bypass}")
    return True

def main():
    cases = [
        {"protocol": "BB84", "detector": "APD", "traffic": "Low", "bypass": True},
        {"protocol": "BB84", "detector": "APD", "traffic": "Medium", "bypass": True},
        {"protocol": "BB84", "detector": "SNSPD", "traffic": "Low", "bypass": True},
        {"protocol": "BB84", "detector": "SNSPD", "traffic": "Medium", "bypass": True},
        {"protocol": "BB84", "detector": "SNSPD", "traffic": "High", "bypass": True},
        
        {"protocol": "BB84", "detector": "APD", "traffic": "Low", "bypass": False},
        {"protocol": "BB84", "detector": "APD", "traffic": "Medium", "bypass": False},
        {"protocol": "BB84", "detector": "SNSPD", "traffic": "Low", "bypass": False},
        {"protocol": "BB84", "detector": "SNSPD", "traffic": "Medium", "bypass": False},
        {"protocol": "BB84", "detector": "SNSPD", "traffic": "High", "bypass": False},
        
        {"protocol": "CV-QKD", "detector": "ThorlabsPDB", "traffic": "Low", "bypass": True},
        {"protocol": "CV-QKD", "detector": "ThorlabsPDB", "traffic": "Medium", "bypass": True},
        {"protocol": "CV-QKD", "detector": "ThorlabsPDB", "traffic": "High", "bypass": True},
        
        {"protocol": "CV-QKD", "detector": "ThorlabsPDB", "traffic": "Low", "bypass": False},
        {"protocol": "CV-QKD", "detector": "ThorlabsPDB", "traffic": "Medium", "bypass": False},
        {"protocol": "CV-QKD", "detector": "ThorlabsPDB", "traffic": "High", "bypass": False},
    ]

    print(f"🌟 Starting Parallel Execution for 16 Paris Cases 🌟")
    
    # 使用 16 个进程并行运行
    with Pool(processes=16) as pool:
        pool.map(run_case, cases)

    print("\n📊 All experiments finished. Collecting results...")
    
    results = []
    for config in cases:
        filename = f"results_Paris_{config['protocol']}_{config['detector']}_{config['traffic']}_Bypass_{config['bypass']}.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                results.append(json.load(f))
        else:
            results.append({**config, "best_power": "FAILED"})

    # 打印汇总表
    print("\n" + "="*80)
    print(f"{'Protocol':<10} | {'Detector':<12} | {'Traffic':<8} | {'Bypass':<8} | {'Best Power':<12}")
    print("-" * 80)
    for r in results:
        power_str = f"{r['best_power']:.2f}W" if isinstance(r['best_power'], (int, float)) else "FAILED"
        print(f"{r['protocol']:<10} | {r['detector']:<12} | {r['traffic']:<8} | {str(r['bypass']):<8} | {power_str:<12}")
    print("="*80)

if __name__ == "__main__":
    main()
