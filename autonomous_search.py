import subprocess
import time
import os
import re
import numpy as np
import json
import optuna

# 探测器与地图配置
MAP_NAME = "Paris"
TRAFFIC = "Low"
DETECTOR = "SNSPD"
BYPASS = True
TARGET_POWER = 537.0

def get_best_power_from_log(log_path):
    if not os.path.exists(log_path):
        return float('inf')
    with open(log_path, 'r') as f:
        content = f.read()
        powers = re.findall(r"Best Power: ([\d.]+)W", content)
        if powers:
            return float(powers[-1])
    return float('inf')

def run_nn_cma_search(sigma, pop_size, max_iter=150):
    print(f"\n🧠 [NN-CMA] Starting search: sigma={sigma}, pop_size={pop_size}")
    
    # 修改 train_cma.py 中的参数
    with open('train_cma.py', 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if 'cma.CMAEvolutionStrategy(initial_params,' in line:
            new_lines.append(f"        es = cma.CMAEvolutionStrategy(initial_params, {sigma}, {{'popsize': pop_size, 'maxiter': max_iter, 'verb_disp': 1}})\n")
        elif 'best_p = optimizer.train(max_iter=' in line:
            new_lines.append(f"    best_p = optimizer.train(max_iter={max_iter}, pop_size={pop_size})\n")
        else:
            new_lines.append(line)
            
    with open('train_cma.py', 'w') as f:
        f.writelines(new_lines)
    
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    process = subprocess.Popen(['python', '-u', 'train_cma.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    
    best_p = float('inf')
    start_time = time.time()
    
    for line in process.stdout:
        print(line, end='', flush=True)
        if "Best Power:" in line:
            p = float(re.search(r"Best Power: ([\d.]+)W", line).group(1))
            if p < best_p:
                best_p = p
                if best_p <= TARGET_POWER:
                    print(f"\n🎯 NN-CMA TARGET REACHED: {best_p}W!")
                    process.kill()
                    return best_p
        
        if time.time() - start_time > 1800: # 30 mins
             print("\n⏰ NN-CMA timeout.")
             process.kill()
             break
             
    process.wait()
    return best_p

def run_linear_bo_search(n_trials=100):
    print(f"\n📊 [Linear-BO] Starting search with {n_trials} trials...")
    # 这里直接调用 bo_optimize.py 的逻辑，或者通过 subprocess 运行
    # 为了保持独立性，我们修改 bo_optimize.py 的配置并运行
    
    with open('bo_optimize.py', 'r') as f:
        content = f.read()
    
    # 确保配置正确 (SNSPD, Low, Paris, Bypass)
    content = re.sub(r"'Detector': '.*'", f"'Detector': '{DETECTOR}'", content)
    content = re.sub(r"'Traffic': '.*'", f"'Traffic': '{TRAFFIC}'", content)
    content = re.sub(r"n_trials=\d+", f"n_trials={n_trials}", content)
    
    with open('bo_optimize.py', 'w') as f:
        f.write(content)
        
    process = subprocess.Popen(['python', '-u', 'bo_optimize.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    best_p = float('inf')
    for line in process.stdout:
        print(line, end='', flush=True)
        if "最优平均功耗:" in line:
            p = float(re.search(r"最优平均功耗: ([\d.]+) W", line).group(1))
            best_p = p
            
    process.wait()
    return best_p

def main():
    print(f"🌟 Starting Autonomous Search Master Loop 🌟")
    print(f"Target: <= {TARGET_POWER}W | Map: {MAP_NAME} | Detector: {DETECTOR}")
    
    overall_best = float('inf')
    
    strategies = [
        ("NN-CMA", lambda: run_nn_cma_search(0.2, 24)),
        ("Linear-BO", lambda: run_linear_bo_search(100)),
        ("NN-CMA-Fine", lambda: run_nn_cma_search(0.1, 32)),
        ("NN-CMA-Aggressive", lambda: run_nn_cma_search(0.5, 16))
    ]
    
    while overall_best > TARGET_POWER:
        for name, func in strategies:
            print(f"\n{'-'*30}")
            print(f"Current Strategy: {name}")
            print(f"Global Best: {overall_best}W")
            print(f"{'-'*30}")
            
            res = func()
            if res < overall_best:
                overall_best = res
                
            if overall_best <= TARGET_POWER:
                print(f"\n🎊 MISSION ACCOMPLISHED! Best Power: {overall_best}W")
                # 记录最终结果
                with open('final_success.log', 'w') as f:
                    f.write(f"Success with {name}\nPower: {overall_best}W\nTime: {time.ctime()}\n")
                return

if __name__ == "__main__":
    main()
