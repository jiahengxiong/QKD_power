import json
import os
import time
import sys
import torch

# [Performance] 限制每个 Worker 的线程数，避免多进程 CPU 争用
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Multiprocessing defaults.
os.environ.setdefault("QKD_MAX_WORKERS", "32")
os.environ.setdefault("QKD_MAX_TASKS_PER_CHILD", "0")
# Default to no hard timeout: some Large/CV-QKD Bypass cases are legitimately slow.
# Set QKD_MAP_TIMEOUT_SEC manually if you want timeout protection for debugging.
os.environ.setdefault("QKD_MAP_TIMEOUT_SEC", "0")
os.environ.setdefault("QKD_MAP_POLL_SEC", "5")
os.environ.setdefault("QKD_MAP_CHUNKSIZE", "1")

# 动态添加路径，确保能找到 train_cma
sys.path.append(os.getcwd())
from train_cma import run_experiment

try:
    import train_cma
except Exception as e:
    import traceback
    print("❌ Could not import train_cma. Real error is:")
    traceback.print_exc()
    raise

# 定义测试用例
TOPOLOGIES = ["Paris", "Tokyo", "Large"]

# 9种组合 (Protocol, Detector, Traffic)
CONFIGS = [
    # BB84 + APD (No High Traffic)
    ("BB84", "APD", "Low"),
    ("BB84", "APD", "Medium"),
    
    # BB84 + SNSPD
    ("BB84", "SNSPD", "Low"),
    ("BB84", "SNSPD", "Medium"),
    ("BB84", "SNSPD", "High"),
    
    # CV-QKD + ThorlabsPDB
    ("CV-QKD", "ThorlabsPDB", "Low"),
    ("CV-QKD", "ThorlabsPDB", "Medium"),
    ("CV-QKD", "ThorlabsPDB", "High")
]

RESULTS_FILE = "result_comparison.json"

def main():
    total_cases = len(TOPOLOGIES) * len(CONFIGS)
    print(f"🚀 Starting Full Evaluation: {len(TOPOLOGIES)} Maps x {len(CONFIGS)} Configs = {total_cases} Cases")
    
    # Load existing results if any
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                all_results = json.load(f)
        except:
            print("⚠️ Could not load existing results. Starting fresh.")
            all_results = []
    else:
        all_results = []
        
    start_time = time.time()
    
    case_idx = 0
    for topo in TOPOLOGIES:
        for prot, det, traf in CONFIGS:
            case_idx += 1
            print(f"\n[{case_idx}/{total_cases}] Checking {topo}-{prot}-{det}-{traf}...")
            
            # Check if already run
            already_run = False
            for r in all_results:
                c = r.get('config', {})
                if c.get('topology') == topo and c.get('protocol') == prot and c.get('detector') == det and c.get('traffic') == traf:
                    # Optional: Check if result is valid
                    # if r.get('bypass_result', {}).get('power', float('inf')) < float('inf'):
                    print(f"⏩ Skipping {topo}-{prot}-{det}-{traf} (Already done)")
                    already_run = True
                    break
            
            if already_run:
                continue
                
            try:
                # Run Experiment
                # run_experiment 内部会处理并行、训练和保存
                result = run_experiment(topo, prot, det, traf)
                
                # [Robust Save] Re-load current results to avoid overwriting manual changes
                if os.path.exists(RESULTS_FILE):
                    try:
                        with open(RESULTS_FILE, 'r') as f:
                            current_results = json.load(f)
                    except:
                        current_results = [] # Fallback if file is corrupted
                else:
                    current_results = []
                
                # Append new result
                current_results.append(result)
                
                # Atomic Write (using a temp file and rename would be even better, but this is okay)
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(current_results, f, indent=4)
                
                # Backup Models to results folder
                import shutil
                model_src_bypass = f"models/gnn_best_{topo}_{prot}_{det}_{traf}_True.pth"
                if os.path.exists(model_src_bypass):
                    shutil.copy2(model_src_bypass, f"results/model_{topo}_{prot}_{det}_{traf}_Bypass.pth")
                    
                model_src_nobypass = f"models/gnn_best_{topo}_{prot}_{det}_{traf}_False.pth"
                if os.path.exists(model_src_nobypass):
                    shutil.copy2(model_src_nobypass, f"results/model_{topo}_{prot}_{det}_{traf}_NoBypass.pth")
                    
                print(f"✅ Finished {topo}-{prot}-{det}-{traf}. Saved to {RESULTS_FILE} and backed up models.")
                
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user. Saving progress...")
                sys.exit(0)
            except Exception as e:
                print(f"❌ Failed {topo}-{prot}-{det}-{traf}: {e}")
                import traceback
                traceback.print_exc()
                # 继续下一个 case，不要退出
                
    total_time = time.time() - start_time
    print(f"🎉 All Done! Total Time: {total_time/3600:.2f} hours")

if __name__ == "__main__":
    main()
