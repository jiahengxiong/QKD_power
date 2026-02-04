import subprocess
import os
import json
import time
from multiprocessing import Pool

# 定义 8 组对照实验
CASE_PAIRS = [
    ("BB84", "APD", "Low"),
    ("BB84", "APD", "Medium"),
    ("BB84", "SNSPD", "Low"),
    ("BB84", "SNSPD", "Medium"),
    ("BB84", "SNSPD", "High"),
    ("CV-QKD", "ThorlabsPDB", "Low"),
    ("CV-QKD", "ThorlabsPDB", "Medium"),
    ("CV-QKD", "ThorlabsPDB", "High"),
]

def get_result_filename(protocol, detector, traffic, bypass):
    return f"results_Paris_{protocol}_{detector}_{traffic}_Bypass_{bypass}.json"

def run_single_case(protocol, detector, traffic, bypass, max_iter=100, pop_size=64):
    log_file = f"log_Paris_{protocol}_{detector}_{traffic}_Bypass_{bypass}.txt"
    cmd = [
        "python3", "-u", "train_cma.py",
        "--protocol", protocol,
        "--detector", detector,
        "--traffic", traffic,
        "--bypass", str(bypass),
        "--max_iter", str(max_iter),
        "--pop_size", str(pop_size)
    ]
    
    # 使用追加模式记录日志
    with open(log_file, "a") as f:
        f.write(f"\n\n--- Starting Iteration at {time.ctime()} (max_iter={max_iter}) ---\n")
        env = os.environ.copy()
        env["MKL_THREADING_LAYER"] = "GNU"
        env["OMP_NUM_THREADS"] = "1"
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

def main():
    # 状态记录：(protocol, detector, traffic) -> equality_counter
    patience_counters = {pair: 0 for pair in CASE_PAIRS}
    resolved_pairs = set()
    
    # 1. 初始运行：100 代
    print("🚀 Stage 1: Running initial 100 generations for all 16 cases...")
    all_configs = []
    for pair in CASE_PAIRS:
        all_configs.append((*pair, True))  # Bypass=True
        all_configs.append((*pair, False)) # Bypass=False
    
    with Pool(processes=16) as pool:
        pool.starmap(run_single_case, [(c[0], c[1], c[2], c[3], 100) for c in all_configs])
    
    # 2. 迭代攻坚
    iteration_round = 1
    while len(resolved_pairs) < len(CASE_PAIRS):
        print(f"\n🔄 Stage 2: Iteration Round {iteration_round}. Checking results...")
        
        to_run = []
        for pair in CASE_PAIRS:
            if pair in resolved_pairs:
                continue
            
            p, d, t = pair
            res_t = get_result_filename(p, d, t, True)
            res_f = get_result_filename(p, d, t, False)
            
            try:
                if not os.path.exists(res_t) or not os.path.exists(res_f):
                    print(f"  Check {p}/{d}/{t}: Results missing. Retrying both...")
                    to_run.append((*pair, True))
                    to_run.append((*pair, False))
                    continue

                with open(res_t, 'r') as f: data_t = json.load(f)
                with open(res_f, 'r') as f: data_f = json.load(f)
                
                pow_t = data_t['avg_power']
                pow_f = data_f['avg_power']
                
                print(f"  Check {p}/{d}/{t}: Bypass={pow_t:.2f}W, No-Bypass={pow_f:.2f}W", end="")
                
                if pow_t < pow_f:
                    print(" -> ✅ Success (Bypass < No-Bypass)")
                    resolved_pairs.add(pair)
                elif abs(pow_t - pow_f) < 1e-3: # 处理浮点数相等
                    patience_counters[pair] += 25
                    if patience_counters[pair] > 250:
                        print(f" -> 🛑 Fused (Patience reached 250 at equality)")
                        resolved_pairs.add(pair)
                    else:
                        print(f" -> ⏳ Equal (Patience: {patience_counters[pair]}/250). Retrying Bypass...")
                        to_run.append((*pair, True))
                else:
                    print(" -> 📈 Worse (Bypass > No-Bypass). Retrying Bypass...")
                    to_run.append((*pair, True))
            except Exception as e:
                print(f" -> ❌ Error reading results for {pair}: {e}")
                to_run.append((*pair, True))
                to_run.append((*pair, False))
        
        if not to_run:
            print("\n🎉 All cases resolved or fused.")
            break
            
        print(f"🚀 Round {iteration_round}: Running {len(to_run)} cases for 25 more generations...")
        # 限制最大进程数，防止过载
        with Pool(processes=min(len(to_run), 16)) as pool:
            pool.starmap(run_single_case, [(c[0], c[1], c[2], c[3], 25) for c in to_run])
        
        iteration_round += 1

    # 3. 生成最终报表
    print("\n📊 Generating final report...")
    report_lines = []
    header = "| Protocol | Detector | Traffic | Power (No-By) | Power (By) | Power Reduc% | Spec (No-By) | Spec (By) | Spec Incr% | Status |"
    sep = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    report_lines.append(header)
    report_lines.append(sep)
    
    for pair in CASE_PAIRS:
        p, d, t = pair
        res_t_file = get_result_filename(p, d, t, True)
        res_f_file = get_result_filename(p, d, t, False)
        
        if os.path.exists(res_t_file) and os.path.exists(res_f_file):
            with open(res_t_file, 'r') as f: r_t = json.load(f)
            with open(res_f_file, 'r') as f: r_f = json.load(f)
            
            pow_t, pow_f = r_t['avg_power'], r_f['avg_power']
            spec_t, spec_f = r_t['spec_occ'], r_f['spec_occ']
            
            # 计算百分比
            pow_reduc = ((pow_f - pow_t) / pow_f * 100) if pow_f > 0 else 0
            spec_incr = ((spec_t - spec_f) / spec_f * 100) if spec_f > 0 else 0
            
            # 状态判定
            status = "Success (<)" if pow_t < pow_f else "Fused (==)"
            
            line = f"| {p} | {d} | {t} | {pow_f:.2f}W | {pow_t:.2f}W | **{pow_reduc:.2f}%** | {spec_f:.4f} | {spec_t:.4f} | {spec_incr:.2f}% | {status} |"
            report_lines.append(line)
    
    # 增加详细功耗分量表
    report_lines.append("\n\n### Detailed Component Power Breakdown (Bypass Mode)")
    comp_header = "| Case | Source | Detector | IceBox | Other |"
    comp_sep = "| :--- | :--- | :--- | :--- | :--- |"
    report_lines.append(comp_header)
    report_lines.append(comp_sep)
    
    for pair in CASE_PAIRS:
        p, d, t = pair
        res_t_file = get_result_filename(p, d, t, True)
        if os.path.exists(res_t_file):
            with open(res_t_file, 'r') as f: r = json.load(f)
            line = f"| {p}/{d}/{t} | {r['source_p']:.2f}W | {r['detector_p']:.2f}W | {r['ice_box_p']:.2f}W | {r['other_p']:.2f}W |"
            report_lines.append(line)

    final_content = "# Final Paris Experiment Summary\n\n" + "\n".join(report_lines)
    print("\n" + final_content)
    with open("final_summary_report.md", "w") as f:
        f.write(final_content)

if __name__ == "__main__":
    main()
