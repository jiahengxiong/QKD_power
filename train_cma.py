import torch
import torch.nn as nn
import numpy as np
import cma
import time
import os
import json
import gc

# 环境变量设置必须在导入 numpy/torch 之后尽快执行，或在最前面
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from qkd_env import QKDEnv
from rl_models import QKDGraphNet
from utils.traffic_generater import gen_traffic_matrix

class CMAESOptimizer:
    def __init__(self, bypass=True, map_name="Paris", traffic_mid="Low", protocol="BB84", detector="SNSPD", device="cuda"):
        self.bypass = bypass
        self.protocol = protocol
        self.detector = detector
        self.traffic_mid = traffic_mid
        self.device = device
        self.wavelength_list = np.linspace(1530, 1565, 10).tolist()
        
        # 核心修复：设置固定随机种子，确保每次生成的 request_list 完全一致
        # 这对于跨运行的 Warm Start 和公平比较至关重要
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.request_list = gen_traffic_matrix(traffic_mid, map_name, self.wavelength_list, protocol, detector)
        print(f"✅ Generated request list (Size: {len(self.request_list)}) with seed 42. Consistent across runs.")
        
        # 初始化环境
        self.env = QKDEnv(
            map_name=map_name,
            protocol=protocol,
            detector=detector,
            traffic_mid=traffic_mid,
            wavelength_list=self.wavelength_list,
            request_list=self.request_list
        )
        
        # 初始化 GNN 模型
        self.model = QKDGraphNet(actual_nodes=self.env.num_nodes, is_bypass=bypass, hidden_dim=8).to(device)
        self.param_shapes = [p.shape for p in self.model.parameters()]
        self.param_sizes = [p.numel() for p in self.model.parameters()]
        self.total_params = sum(self.param_sizes)
        
        # 独立模型保存路径 (增加 _GNN 后缀区分)
        self.model_filename = f"gnn_best_Paris_{protocol}_{detector}_{traffic_mid}_bypass_{bypass}.pth"
        
        print(f"🚀 GNN-CMA-ES Optimizer Initialized. Total Parameters: {self.total_params}")

    def vector_to_model(self, vector):
        """将一维向量还原回模型参数"""
        state_dict = self.model.state_dict()
        curr_idx = 0
        for name, param in self.model.named_parameters():
            size = param.numel()
            new_param = torch.from_numpy(vector[curr_idx:curr_idx+size]).view(param.shape).float().to(self.device)
            param.data.copy_(new_param)
            curr_idx += size

    def evaluate(self, vector):
        """评估一个参数向量的 Fitness (Total Avg Power + Occupied Spectrum)"""
        self.vector_to_model(vector)
        self.model.eval()
        
        state_matrices, context = self.env.reset()
        h_state = torch.zeros(1, 8).to(self.device)
        last_action_t = None
        done = False
        
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state_matrices).unsqueeze(0).to(self.device)
                context_t = torch.FloatTensor(context).unsqueeze(0).to(self.device)
                mu, _, h_next = self.model(state_t, context_t, last_action_t, h_state)
                h_state = h_next
                action_weights = mu.squeeze().cpu().numpy()
                last_action_t = mu.view(1, -1)
                
            next_state, reward, done, info = self.env.step(action_weights)
            state_matrices, context = next_state
            
        # 优化目标：Total Avg Power + Occupied Spectrum
        # 注意：Occupied Spectrum < 1，作为平滑项
        avg_power = info.get('avg_power', 10000.0)
        spec_occ = info.get('spec_occ', 1.0)
        fitness = avg_power + spec_occ
        
        # 记录详细信息，供保存时使用
        self.last_info = info
        
        return fitness

    def save_callback(self, es):
        """CMA-ES 每一代结束后的回调，用于保存模型和当前最优详细数据"""
        if es.result.fbest < self.best_power_found:
            self.best_power_found = es.result.fbest
            # 将最优向量还原到模型并保存
            self.vector_to_model(es.result.xbest)
            model_path = os.path.join("models", self.model_filename)
            tmp_path = model_path + ".tmp"
            torch.save(self.model.state_dict(), tmp_path)
            os.replace(tmp_path, model_path)
            
            # 保存当前最优的物理指标
            self.best_metrics = self.last_info
            
            print(f"✨ New Best Fitness: {self.best_power_found:.4f} (Power: {self.best_metrics['avg_power']:.2f}W) | Model Saved: {self.model_filename}", flush=True)

    def train(self, max_iter=100, pop_size=64):
        # 尝试从现有最优模型加载，进行“热启动”
        model_path = os.path.join("models", self.model_filename)
        self.best_metrics = {}
        
        if os.path.exists(model_path):
            print(f"📂 Found existing best model: {self.model_filename}. Loading for warm start...", flush=True)
            try:
                self.model.load_state_dict(torch.load(model_path))
            except:
                print(f"⚠️ Failed to load {self.model_filename}, starting from scratch.")
        
        # 初始均值为当前模型参数
        initial_params = np.concatenate([p.data.cpu().numpy().flatten() for p in self.model.parameters()])
        
        self.best_power_found = float('inf')
        
        print(f"🚀 Starting BIPOP-CMA-ES optimization...", flush=True)
        
        # 使用 cma.fmin2 直接调用 BIPOP-CMA-ES
        # bipop=True: 开启 BIPOP 重启策略
        # restarts=9: 允许最多 9 次重启（包含 IPOP 增加种群和小种群探索）
        opts = {
            'popsize': pop_size, 
            'maxiter': max_iter, 
            'verb_disp': 1,
            'tolfunhist': 0, 
            'tolfun': 1e-12
        }
        
        # 包装 evaluate 函数，确保健壮性
        self.eval_count = 0
        def objective(x):
            try:
                self.eval_count += 1
                fit = self.evaluate(x)
                
                # 显式触发垃圾回收，防止内存累积导致的潜在段错误
                if self.eval_count % 32 == 0:
                    gc.collect()
                
                if self.eval_count % 8 == 0:
                    print(f"  Eval {self.eval_count} | Power: {fit:.2f}W", flush=True)
                return float(fit) if np.isfinite(fit) else 10000.0
            except Exception as e:
                print(f"❌ Error in objective: {e}", flush=True)
                return 10000.0

        res = cma.fmin2(
            objective, 
            initial_params, 
            0.3, 
            opts,
            callback=self.save_callback,
            bipop=True,
            restarts=9
        )
        
        print(f"\n✅ Optimization Finished. Best Power: {res[1]:.2f}W")
        return res[1]

if __name__ == "__main__":
    import argparse
    import config
    os.makedirs("models", exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Run CMA-ES Optimization for a specific configuration")
    parser.add_argument("--bypass", type=str, default="True", help="Bypass mode (True/False)")
    parser.add_argument("--detector", type=str, default="SNSPD", help="Detector type (SNSPD/APD/ThorlabsPDB)")
    parser.add_argument("--traffic", type=str, default="Low", help="Traffic level (Low/Medium/High)")
    parser.add_argument("--protocol", type=str, default="BB84", help="Protocol (BB84/CV-QKD)")
    parser.add_argument("--max_iter", type=int, default=300, help="Max iterations")
    parser.add_argument("--pop_size", type=int, default=64, help="Population size")
    
    args = parser.parse_args()
    
    # 将参数转换为 bool
    is_bypass = args.bypass.lower() == "true"
    
    # 检测 CUDA 状态
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Config: {args.protocol}, {args.detector}, {args.traffic}, Bypass={is_bypass}")
    
    # 强制开启指定模式进行攻坚
    optimizer = CMAESOptimizer(
        bypass=is_bypass, 
        map_name="Paris", 
        traffic_mid=args.traffic, 
        protocol=args.protocol,
        detector=args.detector,
        device=device
    )

    optimizer.env.provided_request_list = optimizer.request_list
    
    best_p = optimizer.train(max_iter=args.max_iter, pop_size=args.pop_size)
    
    # 返回包含所有物理指标的字典
    result_data = {
        "protocol": args.protocol,
        "detector": args.detector,
        "traffic": args.traffic,
        "bypass": is_bypass,
        "best_fitness": best_p,
        "avg_power": optimizer.best_metrics.get('avg_power', 10000.0),
        "spec_occ": optimizer.best_metrics.get('spec_occ', 1.0),
        "source_p": optimizer.best_metrics.get('source_p', 0.0),
        "detector_p": optimizer.best_metrics.get('detector_p', 0.0),
        "other_p": optimizer.best_metrics.get('other_p', 0.0),
        "ice_box_p": optimizer.best_metrics.get('ice_box_p', 0.0)
    }
    
    # 保存结果到独立文件，供汇总脚本读取
    result_filename = f"results_Paris_{args.protocol}_{args.detector}_{args.traffic}_Bypass_{is_bypass}.json"
    with open(result_filename, "w") as f:
        json.dump(result_data, f)
    
    print("\n" + "="*40)
    print(f"Optimization Done! Results saved to {result_filename}")
    print(f"Final Power: {result_data['avg_power']:.2f}W | Spectrum: {result_data['spec_occ']:.4f}")
    print("="*40)
