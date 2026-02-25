import os
# 在导入任何其他库之前设置环境变量
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
import time
# 忽略 Gym 废弃警告和 Matplotlib 缺失警告
warnings.filterwarnings("ignore", category=UserWarning, module="cma")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
# 对于 gym 的特定废弃消息，可能需要更通用的过滤
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import torch
import torch.nn as nn
import numpy as np
import cma
import time
import os
import json
import gc
import random
import multiprocessing
import traceback

# 环境变量设置
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from qkd_env import QKDEnv
from rl_models import QKDGraphNet
from utils.traffic_generater import gen_traffic_matrix
from concurrent.futures import ProcessPoolExecutor

# === Worker 进程内的全局变量 (进程隔离) ===
_WORKER_ENV = None
_WORKER_MODEL = None

def worker_initializer(map_name, protocol, detector, traffic_mid, wavelength_list, request_list, hidden_dim):
    """
    Worker 进程初始化函数。只在进程启动时执行一次。
    """
    try:
        # [Performance] 强制限制每个 Worker 的线程数，防止 CPU 过载
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        global _WORKER_ENV, _WORKER_MODEL
        
        # 1. 禁用警告
        import warnings
        warnings.filterwarnings("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
        
        # 2. 初始化环境 (默认 is_bypass=False，会在每次 evaluate 时动态修改)
        _WORKER_ENV = QKDEnv(
            map_name=map_name,
            protocol=protocol,
            detector=detector,
            traffic_mid=traffic_mid,
            wavelength_list=wavelength_list,
            request_list=request_list,
            is_bypass=False 
        )
        
        # 3. 初始化模型 (CPU)
        _WORKER_MODEL = QKDGraphNet(actual_nodes=_WORKER_ENV.num_nodes, is_bypass=False, hidden_dim=hidden_dim).to("cpu")
        # 设置为 eval 模式，因为我们不需要梯度
        _WORKER_MODEL.eval()
        
        # [Debug] 确认 Worker 启动成功
        # with open(f"logs/worker_{os.getpid()}_ok.txt", "w") as f: f.write("OK")
        
    except Exception as e:
        import traceback
        import os
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/worker_{os.getpid()}_error.txt", "w") as f:
            f.write(traceback.format_exc())
        raise e

# === 独立的 Worker 函数 ===
def evaluate_worker(args):
    """
    并行评估 Worker。直接复用全局变量。
    Args:
        args: tuple (vector, is_bypass)
    """
    global _WORKER_ENV, _WORKER_MODEL
    
    # [Robustness] 增加更全面的异常捕获，防止 Worker 崩溃
    try:
        vector, is_bypass = args
        
        # 1. 动态更新配置 (防止状态污染)
        _WORKER_ENV.is_bypass = is_bypass
        _WORKER_MODEL.is_bypass = is_bypass # 如果模型内部用到了这个标志
        
        # 2. 加载参数
        # vector 是 numpy array，转为 tensor
        curr_idx = 0
        for param in _WORKER_MODEL.parameters():
            size = param.numel()
            # 这种写法比 named_parameters 更快，且无需 name 匹配
            new_param = torch.from_numpy(vector[curr_idx:curr_idx+size]).view(param.shape).float()
            param.data.copy_(new_param)
            curr_idx += size
            
        # 3. 运行评估循环
        # reset 会根据 _WORKER_ENV.is_bypass 强制更新 config.bypass
        state_matrices, context = _WORKER_ENV.reset()
        
        # 这里的 hidden_dim 可以从模型里取
        hidden_dim = _WORKER_MODEL.hidden_dim if hasattr(_WORKER_MODEL, 'hidden_dim') else 8
        h_state = torch.zeros(1, hidden_dim)
        last_action_t = None
        done = False
        
        step_count = 0
        while not done:
            with torch.no_grad():
                x_global_np, x_wl_np = state_matrices
                # 使用 from_numpy 避免内存拷贝 (Zero-copy)
                x_global_t = torch.from_numpy(x_global_np).float().unsqueeze(0)
                x_wl_t = torch.from_numpy(x_wl_np).float().unsqueeze(0)
                context_t = torch.from_numpy(context).float().unsqueeze(0)
                
                mu, _, h_next = _WORKER_MODEL(x_global_t, x_wl_t, context_t, last_action_t, h_state)
                h_state = h_next
                action_weights = mu.squeeze().numpy()
                last_action_t = mu.view(1, -1)
                
            next_state, reward, done, info = _WORKER_ENV.step(action_weights)
            state_matrices, context = next_state
            step_count += 1
            
        avg_power = info.get('avg_power', 10000.0)
        spec_occ = info.get('spec_occ', 1.0)
        
        # [New] Add detailed component power dict for analysis
        info['component_power'] = _WORKER_ENV.total_component_power
        
        # 综合适应度：平均功耗 + 频谱占用 + 热能风险 + 微观路径权重惩罚
        path_cost_sum = info.get('path_cost', 0.0)
        fitness = avg_power + spec_occ
        # fitness = avg_power + spec_occ + 0.00001 * path_cost_sum
        
        return fitness, info
        
    except Exception as e:
        import traceback
        import sys
        # 打印完整的 traceback 到 stderr，确保能被主进程捕获
        print(f"❌ Worker Critical Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 10000.0, {}
    except BaseException as e: # 捕获 KeyboardInterrupt, SystemExit 等更底层的异常
        import traceback
        import sys
        print(f"❌ Worker Fatal Crash: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 10000.0, {}

class OpenAIESOptimizer:
    """
    OpenAI Evolution Strategies (ES) Optimizer
    使用 Adam 优化器 + 伪梯度估计 (Search Gradients) + Rank Shaping。
    """
    def __init__(self, request_list, executor, bypass=True, map_name="Paris", traffic_mid="Low", protocol="BB84", detector="SNSPD", device="cuda", pop_size=64):
        self.bypass = bypass
        self.map_name = map_name
        self.protocol = protocol
        self.detector = detector
        self.traffic_mid = traffic_mid
        self.device = device
        self.wavelength_list = np.linspace(1530, 1565, 10).tolist()
        self.hidden_dim = 8
        self.executor = executor
        self.request_list = request_list
        
        # 1. 初始化模型
        self.env = QKDEnv(
            map_name=map_name, protocol=protocol, detector=detector, traffic_mid=traffic_mid,
            wavelength_list=self.wavelength_list, request_list=self.request_list, is_bypass=bypass
        )
        self.model = QKDGraphNet(
            num_global_features=7, num_wl_features=5, num_wavelengths=len(self.wavelength_list),
            actual_nodes=self.env.num_nodes, is_bypass=bypass, hidden_dim=self.hidden_dim
        ).to(device)
        
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.model_filename = f"gnn_best_{map_name}_{protocol}_{detector}_{traffic_mid}_bypass_{bypass}.pth"
        
        print(f"🚀 OpenAI-ES Optimizer Initialized. Params: {self.total_params}")
        
        # 2. ES 参数
        self.pop_size = 128 # [Tuning] 增大种群以增强探索
        self.sigma = 0.15   # [Tuning] 增大初始噪声 (0.1 -> 0.15)
        self.lr = 0.02      # 学习率 (Adam)
        # [Optimization] 减小 Weight Decay (0.005 -> 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.001)
        
        # 状态记录
        self.best_fitness_found = float('inf')
        self.best_pure_power_found = float('inf')
        self.prev_best_fitness = float('inf') # [Adaptive Sigma] 记录上一代的最佳
        self.best_metrics = {}
        self.generation = 0
        self.current_center_params = None
        self.stagnation_counter = 0 # [Restart] 停滞计数器
        
        # 尝试热启动
        model_path = os.path.join("models", self.model_filename)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
                print(f"📂 [{bypass}] Loaded warm start model: {self.model_filename}")
            except: pass
            
        # 日志
        self.log_filename = f"log_{map_name}_{protocol}_{detector}_{traffic_mid}_Bypass_{bypass}.txt"
        with open(self.log_filename, "w") as f:
            f.write(f"--- OpenAI-ES Training Log for Bypass={bypass} ---\n")
        self.log_file = open(self.log_filename, "a")

    def get_flat_params(self):
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.model.parameters()])

    def set_flat_params(self, flat_params):
        curr_idx = 0
        for param in self.model.parameters():
            size = param.numel()
            new_param = torch.from_numpy(flat_params[curr_idx:curr_idx+size]).view(param.shape).float().to(self.device)
            param.data.copy_(new_param)
            curr_idx += size

    def step(self):
        """执行一代 OpenAI-ES 训练"""
        center_params = self.get_flat_params()
        self.current_center_params = center_params
        
        # 1. 生成噪声 (Antithetic Sampling)
        half_pop = self.pop_size // 2
        noise = np.random.randn(half_pop, self.total_params)
        
        # 2. 准备评估参数
        eval_params = []
        for i in range(half_pop):
            eval_params.append(center_params + self.sigma * noise[i])
            eval_params.append(center_params - self.sigma * noise[i])
            
        # 3. 并行评估
        args_list = [(x, self.bypass) for x in eval_params]
        start_time = time.time()
        
        fitnesses = []
        infos = []
        try:
            results = list(self.executor.map(evaluate_worker, args_list))
            for fit, info in results:
                fitnesses.append(fit)
                infos.append(info)
        except Exception as e:
            print(f"❌ Parallel Error: {e}")
            return False
            
        duration = time.time() - start_time
        fitnesses = np.array(fitnesses)
        
        # 4. 记录最佳结果
        min_idx = np.argmin(fitnesses)
        if fitnesses[min_idx] < self.best_fitness_found:
            self.best_fitness_found = fitnesses[min_idx]
            self.best_pure_power_found = infos[min_idx].get('avg_power', float('inf'))
            self.best_metrics = infos[min_idx]
            self.save_model(eval_params[min_idx])
            
        # 5. 梯度估计 (Rank Transformation)
        ranks = np.zeros_like(fitnesses)
        ranks[fitnesses.argsort()] = np.arange(len(fitnesses))
        # Utility: Rank 0 (Best) -> 0.5, Rank N-1 (Worst) -> -0.5
        utilities = (len(fitnesses) - 1 - ranks) / (len(fitnesses) - 1) - 0.5
        utilities = (utilities - utilities.mean()) / (utilities.std() + 1e-8)
        
        grad = np.zeros(self.total_params)
        for i in range(half_pop):
            u_pos = utilities[2*i]
            u_neg = utilities[2*i+1]
            grad += noise[i] * (u_pos - u_neg)
            
        grad /= (half_pop * self.sigma)
        
        # 6. Adam 更新
        self.set_flat_params(self.current_center_params)
        self.optimizer.zero_grad(set_to_none=True)
        
        curr_idx = 0
        for param in self.model.parameters():
            size = param.numel()
            g = torch.from_numpy(grad[curr_idx:curr_idx+size]).view(param.shape).float().to(self.device)
            param.grad = -g # Adam Descent towards better utility
            curr_idx += size
            
        self.optimizer.step()
        
        # 7. 日志
        if self.generation % 1 == 0:
            avg_power = infos[min_idx].get('avg_power', 0.0)
            spec_occ = infos[min_idx].get('spec_occ', 0.0)
            fit_std = np.std(fitnesses) # 替换 Unique Count 为 Fitness Std
            
            log_str = (f"Gen {self.generation} (ES) | Pop: {self.pop_size} | Sigma: {self.sigma:.3f} | Time: {duration:.2f}s | "
                       f"Cur: {avg_power:.2f}W (S:{spec_occ:.2%}) | Std: {fit_std:.2f} | Best: {self.best_pure_power_found:.2f}W")
            print(f"[{'Bypass' if self.bypass else 'NoBypass'}] {log_str}")
            #self.log_file.write(log_str + "\n")
            #self.log_file.flush()
            
        # 8. [Adaptive Sigma] 动态调整噪声幅度
        current_best_fit = fitnesses[min_idx]
        
        if self.prev_best_fitness == float('inf'):
            # 第一代，跳过调整，保持初始 Sigma
            pass
        elif current_best_fit < self.prev_best_fitness:
            # 进步了 -> 根据进步幅度自适应收敛
            # 进步越大，收敛越快 (Sigma *= ratio)
            ratio = current_best_fit / (self.prev_best_fitness + 1e-8)
            # 保护性 Clip，防止 ratio 异常
            ratio = np.clip(ratio, 0.5, 0.99) 
            self.sigma *= ratio
        else:
            # 停滞 -> 膨胀
            self.sigma *= 1.02
            
        # [Sigma Clip] 限制最大噪声幅度 (0.10 - 0.25)
        self.sigma = np.clip(self.sigma, 0.1, 0.25)
        self.prev_best_fitness = self.best_fitness_found # 更新历史最佳基准
        
        # [Restart Mechanism] 如果 Sigma 长期顶在上限 (0.25)，说明陷入深坑，强制重启
        if self.sigma >= 0.248:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            
        if self.stagnation_counter >= 10:
            print(f"⚠️ [{self.generation}] Stagnation detected! Restarting with large perturbation...")
            
            # 1. 参数大跳跃 (Jump)
            current_params = self.get_flat_params()
            # 扰动幅度 0.5 (对于权重来说已经很大了)
            perturbation = np.random.randn(self.total_params) * 0.5
            self.set_flat_params(current_params + perturbation)
            
            # 2. 重置状态
            self.sigma = 0.15
            self.stagnation_counter = 0
            # 重置 Adam 动量 (保留 Weight Decay 设置)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.001)
            
        self.generation += 1
        return True

    def save_model(self, best_params):
        self.set_flat_params(best_params)
        torch.save(self.model.state_dict(), f"models/{self.model_filename}")
        self.set_flat_params(self.current_center_params) 
        
    def get_best_result(self):
        return self.best_metrics
    
    def load_from_optimizer(self, other_opt):
        print(f"🔄 [{self.bypass}] Transferring weights...")
        try:
            fname = os.path.join("models", other_opt.model_filename)
            if os.path.exists(fname):
                self.model.load_state_dict(torch.load(fname))
            else:
                pass
        except: pass
        
        # 重置 Adam (带 Weight Decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.005)
        print(f"✅ Transfer complete. Adam reset.")

    def close(self):
        if self.log_file: self.log_file.close()

class CMAESOptimizer:
    def __init__(self, request_list, executor, bypass=True, map_name="Paris", traffic_mid="Low", protocol="BB84", detector="SNSPD", device="cuda", pop_size=64):
        self.bypass = bypass
        self.map_name = map_name
        self.protocol = protocol
        self.detector = detector
        self.traffic_mid = traffic_mid
        self.device = device
        self.wavelength_list = np.linspace(1530, 1565, 10).tolist()
        self.base_pop_size = pop_size # 记录基础种群大小
        self.hidden_dim = 8
        self.executor = executor
        
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.request_list = request_list
        print(f"✅ [{bypass}] Initialized with external request list (Size: {len(self.request_list)})")
        
        self.env = QKDEnv(
            map_name=map_name,
            protocol=protocol,
            detector=detector,
            traffic_mid=traffic_mid,
            wavelength_list=self.wavelength_list,
            request_list=self.request_list,
            is_bypass=bypass
        )
        
        self.model = QKDGraphNet(actual_nodes=self.env.num_nodes, is_bypass=bypass, hidden_dim=self.hidden_dim).to(device)
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.model_filename = f"gnn_best_{map_name}_{protocol}_{detector}_{traffic_mid}_bypass_{bypass}.pth"
        
        print(f"🚀 GNN-CMA-ES Optimizer Initialized. Total Parameters: {self.total_params}")
        print(f"   Hidden Dim: {self.hidden_dim} | GRU: Enabled | Structure: Spatial-Preserved GNN")
        
        # === BIPOP 状态追踪 ===
        self.restart_count = 0
        self.last_large_pop = pop_size
        self.bipop_mode = "large" # 当前模式: 'large' (IPOP) 或 'small' (Local)
        
        # 初始化 CMA-ES 参数
        initial_params = np.concatenate([p.data.cpu().numpy().flatten() for p in self.model.parameters()])
        self.opts = {
            'popsize': pop_size, 
            'verb_disp': 0,
            'verb_log': 0,
            'tolfun': 1e-12
        }
        self.es = cma.CMAEvolutionStrategy(initial_params, 0.5, self.opts)
        
        # 状态记录
        self.best_fitness_found = float('inf')
        self.best_pure_power_found = float('inf')
        self.best_metrics = {}
        self.eval_count = 0
        self.generation = 0
        self.best_solution_vector = initial_params # 记录全局最佳参数用于重启
        
        self.log_filename = f"log_{map_name}_{protocol}_{detector}_{traffic_mid}_Bypass_{bypass}.txt"
        with open(self.log_filename, "w") as f:
            f.write(f"--- BIPOP-CMA-ES Training Log for Bypass={bypass} ---\n")
        self.log_file = open(self.log_filename, "a")
        
        # 尝试热启动
        model_path = os.path.join("models", self.model_filename)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
                print(f"📂 [{bypass}] Loaded warm start model: {self.model_filename}")
                new_params = np.concatenate([p.data.cpu().numpy().flatten() for p in self.model.parameters()])
                self.es = cma.CMAEvolutionStrategy(new_params, 0.5, self.opts)
                self.best_solution_vector = new_params
            except:
                pass

    def vector_to_model(self, vector):
        state_dict = self.model.state_dict()
        curr_idx = 0
        for name, param in self.model.named_parameters():
            size = param.numel()
            new_param = torch.from_numpy(vector[curr_idx:curr_idx+size]).view(param.shape).float().to(self.device)
            param.data.copy_(new_param)
            curr_idx += size
            
    # evaluate 函数保持不变 ...
    def evaluate(self, vector):
        self.vector_to_model(vector)
        self.model.eval()
        
        state_matrices, context = self.env.reset()
        h_state = torch.zeros(1, 8).to(self.device)
        last_action_t = None
        done = False
        
        while not done:
            with torch.no_grad():
                x_global_np, x_wl_np = state_matrices
                x_global_t = torch.FloatTensor(x_global_np).unsqueeze(0).to(self.device)
                x_wl_t = torch.FloatTensor(x_wl_np).unsqueeze(0).to(self.device)
                context_t = torch.FloatTensor(context).unsqueeze(0).to(self.device)
                
                mu, _, h_next = self.model(x_global_t, x_wl_t, context_t, last_action_t, h_state)
                h_state = h_next
                action_weights = mu.squeeze().cpu().numpy()
                last_action_t = mu.view(1, -1)
                
            next_state, reward, done, info = self.env.step(action_weights)
            state_matrices, context = next_state
            
        avg_power = info.get('avg_power', 10000.0)
        spec_occ = info.get('spec_occ', 1.0)
        fitness = avg_power + spec_occ
        self.last_info = info
        return fitness

    def step(self):
        """执行一代训练 (支持 BIPOP 重启)"""
        
        # === BIPOP 重启逻辑 ===
        if self.es.stop():
            self.restart_count += 1
            stop_reason = self.es.stop()
            print(f"🛑 [{self.bypass}] Convergence detected: {stop_reason}. Triggering BIPOP Restart #{self.restart_count}")
            
            # 切换策略：如果上次是小种群(Local)，这次就大种群(Global)，反之亦然
            # 注意：初始是 large，第一次重启通常尝试 small 
            if self.bipop_mode == "large":
                # 切换到小种群模式 (Local Search)
                self.bipop_mode = "small"
                new_popsize = self.base_pop_size
                new_sigma = 0.2 # 小步长精细搜索
                print(f"🔄 Restart Mode: SMALL (Local Search) | Pop: {new_popsize} | Sigma: {new_sigma}")
            else:
                # 切换到大种群模式 (Global Search IPOP)
                self.bipop_mode = "large"
                self.last_large_pop *= 2 # 种群翻倍
                new_popsize = self.last_large_pop
                new_sigma = 0.5 # 大步长
                print(f"🔄 Restart Mode: LARGE (Global Search) | Pop: {new_popsize} | Sigma: {new_sigma}")
            
            # 使用历史最佳解作为新起点
            best_param_mean = self.best_solution_vector
            
            # 更新配置
            new_opts = self.opts.copy()
            new_opts['popsize'] = new_popsize
            new_opts['seed'] = np.random.randint(100000)
            
            # 重启 ES 实例
            self.es = cma.CMAEvolutionStrategy(best_param_mean, new_sigma, new_opts)
        
        # === 正常的 ask/tell 流程 ===
        solutions = self.es.ask()
        
        args_list = [(x, self.bypass) for x in solutions]
        
        fitnesses = []
        infos = []
        
        start_time = time.time()
        
        try:
            results = list(self.executor.map(evaluate_worker, args_list))
            for fit, info in results:
                fitnesses.append(fit)
                infos.append(info)
            duration = time.time() - start_time
                
            best_idx = np.argmin(fitnesses)
            self.last_info = infos[best_idx]
            
            # 更新全局最佳解 (用于重启)
            current_best_fit = fitnesses[best_idx]
            if current_best_fit < self.best_fitness_found:
                self.best_fitness_found = current_best_fit
                self.best_solution_vector = solutions[best_idx] # 保存最佳参数向量
                self.save_model_parallel(solutions[best_idx], self.last_info)
            
            # 日志
            if self.generation % 1 == 0:
                info = self.last_info
                avg_power = info.get('avg_power', 0.0)
                spec_occ = info.get('spec_occ', 0.0)
                hist_best_p = self.best_pure_power_found if self.best_pure_power_found != float('inf') else 0.0
                
                log_str = (f"Gen {self.generation} (R{self.restart_count}-{self.bipop_mode[0].upper()}) | "
                           f"Pop: {self.es.popsize} | Time: {duration:.2f}s | "
                           f"Cur: {avg_power:.2f}W (Spec:{spec_occ:.2%}) | HistBest: {hist_best_p:.2f}W")
                print(f"[{'Bypass' if self.bypass else 'NoBypass'}] {log_str}")
                self.log_file.write(log_str + "\n")
                self.log_file.flush()
                
        except Exception as e:
            print(f"❌ Parallel Execution Error: {e}")
            return False
        
        self.es.tell(solutions, fitnesses)
        self.generation += 1
        
        return True

    def save_model_parallel(self, xbest, info):
        self.best_metrics = info
        self.best_pure_power_found = self.best_metrics.get('avg_power', float('inf'))
        self.vector_to_model(xbest)
        model_path = os.path.join("models", self.model_filename)
        tmp_path = model_path + ".tmp"
        torch.save(self.model.state_dict(), tmp_path)
        os.replace(tmp_path, model_path)
        print(f"✨ [{self.bypass}] New Best: {self.best_pure_power_found:.2f}W")

    def get_best_result(self):
        return {
            "best_fitness": self.best_fitness_found,
            "avg_power": self.best_pure_power_found,
            "spec_occ": self.best_metrics.get('spec_occ', 0.0),
            "components": {
                "source": self.best_metrics.get('source_p', 0.0),
                "detector": self.best_metrics.get('detector_p', 0.0),
                "ice_box": self.best_metrics.get('ice_box_p', 0.0),
                "other": self.best_metrics.get('other_p', 0.0)
            }
        }
        
    def load_from_optimizer(self, other_opt):
        print(f"🔄 [{self.bypass}] Transferring weights from {other_opt.bypass}...")
        best_model_path = os.path.join("models", other_opt.model_filename)
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            self.model.load_state_dict(other_opt.model.state_dict())
        
        new_params = np.concatenate([p.data.cpu().numpy().flatten() for p in self.model.parameters()])
        # 重置重启状态
        self.restart_count = 0
        self.last_large_pop = self.base_pop_size
        self.bipop_mode = "large"
        self.best_solution_vector = new_params
        
        self.es = cma.CMAEvolutionStrategy(new_params, 0.5, self.opts)
        print(f"✅ Transfer complete. CMA-ES reset.")

    def close(self):
        if self.log_file:
            self.log_file.close()

def run_experiment(map_name, protocol, detector, traffic_mid):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"🔥 Starting Experiment: {map_name} | {protocol} | {detector} | {traffic_mid} on {device}")
    
    # 临时文件，避免冲突
    json_filename = f"results/temp_{map_name}_{protocol}_{detector}_{traffic_mid}.json"
    os.makedirs("results", exist_ok=True)
    
    import glob
    for f in glob.glob(f"models/gnn_best_{map_name}_{protocol}_{detector}_{traffic_mid}_*.pth"):
        try: os.remove(f) 
        except: pass
            
    random.seed(42)
    np.random.seed(42)
    wavelength_list = np.linspace(1530, 1565, 10).tolist()
    global_request_list = gen_traffic_matrix(traffic_mid, map_name, wavelength_list, protocol, detector)

    # increase_traffic_value = 20000000
    # for request_id in range(len(global_request_list)):
    #     old_tuple = global_request_list[request_id]
    #     new_tuple = old_tuple[:-1] + (old_tuple[-1] + increase_traffic_value,)
    #     global_request_list[request_id] = new_tuple

    print(f"🌍 Generated Global Request List (Size: {len(global_request_list)})")
    
    hidden_dim = 8 # 全局配置改回 8
    initargs = (map_name, protocol, detector, traffic_mid, wavelength_list, global_request_list, hidden_dim)
    # 增加 max_workers 以应对可能翻倍的种群
    # 使用 Context Manager 管理 ProcessPoolExecutor
    # [Performance] 使用所有可用核心
    import multiprocessing
    # num_workers = multiprocessing.cpu_count()
    # 如果核心数过多，限制一下以免内存爆炸 (e.g. 64核)
    # [Performance] 恢复多进程并行，强制使用 fork 模式
    import multiprocessing
    
    # 尝试获取 fork 上下文 (Linux/Unix 默认，但在某些配置下可能被覆盖)
    try:
        mp_context = multiprocessing.get_context("fork")
    except ValueError:
        # 如果不支持 fork (e.g. Windows)，回退到默认
        mp_context = None
        
    num_workers = multiprocessing.cpu_count()
    # 限制最大 Worker 数
    # [Performance Tuning] 线程数已限制为 1，现在可以全核跑了
    # num_workers = min(num_workers, 32) 
    um_workers = num_workers
    print(f"🚀 Launching ProcessPoolExecutor with {num_workers} workers (Context: {mp_context})")
    
    # 这里的 if-else 是为了保留 SyncExecutor 作为一个 fallback 选项，但我们现在要切回并行
    if True: 
        # Python 3.7+ 支持 mp_context
        executor_cm = ProcessPoolExecutor(
            max_workers=num_workers, 
            initializer=worker_initializer, 
            initargs=initargs,
            mp_context=mp_context
        )
    else:
        # Fake Executor for debugging
        class SyncExecutor:
            def __enter__(self): 
                # Manually initialize worker state in main process
                print("🔧 Initializing Worker State in Main Process...")
                worker_initializer(*initargs)
                return self
            def __exit__(self, exc_type, exc_val, exc_tb): pass
            def map(self, func, iterable):
                return map(func, iterable)
            def shutdown(self, wait=True): pass
            
        executor_cm = SyncExecutor()
        print("⚠️ Running in SYNC mode (No Multiprocessing)")

    with executor_cm as shared_executor:
    
        # 使用 OpenAI-ES (防止局部最优)
        opt_bypass = OpenAIESOptimizer(global_request_list, shared_executor, bypass=True, map_name=map_name, traffic_mid=traffic_mid, protocol=protocol, detector=detector, device=device)
        opt_nobypass = OpenAIESOptimizer(global_request_list, shared_executor, bypass=False, map_name=map_name, traffic_mid=traffic_mid, protocol=protocol, detector=detector, device=device)
        
        phase1_gens = 50
        phase2_gens = 100 
        
        # Phase 1
        print(f"\n=== Phase 1: Pre-training NoBypass ({phase1_gens} gens) ===")
        for gen in range(1, phase1_gens + 1):
            if not opt_nobypass.step(): break
            
            report = {
                "generation": gen,
                "bypass": opt_bypass.get_best_result(),
                "nobypass": opt_nobypass.get_best_result(),
                "status": "phase1_nobypass"
            }
            with open(json_filename, "w") as f: json.dump(report, f, indent=4)
        
        # Phase 2
        print(f"\n=== Phase 2: Transferring Knowledge & Training Bypass ===")
        opt_bypass.load_from_optimizer(opt_nobypass)
        
        max_it = 300
        total_gens = phase1_gens + phase2_gens
        
        final_status = "max_gens_reached"
        
        for gen in range(phase1_gens + 1, max_it + 1):
            if not opt_bypass.step(): break
            
            report = {
                "generation": gen,
                "bypass": opt_bypass.get_best_result(),
                "nobypass": opt_nobypass.get_best_result(),
                "status": "phase2_bypass"
            }
            with open(json_filename, "w") as f: json.dump(report, f, indent=4)
            
            # 比较逻辑
            p_bypass = opt_bypass.best_pure_power_found
            p_nobypass = opt_nobypass.best_pure_power_found
            s_bypass = opt_bypass.best_metrics.get('spec_occ', 0.0)
            s_nobypass = opt_nobypass.best_metrics.get('spec_occ', 0.0)

            # 停止条件判断
            if p_bypass < float('inf') and p_nobypass < float('inf'):
                power_win = p_bypass < p_nobypass
                spec_tradeoff = s_bypass > s_nobypass
                if gen > total_gens:
                    if power_win and spec_tradeoff:
                        print(f"✅ Bypass Wins with Trade-off! Stopping.")
                        final_status = "bypass_wins_tradeoff"
                        break
                    # 其他情况继续跑
        
        # 获取最终结果
        result = {
            "config": {
                "topology": map_name,
                "protocol": protocol,
                "detector": detector,
                "traffic": traffic_mid
            },
            "bypass_result": opt_bypass.get_best_result(),
            "nobypass_result": opt_nobypass.get_best_result(),
            "status": final_status
        }
        
        opt_nobypass.close()
        opt_bypass.close()
        
    # Clean up temp file
    try: os.remove(json_filename)
    except: pass
        
    return result

if __name__ == "__main__":
    # 默认单次运行 (用于调试)
    run_experiment("Tokyo", "BB84", "APD", "Low")
