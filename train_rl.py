import torch
import numpy as np
from qkd_env import QKDEnv
from rl_models import PolicyGradientAgent
import config
import os
from tqdm import tqdm
import time

from utils.traffic_generater import gen_traffic_matrix

def train(bypass=True):
    # Hyperparameters
    num_episodes = 500 if bypass else 100 
    batch_size = 5 # 每 5 个 episode 更新一次 (改动 A)
    lr = 1e-4 
    map_name = "Paris"
    protocol = "BB84"
    detector = "SNSPD"
    traffic_mid = "Low" # Respecting string input
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    strategy = "Bypass" if bypass else "NoBypass"
    print(f"\n🚀 Starting RL training for {strategy}...")
    
    import config
    config.bypass = bypass
    config.protocol = protocol
    config.detector = detector
    
    # Respect original input: Wavelength list from main.py
    wavelength_list = np.linspace(1530, 1565, 10).tolist()
    
    # Generate original request list
    request_list = gen_traffic_matrix(traffic_mid, map_name, wavelength_list, protocol, detector)
    
    # Initialize Environment with original inputs
    env = QKDEnv(
        map_name=map_name, 
        protocol=protocol, 
        detector=detector, 
        traffic_mid=traffic_mid,
        wavelength_list=wavelength_list,
        request_list=request_list
    )
    print(f"Network nodes: {env.num_nodes}")
    
    # 对齐实际节点数初始化 Agent
    # 为不同场景分配独立的架构和学习率
    lr_actual = lr if bypass else lr * 2.0 # NoBypass 简单，可以快一点
    agent = PolicyGradientAgent(num_nodes=env.num_nodes, lr=lr_actual, is_bypass=bypass, device=device)
    
    # Create log directory
    os.makedirs("logs", exist_ok=True)
    log_file = open(f"logs/rl_{strategy}.log", "w")
    
    best_power = float('inf')
    best_state_dict = None 
    last_loss = 0.0 # 记录最近一次更新的 loss，避免积攒期显示 0
    
    # 探索退火
    start_entropy = 0.01
    end_entropy = 0.001
    
    for episode in range(num_episodes):
        start_time = time.time()
        state_matrices, context = env.reset()
        agent.reset_episode() 
        done = False
        episode_reward = 0
        
        # 线性退火计算当前的 entropy 权重
        entropy_coef = max(end_entropy, start_entropy - (start_entropy - end_entropy) * (episode / num_episodes))
        
        while not done:
            action_weights = agent.select_action(state_matrices, context, train=True)
            next_state, reward, done, info = env.step(action_weights)
            episode_reward += reward
            state_matrices, context = next_state
            
        # 结束 Episode，记录数据
        agent.end_episode(episode_reward)
        
        # 累计到 batch_size 后执行更新
        if (episode + 1) % batch_size == 0:
            last_loss = agent.update(entropy_coef=entropy_coef)
            
        duration = time.time() - start_time
        avg_power = info.get('avg_power', 0)
        spec_occ = info.get('spec_occ', 0)
        
        status_str = f"[{strategy}] Ep {episode+1} | Reward: {episode_reward:.2f} | Power: {avg_power:.2f}W | Spec: {spec_occ:.4f} | Loss: {last_loss:.4f} | Ent: {entropy_coef:.4f} | {duration:.1f}s"
        print(status_str)
        log_file.write(status_str + "\n")
        log_file.flush()
        
        # 精英策略保存与定期回滚 (改动 C)
        if avg_power < best_power and avg_power > 0:
            best_power = avg_power
            best_state_dict = agent.model.state_dict().copy()
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/qkd_rl_{strategy}_best.pth")
            
        # 每 50 个 episode，如果当前性能退化严重，考虑回滚到精英策略
        if (episode + 1) % 50 == 0 and best_state_dict is not None:
            # 这里可以用一个简单的概率或者阈值判断是否回滚
            # 为了稳定，我们选择在性能没能突破时，小概率(20%)回滚以重新探索
            if avg_power > best_power * 1.1 and np.random.random() < 0.2:
                print(f"🔄 Rolling back to best policy (Power: {best_power:.2f}W)")
                agent.model.load_state_dict(best_state_dict)

    log_file.close()
    return best_power

if __name__ == "__main__":
    print("Main script started")
    # 跳过 NoBypass，直接攻克最具挑战性的 Bypass 模式
    # p_nobypass = train(bypass=False)
    p_bypass = train(bypass=True)
    
    print("\n" + "="*40)
    print(f"Final Best Power (Bypass): {p_bypass:.2f}W")
    # print(f"Final Best Power (NoBypass): {p_nobypass:.2f}W")
    print("="*40)
