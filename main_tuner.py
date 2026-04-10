import json
import os
import numpy as np
from core.mapper import ExpertMapper
from core.traffic_model import PowerTrafficProjector
from sim_engine.bridge import RealSimulationBridge
# 导入我们刚刚创建的绘图模块
try:
    from visualize_thermal import plot_chiplet_thermal
except ImportError:
    print("[Warning] visualize_thermal.py not found. Thermal maps will not be generated.")
    plot_chiplet_thermal = None

class MoEAutoTuner:
    def __init__(self, hw_file="configs/hw_config.json", moe_file="configs/moe_params.json"):
        print("==================================================")
        print("[System] Initializing Multi-Physics Toolchain...")
        
        # 1. 加载真实的硬件与模型配置
        with open(hw_file, 'r') as f: self.hw_config = json.load(f)
        with open(moe_file, 'r') as f: self.moe_config = json.load(f)
        
        self.Ne = self.moe_config['architecture']['num_experts_total']
        self.Nc = self.hw_config['chiplet_arch']['num_chiplets']
        
        # 2. 实例化算法层 (core)
        self.mapper = ExpertMapper(self.Ne, self.Nc)
        self.projector = PowerTrafficProjector(self.hw_config, self.moe_config)
        
        # 3. 实例化执行层 (sim_engine) - 纯真实环境
        self.sim = RealSimulationBridge()

    def evaluate_policy(self, mapping_matrix):
        """单轮真实的物理闭环评估：PyTorch -> NoC -> Thermal"""
        
        # A. 软件层: 通过 Docker 真实计算 PyTorch 模型，获取专家逻辑负载 Ve
        # 这一步会触发容器内的矩阵运算
        Ve = self.sim.run_pytorchsim_docker(self.moe_config)
        
        # B. 核心数学层: 将映射策略与逻辑负载结合，计算物理芯片负载与功耗
        Vc, compute_power = self.projector.project_traffic_and_power(mapping_matrix, Ve)
        
        # C. 通信层: 将真实物理流量 Vc 灌入 BookSim2 二进制文件
        # 我们已经修正了 255 报错的兼容性逻辑
        latency, noc_power = self.sim.run_booksim_real(Vc)
        
        # D. 热力层: 结合计算功耗与通信功耗，调用 HotSpot 求解偏微分方程
        # 我们已经加入了 -f 指定地板图文件的逻辑
        t_max, _ = self.sim.run_hotspot_real(compute_power, noc_power)
        
        # E. 代价函数计算 (可以根据你的研究重点调整权重)
        # Cost = (峰值温度 * 10) + 网络延迟
        cost = (t_max * 10) + latency
        return cost, t_max, latency

    def optimize(self, epochs=5):
        print(f"\n[Optimizer] Starting Auto-Tuning Loop for {epochs} epochs...")
        best_cost = float('inf')
        best_mapping = None
        
        # 确保输出目录存在
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1:02d} ---")
            
            # 1. 生成映射策略 (这里你可以后续替换为遗传算法或强化学习)
            current_M = self.mapper.generate_mapping(strategy="random")
            
            # 2. 执行全物理链路深度评估
            try:
                cost, t_max, latency = self.evaluate_policy(current_M)
                
                print(f"> Result: T_max = {t_max:.2f}°C | Latency = {latency:.2f} cycles | Cost = {cost:.2f}")
                
                # 3. 如果发现更优策略，保存结果并绘制热力图
                if cost < best_cost:
                    best_cost = cost
                    best_mapping = current_M
                    print("  [*] Optimal Target Updated!")
                    
                    # 自动生成当前最优策略的热力图
                    if plot_chiplet_thermal:
                        img_name = f"outputs/best_thermal_epoch_{epoch+1}.png"
                        plot_chiplet_thermal(save_path=img_name)
                        print(f"  [*] Thermal snapshot saved as {img_name}")
            
            except Exception as e:
                print(f"  [!] Error during optimization at Epoch {epoch+1}: {e}")
                continue

        print("\n[System] Optimization Complete.")
        print(f"[Summary] Best Cost Found: {best_cost:.2f}")
        print("==================================================")

if __name__ == "__main__":
    # 李明，建议先跑 3-5 个 Epoch 测试全链路是否顺畅
    tuner = MoEAutoTuner()
    tuner.optimize(epochs=5)
