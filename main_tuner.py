# ~/MoE_Auto_Tuner/main_tuner.py
import os
import json
import numpy as np
from datetime import datetime
import shutil

os.environ['MKL_THREADING_LAYER'] = 'GNU'
from core.mapper import ExpertMapper
from core.traffic_model import PowerTrafficProjector
from sim_engine.bridge import RealSimulationBridge

try:
    from visualize_thermal import plot_chiplet_thermal, plot_sa_convergence
except ImportError:
    plot_chiplet_thermal = None
    plot_sa_convergence = None
    print("[提示] 未检测到绘图库，将只保存 ptrace 数据，不生成热力图或收敛图。")

class MoEAutoTuner:
    def __init__(self, hw_file="configs/hw_config.json", moe_file="configs/moe_params.json"):
        with open(hw_file, 'r') as f: self.hw_config = json.load(f)
        with open(moe_file, 'r') as f: self.moe_config = json.load(f)
        
        self.Ne = self.moe_config['architecture']['num_experts_total']
        self.Nc = self.hw_config['chiplet_arch']['num_chiplets']
        
        self.checkpoint_path = os.path.abspath(f"moe_weights_Ne{self.Ne}.pt")
        if not os.path.exists(self.checkpoint_path):
            print(f"[警告] 未找到 {self.Ne} 专家的权重文件！请先运行 train_custom_experts.py")
        else:
            print(f"[Tuner] 侦测到权重文件: moe_weights_Ne{self.Ne}.pt")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.abspath(f"outputs/run_Nc{self.Nc}_Ne{self.Ne}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.flp_path = os.path.join(self.run_dir, f"dynamic_chiplet_{self.Nc}.flp")
        self.hotspot_config_path = os.path.join(self.run_dir, f"dynamic_hotspot_{self.Nc}.config")
        
        self._generate_dynamic_flp()
        self._generate_dynamic_hotspot_config()
        
        self.mapper = ExpertMapper(self.Ne, self.Nc)
        self.projector = PowerTrafficProjector(self.hw_config, self.moe_config)
        self.bridge = RealSimulationBridge(self.run_dir, self.flp_path, self.hotspot_config_path)

    def _generate_dynamic_flp(self):
        """生成包含两侧 HBM 的智能完美矩形底平面布局"""
        def get_optimal_grid(n):
            for i in range(int(np.sqrt(n)), 1, -1):
                if n % i == 0:
                    return n // i, i
            c = int(np.ceil(np.sqrt(n)))
            r = int(np.ceil(n / c))
            return c, r

        cols, rows = get_optimal_grid(self.Nc)
        
        chiplet_w = chiplet_h = 0.005
        hbm_w = 0.005
        
        self.compute_total_width = cols * chiplet_w
        self.total_height = rows * chiplet_h
        self.sys_total_width = self.compute_total_width + 2 * hbm_w
        
        with open(self.flp_path, 'w') as f:
            f.write("# Line format: <unit-name> <width> <height> <left-x> <bottom-y>\n")
            
            # 左侧 HBM
            f.write(f"HBM_Left\t{hbm_w:.6f}\t{self.total_height:.6f}\t0.000000\t0.000000\n")
            
            # 中间计算芯粒
            for i in range(self.Nc):
                x_offset = hbm_w + (i % cols) * chiplet_w
                y_offset = (rows - 1 - (i // cols)) * chiplet_h
                f.write(f"Chiplet_{i}\t{chiplet_w:.6f}\t{chiplet_h:.6f}\t{x_offset:.6f}\t{y_offset:.6f}\n")
                
            # 右侧 HBM
            x_right = hbm_w + self.compute_total_width
            f.write(f"HBM_Right\t{hbm_w:.6f}\t{self.total_height:.6f}\t{x_right:.6f}\t0.000000\n")

    def _generate_dynamic_hotspot_config(self):
        base_config = os.path.abspath("./configs/hotspot_base.config")
        min_spreader = max(0.03, max(self.sys_total_width, self.total_height) + 0.01)
        min_sink = max(0.06, max(self.sys_total_width, self.total_height) + 0.03)
        with open(base_config, 'r') as fin, open(self.hotspot_config_path, 'w') as fout:
            for line in fin:
                if '-s_spreader' in line and not line.strip().startswith('#'): fout.write(f"\t\t-s_spreader\t\t\t{min_spreader:.4f}\n")
                elif '-s_sink' in line and not line.strip().startswith('#'): fout.write(f"\t\t-s_sink\t\t\t{min_sink:.4f}\n")
                elif '-model_type' in line and not line.strip().startswith('#'): fout.write(f"\t\t-model_type\t\t\tgrid\n")
                elif '-grid_rows' in line or '-grid_cols' in line and not line.strip().startswith('#'): fout.write(f"\t\t-grid_rows\t\t\t64\n" if '-grid_rows' in line else f"\t\t-grid_cols\t\t\t64\n")
                else: fout.write(line)

    def evaluate_policy(self, M, Ve):
        Vc, compute_p = self.projector.project_traffic_and_power(M, Ve)
        noc_p_array = self.projector.estimate_noc_power(Vc) if hasattr(self.projector, 'estimate_noc_power') else np.zeros_like(compute_p)
        
        total_tokens = np.sum(Ve)
        p_hbm_read, p_hbm_write = self.projector.estimate_hbm_power(total_tokens)
        
        steady_path, grid_path = self.bridge.generate_power_trace(
            compute_p + noc_p_array, p_hbm_read, p_hbm_write
        )
        
        try:
            # 核心切片逻辑：[1 : self.Nc + 1] 精准跳过两侧 HBM
            T_all_celsius = np.loadtxt(steady_path, usecols=1)[:self.Nc + 2] - 273.15
            T_array_celsius = T_all_celsius[1:self.Nc + 1]
            t_max_celsius = np.max(T_array_celsius)
        except Exception as e:
            print(f"[Error] 解析稳态温度文件失败: {e}")
            T_array_celsius = np.zeros(self.Nc)
            t_max_celsius = 0.0
            
        return t_max_celsius, T_array_celsius, np.sum(noc_p_array), grid_path

    def run_sa_optimization(self, epochs=50, T_init=10.0, alpha=0.9):
        print(f"\n[Optimizer] 启动真实物理闭环 SA 优化 (需迭代 {epochs} 次)...")
        Ve = self.bridge.get_expert_traffic(self.moe_config, self.checkpoint_path)
        
        curr_M = self.mapper.get_initial_sa_mapping(Ve)
        curr_cost, curr_T_array, _, grid_path = self.evaluate_policy(curr_M, Ve)
        best_M, best_cost, best_T_array, T = curr_M.copy(), curr_cost, curr_T_array.copy(), T_init

        # 记录历史数据用于收敛曲线绘制 (未修正前的区块平均温度)
        history_curr_cost = [curr_cost]
        history_best_cost = [best_cost]

        shutil.copy(os.path.join(self.run_dir, "chiplet.ptrace"), os.path.join(self.run_dir, "chiplet_best_epoch_0_greedy.ptrace"))
        if plot_chiplet_thermal is not None:
            plot_chiplet_thermal(self.flp_path, grid_path, os.path.join(self.run_dir, "sa_best_epoch_0_greedy.png"))

        for epoch in range(epochs):
            new_M = self.mapper.generate_neighbor(curr_M, Ve, T_real_array=curr_T_array)
            new_cost, new_T_array, _, current_grid_path = self.evaluate_policy(new_M, Ve)
            
            if new_cost < curr_cost or np.random.rand() < np.exp((curr_cost - new_cost) / T):
                curr_M, curr_cost, curr_T_array = new_M, new_cost, new_T_array
                
                if new_cost < best_cost:
                    best_cost, best_M = new_cost, new_M.copy()
                    
                    shutil.copy(os.path.join(self.run_dir, "chiplet.ptrace"), 
                                os.path.join(self.run_dir, f"chiplet_best_epoch_{epoch+1}.ptrace"))
                    
                    if plot_chiplet_thermal is not None:
                        plot_chiplet_thermal(self.flp_path, current_grid_path, 
                                             os.path.join(self.run_dir, f"sa_best_epoch_{epoch+1}.png"))
            
            T *= alpha
            
            # 终端打印的是实际采纳的当前解，而不是废弃的探索解
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:04d} | 当前追踪最高温: {curr_cost:.2f}°C | 历史最优: {best_cost:.2f}°C")
            
            # 收集本轮评估后的数据
            history_curr_cost.append(curr_cost)
            history_best_cost.append(best_cost)
            
        print(f"\n[Summary] 原始计算 SA 优化完成！未修正最佳 T_max (Block Avg): {best_cost:.2f}°C")

        # ================== 终极修复：全自动精准温度修正与统计信息 ==================
        print("\n[Alignment] 正在提取真实网格最高温进行精准校准...")
        
        # 1. 重新跑一次最优解，拿到最终的网格温度文件路径
        _, _, _, final_grid_path = self.evaluate_policy(best_M, Ve)

        # 2. 从网格文件中提取真实的最高温
        true_grid_max = best_cost
        if os.path.exists(final_grid_path):
            temps_kelvin = []
            is_layer_0 = False
            with open(final_grid_path, "r") as fp:
                for line in fp:
                    line = line.strip()
                    if line.startswith("Layer 0"): is_layer_0 = True; continue
                    if line.startswith("Layer 1"): break
                    if is_layer_0 and line and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) >= 2:
                            try: temps_kelvin.append(float(parts[1]))
                            except ValueError: continue
            if temps_kelvin:
                true_grid_max = np.max(temps_kelvin) - 273.15

        # 3. 计算精准的动态偏移量
        exact_shift = true_grid_max - best_cost

        # 4. 对整个历史数据进行精准平移
        shifted_curr_cost = [t + exact_shift for t in history_curr_cost]
        shifted_best_cost = [t + exact_shift for t in history_best_cost]

        mean_temp = np.mean(shifted_curr_cost)
        std_temp = np.std(shifted_curr_cost)

        print(f"============================================================")
        print(f"[Statistics] 依据网格热力图进行精准校准 (+{exact_shift:.2f}°C) 后的最终统计:")
        print(f"  - 修正后最终最优温度: {shifted_best_cost[-1]:.2f}°C (完美匹配热力图 Max)")
        print(f"  - 全部迭代的平均温度: {mean_temp:.2f}°C")
        print(f"  - 全部迭代的温度波动(标准差): {std_temp:.2f}°C")
        print(f"============================================================")
        # =======================================================================

        # 生成收敛曲线图 (使用精准修正后的数据)
        if plot_sa_convergence is not None:
            convergence_img_path = os.path.join(self.run_dir, "sa_convergence_curve.png")
            plot_sa_convergence(shifted_curr_cost, shifted_best_cost, convergence_img_path)
            print(f"[Plot] 精准修正后的收敛曲线已保存至: sa_convergence_curve.png\n")

if __name__ == "__main__":
    tuner = MoEAutoTuner()
    
    # 删除了 tuner.run_baselines()，直接启动闭环优化测试
    tuner.run_sa_optimization(epochs=1000)
