import os
import shutil
from main_tuner import MoEAutoTuner

try:
    from visualize_thermal import plot_chiplet_thermal
except ImportError:
    plot_chiplet_thermal = None
    print("[提示] 未检测到绘图库，将只保存数据，不生成热力图。")

class ExtendedMoETuner(MoEAutoTuner):
    def __init__(self, hw_file="configs/hw_config.json", moe_file="configs/moe_params.json"):
        # 完全复用现有的配置读取、环境搭建和对象实例化
        super().__init__(hw_file, moe_file)

    def run_baselines_and_save_grids(self):
        print(f"\n[补充实验] 启动基准对照测试，并持久化存储三种映射的网格温度文件...")
        # 1. 获取专家真实流量负载
        Ve = self.bridge.get_expert_traffic(self.moe_config, self.checkpoint_path)
        
        # 2. 获取三种基准映射矩阵
        baselines = {
            "Sequential": self.mapper.get_sequential_mapping(),
            "Random": self.mapper.get_random_mapping(),
            "Uniform": self.mapper.get_uniform_mapping()
        }

        results = {}
        for name, M in baselines.items():
            print(f"\n----------------------------------------")
            print(f"正在评估 [{name}] 映射方案...")
            
            # 3. 复用父类的评估接口进行热仿真
            t_max, _, _, grid_path = self.evaluate_policy(M, Ve)
            results[name] = t_max
            print(f"[{name}] 评估完成，最高稳态温度: {t_max:.2f}°C")

            # 4. 保存 功耗分布文件 (ptrace)
            source_ptrace = os.path.join(self.run_dir, "chiplet.ptrace")
            target_ptrace = os.path.join(self.run_dir, f"chiplet_baseline_{name}.ptrace")
            shutil.copy(source_ptrace, target_ptrace)

            # 5. 保存 网格温度文件 (grid_steady.temp)
            target_grid = os.path.join(self.run_dir, f"grid_steady_baseline_{name}.temp")
            shutil.copy(grid_path, target_grid)
            print(f"[文件保存] 网格温度文件已保存: grid_steady_baseline_{name}.temp")

            # 6. 生成热力图
            if plot_chiplet_thermal is not None:
                target_png = os.path.join(self.run_dir, f"baseline_thermal_{name}.png")
                plot_chiplet_thermal(self.flp_path, grid_path, target_png)

        print(f"\n所有实验数据已保存至目录: {self.run_dir}")
        return results

if __name__ == "__main__":
    experiment_runner = ExtendedMoETuner()
    experiment_runner.run_baselines_and_save_grids()
