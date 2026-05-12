import os
import json
import subprocess
import numpy as np

class RealSimulationBridge:
    def __init__(self, run_dir, flp_path, config_path, sims_dir="./simulators"):
        self.run_dir = run_dir
        self.flp_path = flp_path
        self.config_path = config_path  
        self.sims_dir = os.path.abspath(sims_dir)
        self.hotspot_bin = os.path.join(self.sims_dir, "hotspot", "hotspot")
        self.run_sim_script = os.path.abspath("./run_sim.py")

    def run_pytorchsim_docker(self, moe_params, checkpoint_path=None):
        config_path = os.path.join(self.run_dir, "temp_moe_config.json")
        out_trace = os.path.join(self.run_dir, "expert_traffic.json")
        
        with open(config_path, 'w') as f: json.dump(moe_params, f)
            
        cmd = ["python3", self.run_sim_script, "--config", config_path, "--out", out_trace]
        if checkpoint_path and os.path.exists(checkpoint_path):
            cmd.extend(["--checkpoint", checkpoint_path])
            
        subprocess.run(cmd, check=True)

    def get_expert_traffic(self, moe_params, checkpoint_path=None):
        out_trace = os.path.join(self.run_dir, "expert_traffic.json")
        self.run_pytorchsim_docker(moe_params, checkpoint_path)
        with open(out_trace, 'r') as f: data = json.load(f)
        return np.array(data.get("expert_tokens", []), dtype=np.float32).reshape(-1, 1)

    def generate_power_trace(self, total_power_array, p_hbm_read=0.0, p_hbm_write=0.0):
        ptrace_path = os.path.join(self.run_dir, "chiplet.ptrace")
        steady_temp_path = os.path.join(self.run_dir, "steady.temp")
        grid_temp_path = os.path.join(self.run_dir, "grid_steady.temp")
        
        Nc = len(total_power_array)
        with open(ptrace_path, 'w') as f:
            # 必须与 FLP 文件中的元件命名和顺序严格一致
            names = ["HBM_Left"] + [f"Chiplet_{i}" for i in range(Nc)] + ["HBM_Right"]
            f.write("\t".join(names) + "\n")
            
            # 拼装包含 HBM 读写功耗的 trace 数据
            powers = [f"{p_hbm_read:.6f}"] + [f"{p:.6f}" for p in total_power_array.flatten()] + [f"{p_hbm_write:.6f}"]
            f.write("\t".join(powers) + "\n")

        cmd = [self.hotspot_bin, "-c", self.config_path, "-f", self.flp_path,  
               "-p", ptrace_path, "-steady_file", steady_temp_path, "-grid_steady_file", grid_temp_path]
        subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(self.hotspot_bin))
        return steady_temp_path, grid_temp_path
