import os
import json
import subprocess
import re
import numpy as np

class RealSimulationBridge:
    def __init__(self, outputs_dir="./outputs", sims_dir="./simulators"):
        self.out_dir = os.path.abspath(outputs_dir)
        self.sims_dir = os.path.abspath(sims_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        
        # 路径精准指向编译后的 src 目录
        self.booksim_bin = os.path.join(self.sims_dir, "booksim2", "src", "booksim")
        self.hotspot_bin = os.path.join(self.sims_dir, "hotspot", "hotspot")
        self.run_sim_script = os.path.abspath("./run_sim.py")

        # 启动自检：确保二进制文件物理存在
        if not os.path.exists(self.booksim_bin):
            raise FileNotFoundError(f"[Error] BookSim2 binary not found at {self.booksim_bin}")
        if not os.path.exists(self.hotspot_bin):
            raise FileNotFoundError(f"[Error] HotSpot binary not found at {self.hotspot_bin}")

    def run_pytorchsim_docker(self, moe_params):
        """调用 Docker 执行真实推理，挂载宿主机脚本"""
        config_path = os.path.join(self.out_dir, "temp_moe_config.json")
        result_path = os.path.join(self.out_dir, "pytorch_expert_traffic.json")
        
        with open(config_path, 'w') as f:
            json.dump(moe_params, f)

        print("[Bridge] Triggering native PyTorch computation via Docker...")
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{self.out_dir}:/app/outputs",
            "-v", f"{self.run_sim_script}:/workspace/run_sim.py",
            "-w", "/workspace",
            "ghcr.io/psal-postech/torchsim-ci:v1.0.1", "python", "run_sim.py", 
            "--config", "/app/outputs/temp_moe_config.json",
            "--out", "/app/outputs/pytorch_expert_traffic.json"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"PyTorch Docker execution failed. {e}")

        with open(result_path, 'r') as f:
            traffic_data = json.load(f)
        
        Ne = moe_params['architecture']['num_experts_total']
        Ve = np.zeros((Ne, 1))
        for i in range(Ne):
            Ve[i, 0] = traffic_data.get(f"expert_{i}", 0)
        return Ve

    def run_booksim_real(self, chiplet_traffic_Vc):
        """调用 BookSim2，宽容处理 255 状态码并提取结果"""
        bs_config_path = os.path.join(self.out_dir, "current_booksim.cfg")
        log_path = os.path.join(self.out_dir, "booksim.log")
        
        # 参数微调：基于你的流量规模
        avg_injection_rate = float(np.mean(chiplet_traffic_Vc) / 2000000.0)
        
        with open(os.path.join("./configs", "base_booksim.cfg"), 'r') as f:
            cfg_content = f.read()
        cfg_content += f"\ninjection_rate = {avg_injection_rate};\n"
        
        with open(bs_config_path, 'w') as f:
            f.write(cfg_content)

        print(f"[Bridge] Running BookSim2 (Injection: {avg_injection_rate:.6f})...")
        bin_dir = os.path.dirname(self.booksim_bin)

        # 核心逻辑：BookSim2 在退出时即使报 255 错误，只要统计信息已生成即可
        with open(log_path, 'w') as log_file:
            result = subprocess.run(
                [self.booksim_bin, bs_config_path],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=bin_dir
            )
            # 仅在非 0 且非 255 时抛出异常（如 Segfault）
            if result.returncode not in [0, 255]:
                raise RuntimeError(f"BookSim2 failed with fatal code {result.returncode}")

        latency, noc_power = None, 0.5 # 功耗默认给个基准值
        with open(log_path, 'r') as f:
            log_text = f.read()
            # 兼容不同版本的输出格式
            lat_match = re.search(r'(?:Packet latency average|Average latency)\s*=\s*([0-9\.]+)', log_text)
            pow_match = re.search(r'Total Power\s*=\s*([0-9\.]+)', log_text)
            
            if lat_match: 
                latency = float(lat_match.group(1))
            if pow_match: 
                noc_power = float(pow_match.group(1))

        if latency is None:
            with open(log_path, 'r') as f:
                tail = "".join(f.readlines()[-10:])
            raise ValueError(f"Failed to parse latency from BookSim2. Log tail:\n{tail}")

        return latency, noc_power

    def run_hotspot_real(self, compute_power, noc_power_scalar):
        """生成 ptrace 并调用 HotSpot 求解稳态热图"""
        ptrace_path = os.path.join(self.out_dir, "system.ptrace")
        steady_temp_path = os.path.join(self.out_dir, "steady.temp")
        
        # 计算功耗 + 通信功耗均摊
        total_power = compute_power + (noc_power_scalar / len(compute_power))
        Nc = len(total_power)

        with open(ptrace_path, 'w') as f:
            headers = [f"Chiplet_{i}" for i in range(Nc)]
            f.write("\t".join(headers) + "\n")
            power_strs = [f"{p:.6f}" for p in total_power.flatten()]
            f.write("\t".join(power_strs) + "\n")

        print("[Bridge] Solving Thermal PDEs via HotSpot Engine...")
        bin_dir = os.path.dirname(self.hotspot_bin)
        
        # 确保使用绝对路径引用 config
        cmd = [
            self.hotspot_bin,
            "-c", os.path.abspath("./configs/hotspot_base.config"),
            "-f", os.path.abspath("./configs/chiplet_16.flp"),
            "-p", ptrace_path,
            "-steady_file", steady_temp_path
        ]
        
        try:
            subprocess.run(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.STDOUT, 
                check=True, 
                cwd=bin_dir
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"HotSpot failed with code {e.returncode}")

        t_max = 0.0
        with open(steady_temp_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and "Chiplet_" in parts[0]:
                    t_max = max(t_max, float(parts[1]))
        
        return t_max, 0.0
