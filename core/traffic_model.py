import numpy as np

class PowerTrafficProjector:
    def __init__(self, hw_config, moe_config):
        self.hw = hw_config
        self.moe = moe_config
        
        # 提取底层物理能量模型参数
        self.static_power = self.hw['chiplet_arch']['static_power_W']
        # 将 pJ (皮焦耳) 转换为 Joules (焦耳)
        self.e_mac = self.hw['energy_model']['mac_energy_pJ'] * 1e-12 
        
        # 计算单个专家处理 1 个 Token 的计算量 (MACs)
        # 严谨的 FFN 乘加次数: 2 * hidden_dim * ffn_dim
        h_dim = self.moe['architecture']['hidden_dim']
        f_dim = self.moe['architecture']['ffn_dim']
        self.macs_per_token = 2 * h_dim * f_dim

    def project_traffic_and_power(self, M, Ve):
        """
        核心矩阵算法：将逻辑流量坍缩为物理负载
        M: 映射矩阵 (Ne x Nc)
        Ve: 专家逻辑流量向量 (Ne x 1)
        """
        # 1. 物理流量投影: V_c = M^T * V_e
        Vc = np.dot(M.T, Ve)
        
        # 2. 专家级动态能量矩阵运算
        # 每个专家的能量 = Token数 * 单个Token的MAC数 * 每次MAC耗能
        E_expert_dynamic = Ve * self.macs_per_token * self.e_mac # (Ne x 1)
        
        # 3. 能量物理投影: E_chiplet = M^T * E_expert
        E_chiplet_dynamic = np.dot(M.T, E_expert_dynamic) # (Nc x 1)
        
        # 4. 转化为动态功耗 (Power = Energy / Time)
        sim_time_s = self.moe['workload']['simulation_time_ms'] / 1000.0
        P_dynamic = E_chiplet_dynamic / sim_time_s
        
        # 总功耗 = 静态功耗 + 动态功耗
        P_compute_total = self.static_power + P_dynamic.flatten()
        
        return Vc.flatten(), P_compute_total
