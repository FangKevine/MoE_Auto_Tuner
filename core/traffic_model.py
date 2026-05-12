import numpy as np

class PowerTrafficProjector:
    def __init__(self, hw_config, moe_config):
        self.hw = hw_config
        self.moe = moe_config
        
        self.static_power = self.hw['chiplet_arch']['static_power_W']
        self.e_mac = self.hw['energy_model']['mac_energy_pJ'] * 1e-12 
        
        h_dim = self.moe['architecture']['hidden_dim']
        f_dim = self.moe['architecture']['ffn_dim']
        self.macs_per_token = 2 * h_dim * f_dim
        
        # HBM 读写单 Token 能量消耗常数 (J/token)
        self.e_hbm_read_per_token = 8.0e-10  
        self.e_hbm_write_per_token = 8.0e-10 

    def _get_window_time(self, window_tokens):
        batch_size = self.moe['workload']['batch_size']
        seq_length = self.moe['workload']['seq_length']
        macro_tokens = batch_size * seq_length
        
        total_time_s = self.moe['workload']['simulation_time_ms'] / 1000.0
        window_time_s = total_time_s * (window_tokens / macro_tokens)
        
        return max(window_time_s, 1e-9)

    def project_traffic_and_power(self, M, Ve_window):
        Vc_window = np.dot(M.T, Ve_window)
        E_expert_dynamic = Ve_window * self.macs_per_token * self.e_mac 
        E_chiplet_dynamic = np.dot(M.T, E_expert_dynamic) 
        
        total_window_tokens = np.sum(Ve_window)
        window_time_s = self._get_window_time(total_window_tokens)
        
        P_dynamic = E_chiplet_dynamic / window_time_s
        P_compute_total = self.static_power + P_dynamic.flatten()
        
        return Vc_window.flatten(), P_compute_total

    def estimate_noc_power(self, Vc_window):
        h_dim = self.moe['architecture']['hidden_dim']
        bits_per_token = h_dim * 16.0 
        e_bit_noc_J = 1.5e-12 
        
        E_noc_chiplet = Vc_window * bits_per_token * e_bit_noc_J
        total_window_tokens = np.sum(Vc_window)
        window_time_s = self._get_window_time(total_window_tokens)
        
        P_noc_chiplet = E_noc_chiplet / window_time_s
        return P_noc_chiplet

    def estimate_hbm_power(self, total_window_tokens):
        """基于全局 Token 吞吐量计算两侧 HBM 的稳态功耗"""
        window_time_s = self._get_window_time(total_window_tokens)
        p_hbm_read = (total_window_tokens * self.e_hbm_read_per_token) / window_time_s
        p_hbm_write = (total_window_tokens * self.e_hbm_write_per_token) / window_time_s
        
        return p_hbm_read, p_hbm_write
