import numpy as np

class ExpertMapper:
    def __init__(self, num_experts, num_chiplets):
        self.Ne = num_experts
        self.Nc = num_chiplets
        
        # 物理底线检查：如果专家数比芯粒数还少，必然会出现空转芯粒
        if self.Ne < self.Nc:
            raise ValueError(f"物理约束冲突: 专家数({self.Ne})必须大于等于芯粒数({self.Nc})才能保证无空转！")
            
        # ====================================================================
        # 智能网格分解：寻找完美的无缺角矩形阵列 (例如 32 -> 8x4)
        # ====================================================================
        def get_optimal_grid(n):
            for i in range(int(np.sqrt(n)), 1, -1):
                if n % i == 0:
                    return n // i, i  # 返回 cols, rows
            # 只有遇到素数才退化为带缺角的近似正方形
            c = int(np.ceil(np.sqrt(n)))
            r = int(np.ceil(n / c))
            return c, r

        self.cols, self.rows = get_optimal_grid(self.Nc)
        
        # ====================================================================
        # 异构散热能力模型 (保护 HBM 冷源不受热串扰)
        # ====================================================================
        self.G_thermal = np.ones(self.Nc)
        for i in range(self.Nc):
            r, c = i // self.cols, i % self.cols
            is_top_bottom_edge = (r == 0 or r == self.rows - 1)
            is_left_right_edge = (c == 0 or c == self.cols - 1)
            
            if is_left_right_edge:
                # 紧贴两侧 HBM 冷源，散热极佳
                self.G_thermal[i] = 2.0
            elif is_top_bottom_edge:
                # 接触封装上下边缘，散热较好
                self.G_thermal[i] = 1.5
            else:
                # 阵列中心，容易发生热量淤积
                self.G_thermal[i] = 1.0
                
            # 四个角落：双面开阔且贴 HBM，散热无敌
            if is_top_bottom_edge and is_left_right_edge:
                self.G_thermal[i] = 2.5

    # ====================================================================
    # 基准测试静态映射 (Static Baselines)
    # ====================================================================
    def get_sequential_mapping(self):
        M = np.zeros((self.Ne, self.Nc), dtype=int)
        for e in range(self.Ne):
            M[e, e % self.Nc] = 1
        return M

    def get_random_mapping(self):
        M = np.zeros((self.Ne, self.Nc), dtype=int)
        experts = np.random.permutation(self.Ne)
        # 阶段 1：先给每个芯粒分配 1 个随机专家保底
        for i in range(self.Nc):
            M[experts[i], i] = 1
        # 阶段 2：剩下的专家彻底随机分配
        for e in experts[self.Nc:]:
            M[e, np.random.randint(0, self.Nc)] = 1
        return M

    def get_uniform_mapping(self):
        M = np.zeros((self.Ne, self.Nc), dtype=int)
        base_assignment = np.tile(np.arange(self.Nc), int(np.ceil(self.Ne / self.Nc)))[:self.Ne]
        np.random.shuffle(base_assignment)
        for e in range(self.Ne):
            M[e, base_assignment[e]] = 1
        return M

    # ====================================================================
    # 动态映射：负载感知型 SA 引擎 (Load-Aware SA)
    # ====================================================================
    def get_initial_sa_mapping(self, Ve):
        """完全自适应的两阶段热容量贪心初始化"""
        Ve_flat = Ve.flatten()
        sorted_expert_indices = np.argsort(Ve_flat)[::-1]
        
        M = np.zeros((self.Ne, self.Nc), dtype=int)
        chiplet_loads = np.zeros(self.Nc, dtype=float)
        
        best_cooling_chiplets = np.argsort(self.G_thermal)[::-1]
        
        # 阶段 1：强制保底 (1专家/芯粒)。把最重的前 Nc 个专家，喂给散热最好的 Nc 个芯粒
        for i in range(self.Nc):
            exp_idx = sorted_expert_indices[i]
            target_c = best_cooling_chiplets[i]
            M[exp_idx, target_c] = 1
            chiplet_loads[target_c] += Ve_flat[exp_idx]
            
        # 阶段 2：剩余的 (Ne - Nc) 个专家，按动态热容量贪心分配
        for exp_idx in sorted_expert_indices[self.Nc:]:
            T_proxy = chiplet_loads / self.G_thermal
            target_c = np.argmin(T_proxy)
            M[exp_idx, target_c] = 1
            chiplet_loads[target_c] += Ve_flat[exp_idx]
            
        return M

    def generate_neighbor(self, current_M, Ve, T_real_array=None):
        """
        负载感知型 SA 扰动 (Load-Aware SA)
        强制将大核推向边缘，将小核吸入中心保底
        """
        new_M = current_M.copy()
        Ve_flat = Ve.flatten()
        
        if T_real_array is not None:
            hottest_c = np.argmax(T_real_array)
            coolest_c = np.argmin(T_real_array)
        else:
            chiplet_loads = np.sum(new_M * Ve_flat[:, np.newaxis], axis=0)
            T_proxy = chiplet_loads / self.G_thermal
            hottest_c = np.argmax(T_proxy)
            coolest_c = np.argmin(T_proxy)
        
        rand_val = np.random.rand()
        
        if rand_val < 0.50:
            # 策略 A：负载感知精准对调 (Load-Aware Swap)
            experts_in_hot = np.where(new_M[:, hottest_c] == 1)[0]
            experts_in_cool = np.where(new_M[:, coolest_c] == 1)[0]
            
            if len(experts_in_hot) > 0 and len(experts_in_cool) > 0:
                e_h = experts_in_hot[np.argmax(Ve_flat[experts_in_hot])]
                e_c = experts_in_cool[np.argmin(Ve_flat[experts_in_cool])]
                
                new_M[e_h, hottest_c], new_M[e_h, coolest_c] = 0, 1
                new_M[e_c, coolest_c], new_M[e_c, hottest_c] = 0, 1
                
        elif rand_val < 0.85:
            # 策略 B：负载感知精准疏散 (Load-Aware Move)
            experts_in_hot = np.where(new_M[:, hottest_c] == 1)[0]
            if len(experts_in_hot) > 1:
                e_move = experts_in_hot[np.argmax(Ve_flat[experts_in_hot])]
                new_M[e_move, hottest_c], new_M[e_move, coolest_c] = 0, 1
            else:
                experts_in_cool = np.where(new_M[:, coolest_c] == 1)[0]
                if len(experts_in_cool) > 0:
                    e_h = experts_in_hot[np.argmax(Ve_flat[experts_in_hot])]
                    e_c = experts_in_cool[np.argmin(Ve_flat[experts_in_cool])]
                    new_M[e_h, hottest_c], new_M[e_h, coolest_c] = 0, 1
                    new_M[e_c, coolest_c], new_M[e_c, hottest_c] = 0, 1
                    
        else:
            # 策略 C：全局随机突变 (防止陷入局部最优)
            exp = np.random.choice(self.Ne)
            old_c = np.argmax(new_M[exp])
            new_c = np.random.choice(self.Nc)
            
            if old_c != new_c:
                experts_in_old = np.where(new_M[:, old_c] == 1)[0]
                if len(experts_in_old) > 1:
                    new_M[exp, old_c], new_M[exp, new_c] = 0, 1
                else:
                    experts_in_new = np.where(new_M[:, new_c] == 1)[0]
                    if len(experts_in_new) > 0:
                        exp_swap = np.random.choice(experts_in_new)
                        new_M[exp, old_c], new_M[exp, new_c] = 0, 1
                        new_M[exp_swap, new_c], new_M[exp_swap, old_c] = 0, 1
                        
        return new_M
