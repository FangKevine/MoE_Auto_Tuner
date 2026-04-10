import numpy as np

class ExpertMapper:
    def __init__(self, num_experts, num_chiplets):
        self.Ne = num_experts
        self.Nc = num_chiplets

    def generate_mapping(self, strategy="round_robin"):
        """
        生成严格的映射矩阵 M (Ne x Nc)
        M[i, j] = 1 表示专家 i 被映射到物理芯粒 j
        """
        M = np.zeros((self.Ne, self.Nc), dtype=int)
        
        if strategy == "round_robin":
            # 轮询映射：将专家均匀摊到各个芯粒上
            for i in range(self.Ne):
                M[i, i % self.Nc] = 1
                
        elif strategy == "random":
            # 随机映射：作为 Baseline，用于验证热点分布
            for i in range(self.Ne):
                M[i, np.random.randint(0, self.Nc)] = 1
                
        return M
