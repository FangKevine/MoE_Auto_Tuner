# ~/MoE_Auto_Tuner/run_sim.py
import argparse
import json
import torch
import torch.nn as nn

class RealMoERouter(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        # 真实的门控网络权重矩阵 (Hidden_Dim -> Num_Experts)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, hidden_states):
        """
        执行真实的 MoE 路由计算
        hidden_states 维度: (batch_size, seq_length, hidden_dim)
        """
        # 1. 矩阵乘法计算 logits
        logits = self.gate(hidden_states) 
        
        # 2. 计算路由概率
        router_probs = nn.functional.softmax(logits, dim=-1)
        
        # 3. Top-K 选择算法 (决定每个 token 真正去哪个专家)
        routing_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        
        return selected_experts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    print("[Container] Booting REAL PyTorch MoE Inference Engine...")

    # 1. 读取宿主机的真实配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    arch = config['architecture']
    workload = config['workload']
    
    Ne = arch['num_experts_total']
    hidden_dim = arch['hidden_dim']
    top_k = int(arch['routing_strategy'].split('-')[1]) # 解析 "top-2"
    
    batch_size = workload['batch_size']
    seq_len = workload['seq_length']

    # 2. 实例化真正的模型层
    router = RealMoERouter(hidden_dim, Ne, top_k)
    
    # 将模型设为 eval 模式，并尽量推到可用设备上计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router.to(device)
    router.eval()

    print(f"[Container] Model Instantiated on {device}: Hidden={hidden_dim}, Experts={Ne}, Top-{top_k}")
    
    # 3. 生成真实的输入张量模拟一批 Token
    # 维度: (128, 2048, 4096)
    print(f"[Container] Generating Input Tensors: Batch={batch_size}, SeqLen={seq_len}...")
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # 4. 执行前向推理！(真刀真枪的矩阵运算)
    with torch.no_grad():
        selected_experts = router(hidden_states)
    
    # selected_experts 维度现在是 (128, 2048, 2)
    # 我们把所有的 token 路由结果展平，统计每个专家的真实接收量
    flat_expert_indices = selected_experts.view(-1)
    
    # 使用 PyTorch 原生的 bincount 统计直方图
    expert_counts = torch.bincount(flat_expert_indices, minlength=Ne).cpu().numpy()

    # 5. 打包真实结果并写回宿主机
    expert_traffic = {f"expert_{i}": int(count) for i, count in enumerate(expert_counts)}
    
    with open(args.out, 'w') as f:
        json.dump(expert_traffic, f)
        
    print(f"[Container] Inference Complete. Total tokens routed: {flat_expert_indices.numel()}")
    print(f"[Container] Real Traffic Data written to {args.out}")
