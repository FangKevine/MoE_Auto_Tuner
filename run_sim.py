import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import sys
import json
import argparse
import torch
import torch.nn as nn

sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))
from Scheduler.scheduler import PyTorchSimRunner

class SparseDispatcher(object):
    @torch.compiler.disable(recursive=True)
    def __init__(self, num_experts, gates):
        gates = gates.cpu()
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0] 
        self._part_sizes = (gates > 0).sum(0).tolist() 
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    @torch.compiler.disable(recursive=False)
    def dispatch(self, inp):
        device = inp.device
        split_tensors = torch.split(inp.cpu()[self._batch_index].squeeze(1), self._part_sizes, dim=0)
        return tuple(tensor.clone().to(device) for tensor in split_tensors)

    @torch.compiler.disable(recursive=True)
    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat([out.cpu() for out in expert_out], 0)
        if multiply_by_gates: stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device="cpu")
        return zeros.index_add(0, self._batch_index, stitched.float())

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu, self.soft = nn.ReLU(), nn.Softmax(1)
    def forward(self, x): return self.soft(self.fc2(self.relu(self.fc1(x))))

class RealArchitectureMoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([MLP(input_size, output_size, hidden_size) for _ in range(num_experts)])
        self.w_gate, self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts)), nn.Parameter(torch.zeros(input_size, num_experts))
        self.softplus, self.softmax = nn.Softplus(), nn.Softmax(1)
        self.part_sizes = []

    @torch.compiler.disable(recursive=True)
    def noisy_top_k_gating(self, x):
        logits = self.softmax(x.cpu() @ self.w_gate.cpu())
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_gates = top_logits[:, :self.k] / (top_logits[:, :self.k].sum(1, keepdim=True) + 1e-6)
        zeros = torch.zeros_like(logits, requires_grad=True)
        return zeros.scatter(1, top_indices[:, :self.k], top_k_gates)

    def forward(self, x):
        gates = self.noisy_top_k_gating(x)
        dispatcher = SparseDispatcher(self.num_experts, gates)
        self.part_sizes.append(dispatcher._part_sizes)  
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        return dispatcher.combine(expert_outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f: 
        config = json.load(f)
        arch = config['architecture']
        workload = config['workload']
    
    model = RealArchitectureMoE(input_size=28*28, output_size=8, 
                                num_experts=arch['num_experts_total'], 
                                hidden_size=64, k=int(arch['routing_strategy'].split('-')[1]))
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        # 核心修复：strict=False 忽略多余的 mean 和 std
        model.load_state_dict(torch.load(args.checkpoint), strict=False)
        print(f"[Sim Engine] 加载真实训练权重: {os.path.basename(args.checkpoint)}")

    npu_device = PyTorchSimRunner.setup_device().custom_device()
    model.eval()
    opt_model = torch.compile(model.to(npu_device), dynamic=False)

    torch.manual_seed(42)
    SIM_WINDOW_TOKENS = workload['batch_size'] * 20
    x_input = torch.randn(SIM_WINDOW_TOKENS, 28*28).to(npu_device)

    with torch.no_grad(): opt_model(x_input)
    
    with open(args.out, 'w') as f:
        json.dump({"expert_tokens": model.part_sizes[-1]}, f, indent=4)
