import os
import sys
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# 动态挂载 Docker 里的 PyTorchSim 官方库
torchsim_dir = os.environ.get('TORCHSIM_DIR', '/workspace/PyTorchSim')
sys.path.append(torchsim_dir)

# 动态导入官方 MoE 类
import importlib.util
test_moe_path = os.path.join(torchsim_dir, "tests/MoE/test_moe.py")
spec = importlib.util.spec_from_file_location("test_moe", test_moe_path)
test_moe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_moe)

MoE = test_moe.MoE

# 本地 Dynamo 补丁，确保跨环境不报错
def patch_metrics_context_update():
    try:
        from torch._dynamo.utils import get_metrics_context
        ctx = get_metrics_context()
        if ctx is not None:
            original_update = ctx.update
            def patched_update(values, overwrite=True):
                return original_update(values, overwrite=True)
            get_metrics_context().update = patched_update
    except ImportError:
        pass

def train_from_json(config_path="configs/moe_params.json", epochs=5, device_str="npu:0"):
    patch_metrics_context_update()
    
    with open(config_path, 'r') as f:
        moe_config = json.load(f)
    num_experts = moe_config['architecture']['num_experts_total']
    k = int(moe_config['architecture']['routing_strategy'].split('-')[1])

    print(f"\n[Trainer] 读取配置成功 -> 准备训练专家数(Ne): {num_experts}, 路由: Top-{k}")

    try: device = torch.device(device_str)
    except: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    batch_size = 32
    input_size = 28 * 28  
    output_size = 8
    hidden_size = 64

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_dir = os.path.abspath('./dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    train_dataset = datasets.MNIST(root=dataset_dir, train=True, download=True, transform=transform)
    indices = [i for i, label in enumerate(train_dataset.targets) if label < 8][:batch_size * 10]
    train_loader = DataLoader(dataset=Subset(train_dataset, indices), batch_size=batch_size, shuffle=True)

    model = MoE(input_size=input_size, output_size=output_size, num_experts=num_experts, hidden_size=hidden_size, k=k, noisy_gating=True)
    for i in range(num_experts): model.experts[i].requires_grad = True

    model_device = model.to(device=device)
    # 跳过 compile，加快训练速度
    opt_model = model_device

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(opt_model.parameters(), lr=0.001)
    
    opt_model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.view(data.size(0), -1).to(device), target.to(device)
            optimizer.zero_grad()
            output, aux_loss = opt_model(data)
            (loss_fn(output, target) + aux_loss).backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs} Completed.')

    checkpoint_name = os.path.abspath(f"moe_weights_Ne{num_experts}.pt")
    torch.save({key: value.cpu() for key, value in model.state_dict().items()}, checkpoint_name)
    print(f"[Success] 该配置的真实权重已保存: {checkpoint_name}\n")

if __name__ == "__main__":
    train_from_json()
