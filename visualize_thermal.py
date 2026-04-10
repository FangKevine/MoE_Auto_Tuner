import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re  # 引入正则库

def plot_chiplet_thermal(temp_file="outputs/steady.temp", save_path="outputs/thermal_map.png"):
    if not os.path.exists(temp_file):
        print(f"[Error] Temperature file {temp_file} not found.")
        return

    # 1. 容错解析温度数据
    temps = {}
    with open(temp_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and "Chiplet_" in parts[0]:
                # 【核心修复】：使用正则精准提取 Chiplet_ 后面的数字
                match = re.search(r'Chiplet_(\d+)', parts[0])
                if match:
                    idx = int(match.group(1))
                    temp_val = float(parts[1])
                    # HotSpot 可能会输出多层同名组件的温度，我们取最热的一层（硅层通常最热）
                    if idx not in temps:
                        temps[idx] = temp_val
                    else:
                        temps[idx] = max(temps[idx], temp_val)

    if not temps:
        print("[Error] No valid chiplet temperatures found in file.")
        return

    # 2. 矩阵重组 (确保 16 个芯粒都有数据)
    thermal_data = np.zeros((4, 4))
    for i in range(16):
        if i in temps:
            row = 3 - (i // 4)
            col = i % 4
            thermal_data[row, col] = temps[i]
        else:
            print(f"[Warning] Missing data for Chiplet_{i}")

    # 3. 绘制热力图
    plt.figure(figsize=(8, 6))
    # cmap="rocket" 或者 "YlOrRd" 都能呈现很好的高温警示感
    sns.heatmap(thermal_data, annot=True, fmt=".1f", cmap="rocket_r",
                cbar_kws={'label': 'Temperature (°C)'})
    
    plt.title("MoE Multi-Chiplet Thermal Distribution")
    plt.xlabel("Chiplet Column")
    plt.ylabel("Chiplet Row")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Success] 600°C Thermal snapshot saved as {save_path}")

if __name__ == "__main__":
    plot_chiplet_thermal()
