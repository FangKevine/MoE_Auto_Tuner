# ~/MoE_Auto_Tuner/visualize_thermal.py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_chiplet_thermal(flp_filename, grid_temp_filename, save_path):
    if not os.path.exists(grid_temp_filename) or not os.path.exists(flp_filename):
        print(f"[Error] Required files for visualization are missing.")
        return

    fig, axs = plt.subplots(1, figsize=(8, 7))
    total_width = total_length = -np.inf
    
    with open(flp_filename, "r") as fp:
        for line in fp:
            if line.strip() == "" or line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 5:
                width, length = float(parts[1]), float(parts[2])
                x, y = float(parts[3]), float(parts[4])
                rect = plt.Rectangle((x, y), width, length, fc="none", ec="black", lw=0.5, alpha=0.3)
                axs.add_patch(rect)
                total_width, total_length = max(total_width, x + width), max(total_length, y + length)

    # ========================================================
    # 精准拦截，只读取 Layer 0 (Silicon) 硅片层的温度
    # ========================================================
    temps_kelvin = []
    is_layer_0 = False
    
    with open(grid_temp_filename, "r") as fp:
        for line in fp:
            line = line.strip()
            
            # 状态机：遇到 Layer 0 开始读取，遇到 Layer 1 立即终止
            if line.startswith("Layer 0"):
                is_layer_0 = True
                continue
            if line.startswith("Layer 1"):
                break
                
            if is_layer_0 and line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    try: temps_kelvin.append(float(parts[1]))
                    except ValueError: continue

    if not temps_kelvin:
        print("[Error] Failed to extract Layer 0 temperatures.")
        return

    temps = [t - 273.15 for t in temps_kelvin]
    grid_res = int(np.sqrt(len(temps)))
    temps_reshaped = np.reshape(temps[:grid_res*grid_res], (grid_res, grid_res))
    
    # 翻转矩阵以对齐物理坐标轴
    temps_reshaped = np.flipud(temps_reshaped)

    im = axs.imshow(temps_reshaped, cmap='hot_r', extent=(0, total_width, 0, total_length), 
                    origin='lower', aspect='equal', interpolation='gaussian')
    
    cbar = fig.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)', fontweight='bold')
    
    axs.set_title(f"Dynamic MoE Thermal Map (Max: {np.max(temps):.2f}°C)")
    axs.set_xlabel("Horizontal Position (mm)")
    axs.set_ylabel("Vertical Position (mm)")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sa_convergence(history_curr, history_best, save_path):
    """绘制模拟退火算法的收敛曲线 (接收到的已经是主程序精准校准过的数据)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(len(history_curr))
    
    # 绘制当前状态的变化轨迹
    ax.plot(epochs, history_curr, label='Current State Max Temp', alpha=0.5, color='orange', linestyle='-')
    
    # 绘制历史最优状态的下降轨迹
    ax.plot(epochs, history_best, label='Global Best Max Temp', color='red', linewidth=2)
    
    ax.set_title('Simulated Annealing Convergence Curve (Calibrated)', fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Max Chiplet Temperature (°C)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
