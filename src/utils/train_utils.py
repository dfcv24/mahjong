import matplotlib.pyplot as plt
import os
from datetime import datetime

# 创建保存可视化结果的文件夹
def create_results_dir(task="discard_models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"models/{task}/training_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

# 可视化训练过程
def plot_training_history(history, results_dir):
    """
    绘制训练和验证的损失与准确率曲线
    
    参数:
    history: 包含训练历史数据的字典
    results_dir: 保存图表的目录
    """
    # 确定训练周期数量
    epochs = range(1, len(history['val_loss']) + 1)
    
    # 创建两个子图：左边是损失，右边是准确率
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', marker='o', markersize=4, linewidth=2, label='训练损失')
    ax1.plot(epochs, history['val_loss'], 'r-', marker='s', markersize=4, linewidth=2, label='验证损失')
    ax1.set_title('训练与验证损失')
    ax1.set_xlabel('周期')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制准确率曲线
    ax2.plot(epochs, history['train_accuracy'], 'b-', marker='o', markersize=4, linewidth=2, label='训练准确率')
    ax2.plot(epochs, history['val_accuracy'], 'r-', marker='s', markersize=4, linewidth=2, label='验证准确率')
    if 'val_top3_accuracy' in history:
        ax2.plot(epochs, history['val_top3_accuracy'], 'g-', marker='^', markersize=4, linewidth=2, label='验证Top-3准确率')
    ax2.set_title('训练与验证准确率')
    ax2.set_xlabel('周期')
    ax2.set_ylabel('准确率')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 设置准确率图的y轴范围从0到1
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # 确保目录存在
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 保存图表
    save_path = os.path.join(results_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"训练历史图表已保存至 {save_path}")
# 设置中文字体支持
def setup_chinese_font():
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']