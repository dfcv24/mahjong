import os
import json
import time
import glob
import argparse
import matplotlib
# 使用Agg后端，这样就不需要图形显示设备
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from src.utils.train_utils import *

def load_latest_stats(stats_dir):
    """加载最新的统计数据文件"""
    json_files = glob.glob(os.path.join(stats_dir, "training_stats_*.json"))
    if not json_files:
        return None
    
    # 按修改时间排序，获取最新的文件
    latest_file = max(json_files, key=os.path.getmtime)
    
    try:
        with open(latest_file, 'r') as f:
            stats = json.load(f)
        print(f"已加载统计数据: {latest_file}")
        return stats
    except Exception as e:
        print(f"加载统计文件时出错: {e}")
        return None

def generate_plots(stats_dir, output_dir):
    """生成并保存监控图表"""
    stats = load_latest_stats(stats_dir)
    if not stats:
        print("未找到统计数据文件，无法生成图表")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 奖励趋势图
    plt.figure(figsize=(12, 6))
    plt.plot(stats['episode_rewards'])
    plt.title('奖励趋势')
    plt.xlabel('局数')
    plt.ylabel('奖励')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"rewards_{timestamp}.png"))
    plt.close()
    
    # 2. 胜率趋势图
    if len(stats['episode_rewards']) > 10:
        plt.figure(figsize=(12, 6))
        win_rates = []
        window_size = 10
        for i in range(window_size, len(stats['episode_rewards'])+1):
            # 计算滑动窗口胜率
            win_count = sum(1 for j in range(i-window_size, i) if stats['episode_rewards'][j] > 0)
            win_rates.append(win_count/window_size)
        
        plt.plot(range(window_size, len(stats['episode_rewards'])+1), win_rates)
        plt.title('胜率趋势 (窗口=10)')
        plt.xlabel('局数')
        plt.ylabel('胜率')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"win_rate_{timestamp}.png"))
        plt.close()
    
    # 3. PPO损失图
    if stats['ppo_stats']['policy_loss']:
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(stats['ppo_stats']['policy_loss'], 'r-')
        plt.title('策略损失')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(stats['ppo_stats']['value_loss'], 'b-')
        plt.title('价值损失')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(stats['ppo_stats']['entropy'])
        plt.title('策略熵')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"ppo_losses_{timestamp}.png"))
        plt.close()
    
    # 4. 动作分布图
    plt.figure(figsize=(12, 6))
    action_labels = list(stats['action_counts'].keys())
    action_values = list(stats['action_counts'].values())
    plt.bar(action_labels, action_values)
    plt.title("动作分布")
    plt.xlabel("动作类型")
    plt.ylabel("次数")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"action_distribution_{timestamp}.png"))
    plt.close()
    
    # 5. 训练摘要图
    plt.figure(figsize=(12, 8))
    plt.text(0.5, 0.5, 
             f"训练摘要\n\n" +
             f"总游戏局数: {stats['total_games']}\n" +
             f"总胜局数: {stats['total_wins']}\n" +
             f"总胜率: {stats['win_rate']:.2f}\n" +
             f"训练时间: {stats['total_training_time']/3600:.1f} 小时\n\n" +
             f"动作统计:\n" +
             "\n".join([f"{k}: {v}" for k, v in stats['action_counts'].items()]),
             ha='center', va='center', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_summary_{timestamp}.png"))
    plt.close()
    
    print(f"已生成并保存监控图表到 {output_dir}")
    return True

def monitor_training(stats_dir, output_dir, refresh_interval=300):
    """定期监控训练状态并生成图表"""
    print(f"开始监控训练... (刷新间隔: {refresh_interval}秒)")
    print(f"统计数据目录: {stats_dir}")
    print(f"图表输出目录: {output_dir}")
    
    try:
        while True:
            if generate_plots(stats_dir, output_dir):
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 已更新监控图表")
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("监控已停止")

if __name__ == "__main__":
    setup_chinese_font()
    parser = argparse.ArgumentParser(description="麻将PPO训练监控 (无图形界面版)")
    parser.add_argument("--stats_dir", type=str, default="stats",
                      help="统计数据目录")
    parser.add_argument("--output_dir", type=str, default="monitor_outputs",
                      help="图表输出目录")
    parser.add_argument("--refresh", type=int, default=300,
                      help="刷新间隔(秒)")
    
    args = parser.parse_args()
    monitor_training(args.stats_dir, args.output_dir, args.refresh)
