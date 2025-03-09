import argparse
import os
import logging
import torch
# 修正导入路径
from src.client.client_env import MahjongClient
from src.agents.new_ppo_agent import MahjongPPO
from src.utils.train_utils import setup_chinese_font
from datetime import datetime
import time
import json
import matplotlib.pyplot as plt
import numpy as np

# 配置 matplotlib 使用 Agg 后端以支持无显示器环境
import matplotlib
matplotlib.use('Agg')

setup_chinese_font()

# 配置日志
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PPO-Training")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="麻将 PPO 训练脚本")
    
    parser.add_argument("--server_ip", type=str, default="127.0.0.1",
                        help="游戏服务器IP地址")
    parser.add_argument("--server_port", type=int, default=5000,
                        help="游戏服务器端口")
    parser.add_argument("--load_model", type=str, default=None,
                        help="要加载的模型文件路径 (如果有)")
    parser.add_argument("--save_dir", type=str, default="models/ppo_models/20252348",
                        help="模型保存目录")
    parser.add_argument("--update_interval", type=int, default=10,
                        help="模型参数更新的回合间隔")
    parser.add_argument("--save_interval", type=int, default=200,
                        help="保存模型的回合间隔")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="训练设备 (cuda 或 cpu)")
    parser.add_argument("--lr_actor", type=float, default=1e-4,
                        help="策略网络学习率")
    parser.add_argument("--lr_critic", type=float, default=1e-3,
                        help="价值网络学习率")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="批量训练大小")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="预训练模型路径 (如果不提供，将使用默认路径)")
    parser.add_argument("--log_interval", type=int, default=1,
                        help="每隔多少局打印一次日志")
    parser.add_argument("--plot_interval", type=int, default=10,
                        help="每隔多少局绘制一次图表")
    parser.add_argument("--stats_dir", type=str, default="stats/20252348",
                        help="统计数据保存目录")
    parser.add_argument("--no_plot", action="store_true",
                        help="设置此标志禁用绘图功能")
    parser.add_argument("--unfreeze_policy_at", type=int, default=1000,
                        help="解冻策略网络的回合数")
    parser.add_argument("--unfreeze_all_at", type=int, default=2000,
                        help="解冻所有参数的回合数")
    parser.add_argument("--freeze_initial", action="store_true", default=True,
                        help="是否在初始阶段冻结特征提取器和策略网络")
    
    return parser.parse_args()

def setup_environment(args):
    """设置训练环境"""
    # 确保模型保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"模型将保存到 {args.save_dir}")
    
    # 设置设备
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    # 创建统计数据目录
    os.makedirs(args.stats_dir, exist_ok=True)
    logger.info(f"统计数据将保存到 {args.stats_dir}")
    
    return device

def plot_training_metrics(client, args, timestamp):
    """绘制训练指标图表"""
    if args.no_plot:
        return
    
    # 获取所有训练统计数据
    stats = client.get_training_stats()
    
    # 创建图表目录
    plot_dir = os.path.join(args.stats_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    try:
        # 绘制奖励图
        plt.figure(figsize=(12, 6))
        # 每10局取一个平均值，最后不到10局的取平均
        avg_rewards = [np.mean(stats['episode_rewards'][i:i+10]) for i in range(0, len(stats['episode_rewards']), 10)]
        avg_rewards += [np.mean(stats['episode_rewards'][-10:])]
        plt.plot(avg_rewards)
        plt.title("游戏奖励")
        plt.xlabel("游戏局数")
        plt.ylabel("平均10局奖励")
        # plt.ylabel("奖励")
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f"rewards_{timestamp}.png"))
        plt.close()
        
        # 绘制游戏长度图
        plt.figure(figsize=(12, 6))
        plt.plot(stats['episode_lengths'])
        plt.title("游戏回合数")
        plt.xlabel("游戏局数")
        plt.ylabel("回合数")
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f"game_lengths_{timestamp}.png"))
        plt.close()
        
        # 绘制PPO损失
        if stats['ppo_stats']['policy_loss']:
            plt.figure(figsize=(12, 10))
            plt.subplot(3, 1, 1)
            plt.plot(stats['ppo_stats']['policy_loss'])
            plt.title("策略损失")
            plt.grid(True)
            
            plt.subplot(3, 1, 2)
            plt.plot(stats['ppo_stats']['value_loss'])
            plt.title("价值损失")
            plt.grid(True)
            
            plt.subplot(3, 1, 3)
            plt.plot(stats['ppo_stats']['entropy'])
            plt.title("熵")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"ppo_losses_{timestamp}.png"))
            plt.close()
        
        # 绘制动作分布
        action_labels = list(stats['action_counts'].keys())
        action_values = list(stats['action_counts'].values())
        plt.figure(figsize=(12, 6))
        plt.bar(action_labels, action_values)
        plt.title("动作分布")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"action_distribution_{timestamp}.png"))
        plt.close()
        
        # 绘制胜率趋势
        if len(client.recent_win_rate) > 0:
            win_rates = []
            window_size = min(10, len(stats['episode_rewards']))
            for i in range(window_size, len(stats['episode_rewards'])+1):
                # 计算滑动窗口胜率
                win_count = sum(1 for j in range(i-window_size, i) 
                              if stats['episode_rewards'][j] > 0)
                win_rates.append(win_count/window_size)
            
            plt.figure(figsize=(12, 6))
            plt.plot(range(window_size, len(stats['episode_rewards'])+1), win_rates)
            plt.title(f"胜率趋势 (窗口大小: {window_size})")
            plt.xlabel("游戏局数")
            plt.ylabel("胜率")
            plt.grid(True)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(plot_dir, f"win_rate_trend_{timestamp}.png"))
            plt.close()
        
        logger.info(f"已保存训练指标图表到 {plot_dir}")
    except Exception as e:
        logger.error(f"绘制图表时出错: {e}")

def save_training_stats(client, args, timestamp):
    """保存训练统计数据到JSON文件"""
    stats = client.get_training_stats()
    
    # 将不可序列化的数据转换为列表
    for key in stats['ppo_stats']:
        stats['ppo_stats'][key] = [float(x) for x in stats['ppo_stats'][key]]
    
    stats['episode_rewards'] = [float(x) for x in stats['episode_rewards']]
    stats['episode_lengths'] = [int(x) for x in stats['episode_lengths']]
    
    # 添加时间戳
    stats['timestamp'] = timestamp
    stats['total_training_time_hours'] = float(stats['total_training_time'] / 3600)
    
    # 保存到JSON文件
    stats_file = os.path.join(args.stats_dir, f"training_stats_{timestamp}.json")
    try:
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        logger.info(f"训练统计数据已保存到 {stats_file}")
    except Exception as e:
        logger.error(f"保存统计数据时出错: {e}")

def main():
    """主函数：解析参数并启动训练"""
    args = parse_arguments()
    device = setup_environment(args)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    try:
        # 自定义PPO智能体参数
        agent = MahjongPPO(
            device=args.device,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            pretrained_model_path=args.pretrained_model_path,
            freeze_initial=args.freeze_initial  # 设置是否初始冻结参数
        )
        
        # 创建客户端
        client = MahjongClient(
            server_ip=args.server_ip,
            server_port=args.server_port,
            agent=agent,
            logger=logger
        )
        
        # 设置模型保存间隔
        client.update_frequency = args.update_interval
        client.save_frequency = args.save_interval
        
        # 定制模型保存路径
        original_save_model = client.save_model
        
        def custom_save_model(filepath):
            # 使用指定目录保存模型
            actual_filepath = os.path.join(args.save_dir, os.path.basename(filepath))
            original_save_model(actual_filepath)
        
        # 替换保存模型方法
        client.save_model = custom_save_model
        
        # 在start之前确保已经连接
        # if not client.connect_to_server():
        #     logger.error("无法连接到服务器，训练终止")
        #     return
            
        # 加载模型如果有指定
        if args.load_model and os.path.exists(args.load_model):
            client.load_model(args.load_model)
            logger.info(f"成功加载模型: {args.load_model}")
            
        # 启动游戏
        player_name = "PPOAgent"
        logger.info("正在启动麻将PPO训练...")
        logger.info(f"服务器: {args.server_ip}:{args.server_port}")
        
        if client.start(args.load_model):
            logger.info("麻将客户端启动成功，开始训练...")
            client.send_play_game(player_name)
            logger.info("已发送游戏启动请求")
            
            # 在主线程中等待，可以通过键盘中断来停止
            last_plot_episode = 0
            try:
                while True:
                    # 打印训练信息
                    if client.episode_count > 0:
                        # 阶段性解冻参数
                        if args.freeze_initial:
                            # 第一阶段解冻：只解冻策略网络
                            if client.episode_count == args.unfreeze_policy_at:
                                logger.info(f"已达到 {args.unfreeze_policy_at} 回合，解冻策略网络")
                                agent.unfreeze_policy_only()
                            # 第二阶段解冻：解冻所有网络
                            elif client.episode_count == args.unfreeze_all_at:
                                logger.info(f"已达到 {args.unfreeze_all_at} 回合，解冻所有参数")
                                agent.unfreeze_feature_extractor_policy()
                        
                        # 定期打印日志
                        if client.episode_count % args.log_interval == 0:
                            # 计算平均奖励和胜率
                            avg_reward = sum(client.recent_rewards) / len(client.recent_rewards) if client.recent_rewards else 0
                            win_rate = sum(client.recent_win_rate) / len(client.recent_win_rate) if client.recent_win_rate else 0
                        
                            logger.info(f"游戏局数: {client.episode_count}, "
                                        f"平均奖励: {avg_reward:.2f}, "
                                        f"胜率: {win_rate:.2f}, "
                                        f"训练时间: {client.total_training_time/3600:.2f}小时")
                    
                    # 生成训练图表
                    if client.episode_count - last_plot_episode >= args.plot_interval:
                        plot_training_metrics(client, args, timestamp)
                        save_training_stats(client, args, timestamp)
                        last_plot_episode = client.episode_count
                    
                    # 检查游戏是否完成，如果完成则重新开始
                    if client.game_completed:
                        logger.debug("游戏已完成，重新开始...")
                        client.send_play_game(player_name)
                        client.game_completed = False
                    
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("接收到停止信号，正在关闭...")
            finally:
                # 最终保存统计数据和图表
                plot_training_metrics(client, args, timestamp)
                save_training_stats(client, args, timestamp)
                
                # 停止客户端 - 修复：使用close而不是stop
                client.close()
                
                # 保存最终模型
                final_model_path = os.path.join(args.save_dir, f"mahjong_ppo_final_{timestamp}.pt")
                client.save_model(final_model_path)
                logger.info(f"训练完成，最终模型保存至 {final_model_path}")
        else:
            logger.error("麻将客户端启动失败")
    
    except Exception as e:
        logger.exception(f"训练过程中发生错误: {e}")

if __name__ == "__main__":
    main()
