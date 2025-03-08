# train_ppo.py

import os
import torch
import numpy as np
from src.environment.mahjong_env import MahjongEnv
from src.agents.ppo_agent import MahjongPPOAgent, MaskablePPO
from src.models.model import MahjongTotalSingleModel
import json
import datetime
import argparse
from stable_baselines3.common.callbacks import BaseCallback

class MahjongActionMaskCallback(BaseCallback):
    """监控有效动作选择的回调函数"""
    
    def __init__(self, verbose=0):
        super(MahjongActionMaskCallback, self).__init__(verbose)
        self.valid_actions = 0
        self.total_actions = 0
        self.completed_games = 0
        self.rewards_history = []
    
    def _on_step(self) -> bool:
        """每一步调用一次此方法"""
        try:
            # 尝试获取当前观察和动作
            if hasattr(self.model, 'env') and hasattr(self.model.env, 'get_attr'):
                # 获取游戏完成状态
                if 'games_played' in self.locals.get('infos', [{}])[0]:
                    games_count = self.locals['infos'][0]['games_played']
                    if games_count > self.completed_games:
                        # 新游戏完成
                        new_games = games_count - self.completed_games
                        self.completed_games = games_count
                        
                        # 记录新完成游戏的奖励
                        if 'episode_reward' in self.locals['infos'][0]:
                            reward = self.locals['infos'][0]['episode_reward']
                            self.rewards_history.append(reward)
                            
                            # 计算和打印平均奖励
                            avg_reward = sum(self.rewards_history[-100:]) / min(100, len(self.rewards_history))
                            print(f"完成 {self.completed_games} 场游戏，最近100场平均奖励: {avg_reward:.2f}")
                
                # 检查有效动作
                if len(self.model.env.get_attr('current_obs')) > 0:
                    obs = self.model.env.get_attr('current_obs')[0]
                    action_mask = obs[200:244]
                    if 'actions' in self.locals and len(self.locals['actions']) > 0:
                        last_action = self.locals['actions'][0]
                        
                        # 统计有效动作
                        self.total_actions += 1
                        if action_mask[last_action] == 1:
                            self.valid_actions += 1
                        
                        # 每1000步打印一次统计信息
                        if self.num_timesteps % 1000 == 0:
                            valid_rate = self.valid_actions / max(1, self.total_actions)
                            print(f"有效动作率: {valid_rate:.4f} ({self.valid_actions}/{self.total_actions})")
        except Exception as e:
            # 如果获取失败，输出错误但不中断训练
            print(f"回调函数出错: {e}")
            import traceback
            traceback.print_exc()
        
        return True
    
    def on_training_end(self) -> None:
        """训练结束时调用此方法"""
        valid_rate = self.valid_actions / max(1, self.total_actions)
        print(f"训练结束，最终有效动作率: {valid_rate:.4f} ({self.valid_actions}/{self.total_actions})")
        print(f"总共完成 {self.completed_games} 场游戏")
        if self.rewards_history:
            avg_reward = sum(self.rewards_history) / len(self.rewards_history)
            print(f"平均每局奖励: {avg_reward:.2f}")

def load_existing_model(model_path, device):
    """加载现有的麻将单一模型"""
    print(f"加载现有模型: {model_path}")
    model = MahjongTotalSingleModel().to(device)
    
    try:
        # 尝试加载完整的checkpoint（包含优化器状态等）
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 直接加载模型状态字典
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"模型加载异常: {e}, 尝试直接加载状态字典")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()  # 设置为评估模式
    return model

def train_ppo(pretrained_model_path=None, ppo_model_path=None, 
              total_timesteps=100000, learning_rate=3e-4):
    """使用PPO训练麻将AI代理"""
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", "ppo_training", f"ppo_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建麻将环境
    env = MahjongEnv(server_ip="127.0.0.1", server_port=5000)
    
    # 创建PPO代理
    agent = MahjongPPOAgent(
        env, 
        model_path=ppo_model_path, 
        pretrained_model_path=pretrained_model_path,
        learning_rate=learning_rate
    )
    
    # 创建监控回调
    callback = MahjongActionMaskCallback()
    
    # 训练代理
    agent.train(total_timesteps=total_timesteps, callback=callback)
    
    # 保存训练后的模型
    output_model_path = os.path.join(results_dir, "mahjong_ppo_model.zip")
    agent.save(output_model_path)
    
    # 记录训练配置
    config = {
        "pretrained_model_path": pretrained_model_path,
        "ppo_model_path": ppo_model_path,
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "timestamp": timestamp,
        "device": str(device)
    }
    
    with open(os.path.join(results_dir, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"训练完成，模型已保存至: {output_model_path}")
    return output_model_path

def main():
    parser = argparse.ArgumentParser(description="使用PPO训练麻将AI")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="预训练的MahjongTotalModel路径")
    parser.add_argument("--ppo_model", type=str, default=None,
                        help="已有PPO模型的路径，用于继续训练")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="训练总时间步数")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="学习率")
    
    args = parser.parse_args()
    
    train_ppo(
        pretrained_model_path=args.pretrained_model,
        ppo_model_path=args.ppo_model,
        total_timesteps=args.timesteps,
        learning_rate=args.lr
    )

if __name__ == "__main__":
    main()