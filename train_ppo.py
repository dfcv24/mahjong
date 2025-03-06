# train_ppo.py

import os
import torch
import numpy as np
from src.environment.mahjong_env import MahjongGymEnv
from src.agents.ppo_agent import MahjongPPOAgent
from src.models.model import MahjongDiscardModel
import json
import datetime
import argparse

def load_existing_model(model_path, device):
    """加载现有的出牌决策模型"""
    print(f"加载现有模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = MahjongDiscardModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    return model

def train_ppo(existing_model_path=None, ppo_model_path=None, 
              total_timesteps=100000, learning_rate=3e-4):
    """使用PPO训练麻将AI代理"""
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", "ppo_training", f"ppo_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载现有模型作为指导
    guide_model = None
    if existing_model_path and os.path.exists(existing_model_path):
        guide_model = load_existing_model(existing_model_path, device)
    
    # 创建麻将环境
    env = MahjongGymEnv(model=guide_model, device=device)
    
    # 创建PPO代理
    agent = MahjongPPOAgent(env, model_path=ppo_model_path, learning_rate=learning_rate)
    
    # 训练代理
    agent.train(total_timesteps=total_timesteps)
    
    # 保存训练后的模型
    output_model_path = os.path.join(results_dir, "mahjong_ppo_model.zip")
    agent.save(output_model_path)
    
    # 记录训练配置
    config = {
        "existing_model_path": existing_model_path,
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
    parser.add_argument("--existing_model", type=str, default=None,
                        help="现有麻将模型的路径，用作指导")
    parser.add_argument("--ppo_model", type=str, default=None,
                        help="已有PPO模型的路径，用于继续训练")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="训练总时间步数")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="学习率")
    
    args = parser.parse_args()
    
    train_ppo(
        existing_model_path=args.existing_model,
        ppo_model_path=args.ppo_model,
        total_timesteps=args.timesteps,
        learning_rate=args.lr
    )

if __name__ == "__main__":
    main()