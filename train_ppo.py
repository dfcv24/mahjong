# train_ppo.py

import os
import torch
import numpy as np
from src.environment.mahjong_env import MahjongEnv
from src.agents.ppo_agent import MahjongPPOAgent
from src.models.model import MahjongTotalModel
import json
import datetime
import argparse

def load_existing_model(model_path, device):
    """加载现有的麻将全能模型"""
    print(f"加载现有模型: {model_path}")
    model = MahjongTotalModel().to(device)
    
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
    
    # 训练代理
    agent.train(total_timesteps=total_timesteps)
    
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