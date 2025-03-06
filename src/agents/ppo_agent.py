# src/agents/ppo_agent.py

import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import torch.nn as nn
from ..utils.constants import NUM_TILE_TYPES

class MahjongFeatureExtractor(BaseFeaturesExtractor):
    """
    提取麻将游戏特征的自定义特征提取器
    """
    def __init__(self, observation_space, features_dim=128):
        super(MahjongFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # 定义每个观察类型的嵌入维度
        hand_embedding_dim = 64
        turn_embedding_dim = 16
        discard_embedding_dim = 32
        last_discard_embedding_dim = 16
        
        # 手牌嵌入
        self.hand_embedding = nn.Sequential(
            nn.Embedding(NUM_TILE_TYPES + 1, hand_embedding_dim, padding_idx=NUM_TILE_TYPES),
            nn.Flatten(),
            nn.Linear(14 * hand_embedding_dim, 128),
            nn.ReLU()
        )
        
        # 回合数嵌入
        self.turn_embedding = nn.Embedding(50, turn_embedding_dim)
        
        # 玩家出牌历史嵌入
        self.discard_embedding = nn.Sequential(
            nn.Embedding(NUM_TILE_TYPES + 1, 8, padding_idx=NUM_TILE_TYPES),
            nn.Flatten(),
            nn.Linear(3 * 20 * 8, 64),
            nn.ReLU()
        )
        
        # 最后出牌嵌入
        self.last_discard_embedding = nn.Embedding(NUM_TILE_TYPES + 1, last_discard_embedding_dim)
        
        # 组合所有特征
        combined_dim = 128 + turn_embedding_dim + 64 + last_discard_embedding_dim
        self.final_layer = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        # 提取每个部分的特征
        hand_features = self.hand_embedding(observations['hand'].long())
        turn_features = self.turn_embedding(observations['turn'].long()).squeeze(1)
        discard_features = self.discard_embedding(observations['player_discards'].long())
        last_discard_features = self.last_discard_embedding(observations['last_discard'].long()).squeeze(1)
        
        # 组合特征
        combined_features = torch.cat([
            hand_features, 
            turn_features, 
            discard_features, 
            last_discard_features
        ], dim=1)
        
        return self.final_layer(combined_features)

class MahjongPPOAgent:
    """
    使用 PPO 算法的麻将 AI 代理
    """
    def __init__(self, env, model_path=None, learning_rate=3e-4):
        self.env = env
        policy_kwargs = {
            'features_extractor_class': MahjongFeatureExtractor,
            'features_extractor_kwargs': {'features_dim': 128}
        }
        
        if model_path and os.path.exists(model_path):
            # 加载已有模型
            self.model = PPO.load(model_path, env=env)
            print(f"加载已有PPO模型: {model_path}")
        else:
            # 创建新模型
            self.model = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs,
                           learning_rate=learning_rate, verbose=1)
            print("创建新的PPO模型")
    
    def train(self, total_timesteps=100000, log_interval=10):
        """训练模型"""
        print(f"开始PPO训练，总时间步数: {total_timesteps}")
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        
    def save(self, path):
        """保存模型"""
        self.model.save(path)
        print(f"模型已保存至: {path}")
    
    def predict(self, observation, deterministic=True):
        """预测动作"""
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action