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
from ..models.config import D_MODEL, NHEAD, NUM_ENCODER_LAYERS
from ..models.model import MahjongTotalModel

class MahjongFeatureExtractor(BaseFeaturesExtractor):
    """
    提取麻将游戏特征的自定义特征提取器，结构与MahjongTotalModel一致
    """
    def __init__(self, observation_space, features_dim=128, input_size=198):
        super(MahjongFeatureExtractor, self).__init__(observation_space, features_dim)
        
        self.input_size = input_size
        
        # 输入嵌入，与MahjongTotalModel保持一致
        self.tile_embed = nn.Embedding(NUM_TILE_TYPES + 1, D_MODEL, padding_idx=NUM_TILE_TYPES)
        self.rush_tile_embed = nn.Embedding(NUM_TILE_TYPES + 1, D_MODEL, padding_idx=NUM_TILE_TYPES)
        self.turn_embed = nn.Embedding(10, D_MODEL)
        self.pos_embed = nn.Embedding(input_size + 2, D_MODEL)  # +2是rush牌和回合token
        
        # 特征类型编码
        self.feature_type_embed = nn.Embedding(5, D_MODEL)  # 0=手牌, 1=rush牌, 2=hitout牌, 3=回合, 4=rush_tile
        
        # 添加嵌入后的dropout
        self.embed_dropout = nn.Dropout(0.1)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, 
            nhead=NHEAD,
            dim_feedforward=4*D_MODEL,
            batch_first=True,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, NUM_ENCODER_LAYERS)
        
        # 最终输出层
        self.final_layer = nn.Sequential(
            nn.Linear(D_MODEL, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        batch_size = observations['features'].size(0)
        features = observations['features']  # [batch, input_size]
        rush_tile = observations['rush_tile']  # [batch]
        turn = observations['turn']  # [batch]
        actual_input_size = features.size(1)
        
        # 嵌入所有特征
        tile_embeddings = self.tile_embed(features)  # [batch, input_size, d_model]
        rush_tile_embedding = self.rush_tile_embed(rush_tile).unsqueeze(1)  # [batch, 1, d_model]
        
        # 回合处理（分桶）
        boundaries = torch.tensor([4, 8, 12, 16, 20, 24, 28, 32], device=features.device)
        turn_bucket = torch.bucketize(turn, boundaries)
        turn_embeddings = self.turn_embed(turn_bucket).unsqueeze(1)  # (B, 1, D)
        
        # 准备位置编码
        positions = torch.arange(actual_input_size, device=features.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_embed(positions)  # [batch, actual_input_size, d_model]
        
        rush_tile_pos = self.pos_embed(torch.full((batch_size, 1), self.input_size, device=features.device))
        turn_pos = self.pos_embed(torch.full((batch_size, 1), self.input_size + 1, device=features.device))
        
        # 准备特征类型编码
        feature_types = torch.zeros(batch_size, self.input_size, dtype=torch.long, device=features.device)
        
        # 标记不同类型的特征
        # 当前玩家
        feature_types[:, :14] = 0  # 手牌
        feature_types[:, 14:30] = 1  # rush牌
        feature_types[:, 30:60] = 2  # hitout牌
        
        # 其他玩家 (3个玩家)
        for i in range(1, 4):
            start_idx = 60 * i
            feature_types[:, start_idx:start_idx+16] = 1  # rush牌
            feature_types[:, start_idx+16:start_idx+46] = 2  # hitout牌
        
        # 获取特征类型嵌入
        feature_type_embeddings = self.feature_type_embed(feature_types)  # [batch, input_size, d_model]
        
        # 组合手牌所有嵌入
        combined = tile_embeddings + pos_embeddings + feature_type_embeddings  # [batch, input_size, d_model]
        
        # 添加rush牌和回合的特征类型嵌入
        rush_tile_type = self.feature_type_embed(torch.full((batch_size, 1), 4, device=features.device))  # rush_tile类型
        turn_type = self.feature_type_embed(torch.full((batch_size, 1), 3, device=features.device))  # 回合类型
        
        # 组合rush牌和回合嵌入
        rush_tile_full = rush_tile_embedding + rush_tile_pos + rush_tile_type
        turn_full = turn_embeddings + turn_pos + turn_type
        
        # 合并所有特征
        sequence = torch.cat([combined, rush_tile_full, turn_full], dim=1)  # [batch, input_size+2, d_model]
        sequence = self.embed_dropout(sequence)
        
        # 生成注意力掩码
        padding_mask = (features == NUM_TILE_TYPES)  # [batch, input_size]
        
        # 为rush_tile和回合token添加False（不掩盖）
        padding_mask = torch.cat([
            padding_mask, 
            torch.zeros(batch_size, 2, dtype=torch.bool, device=features.device)
        ], dim=1)  # [batch, input_size+2]
        
        # 通过Transformer编码器
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)  # [batch, input_size+2, d_model]
        
        # 使用rush_tile和回合位置的输出作为决策基础
        decision_features = (encoded[:, -2, :] + encoded[:, -1, :]) / 2  # [batch, d_model]
        
        return self.final_layer(decision_features)


class MahjongPPOAgent:
    """
    使用 PPO 算法的麻将 AI 代理，支持加载预训练模型参数
    """
    def __init__(self, env, model_path=None, pretrained_model_path=None, learning_rate=3e-4):
        self.env = env
        policy_kwargs = {
            'features_extractor_class': MahjongFeatureExtractor,
            'features_extractor_kwargs': {'features_dim': 128, 'input_size': 198}
        }
        
        if model_path and os.path.exists(model_path):
            # 加载已有PPO模型
            self.model = PPO.load(model_path, env=env)
            print(f"加载已有PPO模型: {model_path}")
        else:
            # 创建新模型
            self.model = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs,
                           learning_rate=learning_rate, verbose=1)
            print("创建新的PPO模型")
            
            # 如果提供了预训练模型路径，加载预训练参数
            if pretrained_model_path and os.path.exists(pretrained_model_path):
                self.load_pretrained_parameters(pretrained_model_path)
    
    def load_pretrained_parameters(self, pretrained_model_path):
        """从预训练的MahjongTotalModel加载参数到PPO的特征提取器和策略网络"""
        try:
            # 加载预训练模型
            pretrained_model = MahjongTotalModel()
            pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
            print(f"加载预训练模型: {pretrained_model_path}")
            
            # 获取PPO的特征提取器
            ppo_feature_extractor = self.model.policy.features_extractor
            
            # 复制嵌入层参数
            ppo_feature_extractor.tile_embed.load_state_dict(pretrained_model.tile_embed.state_dict())
            ppo_feature_extractor.rush_tile_embed.load_state_dict(pretrained_model.rush_tile_embed.state_dict())
            ppo_feature_extractor.turn_embed.load_state_dict(pretrained_model.turn_embed.state_dict())
            ppo_feature_extractor.pos_embed.load_state_dict(pretrained_model.pos_embed.state_dict())
            ppo_feature_extractor.feature_type_embed.load_state_dict(pretrained_model.feature_type_embed.state_dict())
            
            # 复制Transformer编码器参数
            ppo_feature_extractor.encoder.load_state_dict(pretrained_model.encoder.state_dict())
            
            # 复制所有决策头参数到PPO的动作网络和价值网络
            try:
                # 决策头参数
                discard_head_params = pretrained_model.disacrd_head
                action_head_params = pretrained_model.action_head
                chi_head_params = pretrained_model.chi_head
                
                # 检查PPO模型的动作网络结构
                print("PPO动作网络结构:", self.model.policy.action_net)
                print("PPO值网络结构:", self.model.policy.value_net)
                
                # 初始化action_net的参数，使用出牌决策头参数
                with torch.no_grad():
                    # 复制第一层参数
                    if len(self.model.policy.action_net) >= 1 and hasattr(discard_head_params, '0'):
                        self.model.policy.action_net[0].weight.data.copy_(discard_head_params[0].weight.data)
                        self.model.policy.action_net[0].bias.data.copy_(discard_head_params[0].bias.data)
                        print("复制出牌头第一层参数到action_net")
                    
                    # 复制第二层参数(如果存在)
                    if len(self.model.policy.action_net) >= 3 and hasattr(discard_head_params, '3'):
                        self.model.policy.action_net[2].weight.data.copy_(discard_head_params[3].weight.data)
                        self.model.policy.action_net[2].bias.data.copy_(discard_head_params[3].bias.data)
                        print("复制出牌头第二层参数到action_net")

                    # 复制value_net的参数，使用相同的预训练参数以获得更好的值函数
                    if len(self.model.policy.value_net) >= 1:
                        self.model.policy.value_net[0].weight.data.copy_(discard_head_params[0].weight.data)
                        self.model.policy.value_net[0].bias.data.copy_(discard_head_params[0].bias.data)
                        print("复制出牌头参数到value_net")
                    
                    if len(self.model.policy.value_net) >= 3:
                        self.model.policy.value_net[2].weight.data.copy_(discard_head_params[3].weight.data)
                        self.model.policy.value_net[2].bias.data.copy_(discard_head_params[3].bias.data)
                        print("复制出牌头第二层参数到value_net")
                
                print("成功将预训练模型的所有决策头参数加载到PPO代理")
            
            except Exception as e:
                print(f"加载决策头参数时出错: {e}")
                import traceback
                traceback.print_exc()
            
            print("成功加载预训练模型参数到PPO代理")
        except Exception as e:
            print(f"加载预训练模型参数失败: {e}")
    
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