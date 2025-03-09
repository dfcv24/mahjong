import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import os
from src.models.model import MahjongTotalSingleModel
from src.utils.constants import *
import copy

class MahjongFeatureExtractor(nn.Module):
    """从预训练模型提取的特征提取器"""
    def __init__(self, pretrained_model):
        super(MahjongFeatureExtractor, self).__init__()
        # 复制预训练模型的嵌入层
        self.tile_embed = copy.deepcopy(pretrained_model.tile_embed)
        self.rush_tile_embed = copy.deepcopy(pretrained_model.rush_tile_embed)
        self.turn_embed = copy.deepcopy(pretrained_model.turn_embed)
        self.pos_embed = copy.deepcopy(pretrained_model.pos_embed)
        self.feature_type_embed = copy.deepcopy(pretrained_model.feature_type_embed)
        self.embed_dropout = copy.deepcopy(pretrained_model.embed_dropout)
        
        # 复制Transformer编码器
        self.encoder = copy.deepcopy(pretrained_model.encoder)
        
        # 输入特征维度
        self.input_size = pretrained_model.input_size
    
    def forward(self, features, rush_tile_id, turn):
        batch_size = features.size(0)
        actual_input_size = features.size(1)
    
        # 嵌入所有特征
        tile_embeddings = self.tile_embed(features)  # [batch, input_size, d_model]
        rush_tile_embedding = self.rush_tile_embed(rush_tile_id).unsqueeze(1)  # [batch, 1, d_model]
        
        # 回合处理（分桶）
        boundaries = torch.tensor([4,8,12,16,20,24,28,32], device=features.device)
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
        
        # 生成注意力掩码，处理填充值(NUM_TILE_TYPES)
        padding_mask = (features == NUM_TILE_TYPES)  # [batch, input_size]
        
        # 为rush_tile和回合token添加False（不掩盖）
        padding_mask = torch.cat([
            padding_mask, 
            torch.zeros(batch_size, 2, dtype=torch.bool, device=features.device)
        ], dim=1)  # [batch, input_size+2]
        
        # 通过Transformer编码器
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)  # [batch, input_size+2, d_model]
        
        # 使用rush_tile和回合位置的输出作为决策基础
        # 这里使用了一个简单的平均池化来融合这两个token的信息
        decision_features = (encoded[:, -2, :] + encoded[:, -1, :]) / 2  # [batch, d_model]
        
        return decision_features

class MahjongPolicyNetwork(nn.Module):
    """策略网络：基于预训练模型的动作头"""
    def __init__(self, pretrained_model, feature_dim=128):
        super(MahjongPolicyNetwork, self).__init__()
        # 参考预训练模型的统一动作头结构创建新的动作头
        self.action_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim*2, pretrained_model.unified_action_head[-1].out_features)  # 保持与预训练模型相同的输出维度
        )
        
        # 如果可能，复制预训练模型的权重
        try:
            # 尝试加载预训练模型的权重
            self.action_head[0].weight.data.copy_(pretrained_model.unified_action_head[0].weight.data)
            self.action_head[0].bias.data.copy_(pretrained_model.unified_action_head[0].bias.data)
            self.action_head[3].weight.data.copy_(pretrained_model.unified_action_head[3].weight.data)
            self.action_head[3].bias.data.copy_(pretrained_model.unified_action_head[3].bias.data)
        except Exception as e:
            print(f"无法复制预训练模型权重: {e}")
            print("使用随机初始化的动作头")
    
    def forward(self, features, action_mask=None):
        # 获取动作的logits
        logits = self.action_head(features)
        
        # 应用动作掩码
        if action_mask is not None:
            # 将掩码中的False位置设为一个很小的值
            logits = logits + ((1 - action_mask.float()) * -1e9)
        
        return logits
    
    def get_action_and_log_prob(self, features, action_mask=None):
        # 获取动作的logits
        logits = self.forward(features, action_mask)
        
        # 创建概率分布
        action_probs = F.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        
        # 采样一个动作
        action = dist.sample()
        
        # 计算log概率
        log_prob = dist.log_prob(action)
        
        return action, log_prob, action_probs

class MahjongValueNetwork(nn.Module):
    """价值网络：估计状态值函数"""
    def __init__(self, feature_dim=128):
        super(MahjongValueNetwork, self).__init__()
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, features):
        return self.value_head(features)

class MahjongPPO:
    def __init__(self, state_dim=198, action_dim=44, device="cuda", lr_actor=3e-4, lr_critic=1e-3, pretrained_model_path=None, freeze_initial=True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"PPO运行在设备: {self.device}")
        
        # 添加预训练模型路径参数
        self.pretrained_model_path = pretrained_model_path
        self.freeze_initial = freeze_initial
        
        # 存储学习率
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        
        # 加载预训练模型
        self.pretrained_model = self._load_pretrained_model()
        
        # 特征提取器
        self.feature_extractor = MahjongFeatureExtractor(self.pretrained_model).to(self.device)
        
        # 策略网络 (actor)
        self.policy = MahjongPolicyNetwork(self.pretrained_model).to(self.device)
        
        # 价值网络 (critic)
        self.value = MahjongValueNetwork().to(self.device)
        
        # 旧策略网络 (用于计算重要性采样比率)
        self.old_policy = copy.deepcopy(self.policy).to(self.device)
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(list(self.feature_extractor.parameters()) + 
                                                list(self.policy.parameters()), lr=lr_actor)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_critic)
        
        # PPO超参数
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # 初始阶段冻结参数
        if self.freeze_initial:
            self.freeze_feature_extractor_policy()
            print("初始阶段已冻结特征提取器和策略网络，只训练价值网络")
        
    def _load_pretrained_model(self):
        """加载预训练模型"""
        # 使用传入的模型路径，如果未指定则使用默认路径
        if self.pretrained_model_path is None:
            # 尝试查找默认路径
            possible_paths = [
                "models/mahjong_total_single_best.pth",
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                             "models/mahjong_total_single_best.pth")
            ]
            
            model_path = None
            for path in possible_paths:
                if (os.path.exists(path)):
                    model_path = path
                    break
            
            if model_path is None:
                print("找不到预训练模型，将使用随机初始化的模型")
                # 创建一个随机初始化的模型
                model = MahjongTotalSingleModel(
                    dropout_rate=0.1,
                    input_size=198,
                    output_size=44
                )
                return model
        else:
            model_path = self.pretrained_model_path
            
            # 确保模型文件存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"找不到预训练模型: {model_path}")
        
        # 初始化模型
        model = MahjongTotalSingleModel(
            dropout_rate=0.1,
            input_size=198,
            output_size=44
        )
        
        try:
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载预训练模型 {model_path}，准确率: {checkpoint.get('accuracy', 'unknown'):.4f}")
        except Exception as e:
            print(f"加载预训练模型出错: {e}")
            print("将使用随机初始化的模型")
        
        # 将模型设为评估模式并冻结参数
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        return model
    
    def select_action(self, features, rush_tile_id, turn, action_mask):
        """选择一个动作，并返回对应的log概率"""
        with torch.no_grad():
            # 特征提取
            combined_features = self.feature_extractor(features, rush_tile_id, turn)
            
            # 从旧策略中采样动作
            action, log_prob, probs = self.old_policy.get_action_and_log_prob(combined_features, action_mask)
            
            # 估计状态值
            state_value = self.value(combined_features)
        
        return action.item(), log_prob, state_value, combined_features
    
    # 添加处理不同动作空间的方法
    def select_action_with_dynamic_space(self, features, rush_tile_id, turn, action_mask, is_rush_action=False):
        """选择一个动作，并返回对应的log概率，支持动态动作空间"""
        with torch.no_grad():
            # 特征提取
            combined_features = self.feature_extractor(features, rush_tile_id, turn)
            
            # 根据动作空间类型处理
            if is_rush_action:
                # 针对吃碰杠胡动作，需要适配策略网络输出
                # 假设动作空间为5维 [过, 吃, 碰, 杠, 胡]
                # 这里需要确保action_mask的形状正确
                # 可以考虑添加一个专门针对rush动作的策略头
                pass
            
            # 从旧策略中采样动作
            action, log_prob, probs = self.old_policy.get_action_and_log_prob(combined_features, action_mask)
            
            # 估计状态值
            state_value = self.value(combined_features)
        
        return action.item(), log_prob, state_value, combined_features
    
    def update_policy(self, memory, batch_size=64, epochs=10, return_stats=False):
        """使用收集的经验更新策略和价值网络"""
        # 从记忆中提取数据
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.tensor(memory.actions, dtype=torch.long).to(self.device).detach()
        old_log_probs = torch.stack(memory.log_probs).to(self.device).detach()
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(self.device).detach()
        masks = torch.tensor(memory.masks, dtype=torch.float32).to(self.device).detach()
        old_state_values = torch.stack(memory.values).squeeze().to(self.device).detach()
        action_masks = torch.stack(memory.action_masks).to(self.device).detach()
        
        # 用于统计的变量
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy = 0
        updates = 0
        
        # 计算GAE(Generalized Advantage Estimation)
        advantages = []
        gae = 0
        with torch.no_grad():
            for i in reversed(range(len(rewards))):
                if i == len(rewards) - 1:
                    next_value = 0  # 终止状态的价值
                else:
                    next_value = old_state_values[i + 1]
                
                delta = rewards[i] + self.gamma * next_value * masks[i] - old_state_values[i]
                gae = delta + self.gamma * self.gae_lambda * masks[i] * gae
                advantages.insert(0, gae)
            
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
            returns = advantages + old_state_values
            
            # 标准化优势函数
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 更新策略和价值网络
        for _ in range(epochs):
            # 生成随机索引
            indices = torch.randperm(len(old_states))
            
            # 分批次训练
            for start in range(0, len(old_states), batch_size):
                end = start + batch_size
                if end > len(old_states):
                    end = len(old_states)
                batch_indices = indices[start:end]
                
                # 特征提取（这里不同于select_action，这里需要梯度）
                batch_states = old_states[batch_indices]
                
                # 获取当前策略的log概率
                current_logits = self.policy(batch_states, action_masks[batch_indices])
                current_probs = F.softmax(current_logits, dim=-1)
                dist = Categorical(current_probs)
                current_log_probs = dist.log_prob(old_actions[batch_indices])
                
                # 计算价值损失
                current_values = self.value(batch_states).squeeze()
                value_loss = F.mse_loss(current_values, returns[batch_indices])
                
                # 计算熵
                entropy = dist.entropy().mean()
                
                # 计算重要性采样比率
                ratios = torch.exp(current_log_probs - old_log_probs[batch_indices])
                
                # 计算PPO损失
                batch_advantages = advantages[batch_indices]
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 更新网络 - 根据参数冻结状态来选择性更新
                # 检查策略网络是否有需要更新的参数
                has_policy_params = any(p.requires_grad for p in self.policy.parameters()) or any(p.requires_grad for p in self.feature_extractor.parameters())
                
                if has_policy_params:
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward(retain_graph=True)  # 使用retain_graph=True以允许多次反向传播
                    self.policy_optimizer.step()
                
                # 价值网络始终更新
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                
                # 累积统计数据
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy += entropy.item()
                updates += 1
        
        # 计算平均值
        avg_policy_loss /= max(updates, 1)
        avg_value_loss /= max(updates, 1)
        avg_entropy /= max(updates, 1)
        
        # 如果需要返回统计数据
        if return_stats:
            return {
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'entropy': avg_entropy,
                'learning_rate': self.policy_optimizer.param_groups[0]['lr']
            }
        # 添加默认返回值，以防止不需要统计数据时出错
        return None
    
    def copy_policy_to_old_policy(self):
        """将当前策略复制到旧策略"""
        self.old_policy.load_state_dict(self.policy.state_dict())
    
    def freeze_feature_extractor_policy(self):
        """冻结特征提取器和策略网络的参数"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        for param in self.policy.parameters():
            param.requires_grad = False
        
        # 修复: 不要尝试创建空的优化器，只需要将现有优化器中的参数冻结即可
        print("特征提取器和策略网络已冻结，优化器仅更新价值网络参数")
    
    def unfreeze_feature_extractor_policy(self):
        """解冻特征提取器和策略网络的参数"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        
        for param in self.policy.parameters():
            param.requires_grad = True
        
        # 重新创建包含所有需要训练参数的优化器
        self.policy_optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.policy.parameters()),
            lr=self.lr_actor
        )
        print("特征提取器和策略网络已解冻")
    
    def unfreeze_policy_only(self):
        """只解冻策略网络的参数，特征提取器保持冻结"""
        for param in self.policy.parameters():
            param.requires_grad = True
        
        # 重新创建只包含策略网络参数的优化器
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.lr_actor
        )
        print("策略网络已解冻，特征提取器保持冻结")

class PPOMemory:
    """经验回放缓存"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.action_masks = []
    
    def push(self, state, action, log_prob, reward, value, mask, action_mask):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.masks.append(mask)
        self.action_masks.append(action_mask)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.masks.clear()
        self.action_masks.clear()
        
    def __len__(self):
        return len(self.states)