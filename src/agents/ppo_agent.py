# src/agents/ppo_agent.py

import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.policies import ActorCriticCnnPolicy
from gym import spaces
import torch.nn as nn
from ..utils.constants import NUM_TILE_TYPES
from ..models.config import D_MODEL, NHEAD, NUM_ENCODER_LAYERS
from ..models.model import MahjongTotalSingleModel

class MahjongFeatureExtractor(BaseFeaturesExtractor):
    """
    提取麻将游戏特征的自定义特征提取器，结构与MahjongTotalSingleModel一致
    """
    def __init__(self, observation_space, features_dim=D_MODEL):
        super(MahjongFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # 输入嵌入，与MahjongTotalSingleModel保持一致
        self.tile_embed = nn.Embedding(NUM_TILE_TYPES + 1, D_MODEL, padding_idx=NUM_TILE_TYPES)
        self.rush_tile_embed = nn.Embedding(NUM_TILE_TYPES + 1, D_MODEL, padding_idx=NUM_TILE_TYPES)  # 新增rush嵌入
        self.turn_embed = nn.Embedding(10, D_MODEL)  # 新增turn嵌入，支持最多10轮
        self.pos_embed = nn.Embedding(200, D_MODEL)  # 只处理前200维特征，不包括掩码
        
        # 特征类型编码 - 确保与MahjongTotalSingleModel兼容
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
        
    def forward(self, observations):
        # observations是244维向量，但我们只处理前200维（不包括动作掩码）
        batch_size = observations.shape[0]
        device = observations.device

        # 拆分观察向量，只取前200维
        features = observations[:, :198]  # 前198维是特征向量
        rush_tile = observations[:, 198:199]  # 第199维是rush牌
        turn = observations[:, 199:200]  # 第200维是轮数
        # 不使用后44维的动作掩码
        
        # 嵌入所有特征（使用专用嵌入层）
        tile_embeddings = self.tile_embed(features.long())  # [batch, 198, d_model]
        rush_tile_embedding = self.rush_tile_embed(rush_tile.long())  # [batch, 1, d_model]，使用专用rush嵌入
        turn_embedding = self.turn_embed(turn.long())  # [batch, 1, d_model]，使用专用turn嵌入
        
        # 为每个位置生成位置编码
        positions = torch.arange(200, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_embed(positions)  # [batch, 200, d_model]
        
        # 准备特征类型编码
        feature_types = torch.zeros(batch_size, 200, dtype=torch.long, device=device)
        
        # 标记不同类型的特征
        # 当前玩家
        feature_types[:, :14] = 0  # 手牌
        feature_types[:, 14:30] = 1  # rush牌
        feature_types[:, 30:60] = 2  # hitout牌
        
        # 其他玩家 (3个玩家)
        for i in range(1, 4):
            start_idx = 60 + (i-1) * 46
            feature_types[:, start_idx:start_idx+16] = 1  # rush牌
            feature_types[:, start_idx+16:start_idx+46] = 2  # hitout牌
        
        # 标记rush牌和轮数
        feature_types[:, 198] = 4  # rush_tile类型
        feature_types[:, 199] = 3  # 回合类型
        
        # 获取特征类型嵌入
        feature_type_embeddings = self.feature_type_embed(feature_types)  # [batch, 200, d_model]
        
        # 组合所有嵌入
        combined_features = tile_embeddings + pos_embeddings[:, :198] + feature_type_embeddings[:, :198]
        combined_rush_tile = rush_tile_embedding + pos_embeddings[:, 198:199] + feature_type_embeddings[:, 198:199]
        combined_turn = turn_embedding + pos_embeddings[:, 199:200] + feature_type_embeddings[:, 199:200]
        
        # 合并所有特征
        sequence = torch.cat([
            combined_features, 
            combined_rush_tile, 
            combined_turn
        ], dim=1)  # [batch, 200, d_model]
        
        sequence = self.embed_dropout(sequence)
        
        # 生成注意力掩码
        padding_mask = (features == NUM_TILE_TYPES)  # [batch, 198]
        
        # 为rush_tile和轮数添加False（不掩盖）
        additional_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=device)
        padding_mask = torch.cat([padding_mask, additional_mask], dim=1)  # [batch, 200]
        
        # 通过Transformer编码器
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)  # [batch, 200, d_model]
        
        # 使用rush_tile和轮数位置的输出作为决策基础
        decision_features = (encoded[:, 198, :] + encoded[:, 199, :]) / 2  # [batch, d_model]

        return decision_features

class MaskablePPO(PPO):
    """扩展PPO类，增加基于掩码的动作选择"""
    
    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        重写predict方法，确保所选动作始终是有效的
        """
        # 使用原始模型进行预测
        with torch.no_grad():
            obs_tensor = torch.as_tensor([observation]).to(self.policy.device)
            actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()
            
        # 如果是确定性预测，获取最高概率的有效动作
        if deterministic:
            action_mask = observation[200:244].astype(bool)
            # 检查是否有有效动作
            if np.any(action_mask):
                # 过滤出有效动作，选择其中概率最高的
                valid_actions = np.where(action_mask)[0]
                action_probs = self.policy.get_distribution(obs_tensor).distribution.probs.cpu().numpy()
                # 将概率限制为有效动作
                action_probs = action_probs[0][valid_actions]
                # 找到有效动作中最高概率的索引
                best_valid_idx = np.argmax(action_probs)
                # 使用该索引获取实际动作
                actions[0] = valid_actions[best_valid_idx]
        else:
            # 随机情况也要尊重动作掩码
            action_mask = observation[200:244].astype(bool)
            if np.any(action_mask):
                valid_actions = np.where(action_mask)[0]
                # 随机选一个有效动作
                actions[0] = np.random.choice(valid_actions)
        
        return actions, state

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        """
        自定义构建方法，跳过父类的 mlp_extractor 构建
        """
        # 不调用父类的 _build 方法，因为它会创建默认的 mlp_extractor
        # super(CustomPolicy, self)._build(lr_schedule)
        
        # 获取特征维度
        features_dim = self.features_extractor.features_dim
        print(f"特征提取器特征维度: {features_dim}")
        
        # 创建自定义的 mlp_extractor，保持维度不变
        class CustomMlpExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                # 使用 Identity 层，不改变特征维度
                self.policy_net = nn.Identity()
                self.value_net = nn.Identity()
            
            def forward(self, features):
                return self.policy_net(features), self.value_net(features)
    
        # 替换默认的 mlp_extractor
        self.mlp_extractor = CustomMlpExtractor()

        # 自定义 action_net
        self.action_net = nn.Sequential(
            nn.Linear(features_dim, features_dim * 2),  # 输入维度是特征提取器的输出维度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim * 2, 44)  # 输出维度是动作空间的大小
        )

        # 自定义值函数网络 - 确保第一层的输入维度与特征维度一致
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, 64),  # 确保输入维度与特征提取器输出维度匹配
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        # 初始化权重
        self._initialize_weights(self.action_net)
        self._initialize_weights(self.value_net)
    
    def _initialize_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            # 使用 Xavier 初始化线性层
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Sequential):
            # 递归初始化 Sequential 中的每一层
            for submodule in module:
                self._initialize_weights(submodule)
                
    def forward(self, obs, deterministic=False):
        """
        前向传递函数，处理输入观察并输出动作、状态值和动作对数概率
        
        参数:
            obs: 观察向量，包含环境状态
            deterministic: 是否使用确定性策略
            
        返回:
            actions: 选择的动作
            values: 状态值估计
            log_probs: 动作的对数概率
        """
        # 提取特征
        features = self.extract_features(obs)
        
        # 通过 MLP 提取器 - 在我们的实现中是 Identity 层，所以不会改变特征
        pi_features, vf_features = self.mlp_extractor(features)
        
        # 应用动作网络得到 logits
        logits = self.action_net(pi_features)
        
        # 创建分布
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        
        # 采样动作
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        
        # 计算值函数
        values = self.value_net(vf_features)
        
        return actions, values, log_probs
        
    def _predict(self, observation, deterministic=False):
        """
        预测方法，用于处理观察并输出预测的动作
        """
        # 获取动作分布
        observation = observation.float()
        features = self.extract_features(observation)
        pi_features, _ = self.mlp_extractor(features)
        logits = self.action_net(pi_features)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        
        # 采样动作
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
        
        return actions, None
        
    def evaluate_actions(self, obs, actions):
        """
        评估给定的观察和动作，用于训练时的梯度计算
        """
        features = self.extract_features(obs)
        pi_features, value_features = self.mlp_extractor(features)
        
        # 评估动作
        logits = self.action_net(pi_features)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # 评估价值
        values = self.value_net(value_features)
        
        return values, log_prob, entropy

class MahjongPPOAgent:
    """
    使用 PPO 算法的麻将 AI 代理，支持加载预训练模型参数
    """
    def __init__(self, env, model_path=None, pretrained_model_path=None, learning_rate=3e-4):
        self.env = env
        # 明确指定特征维度为128
        features_dim = D_MODEL
        policy_kwargs = {
            'features_extractor_class': MahjongFeatureExtractor,
            'features_extractor_kwargs': {'features_dim': features_dim}
        }
        
        if model_path and os.path.exists(model_path):
            # 加载已有PPO模型
            self.model = MaskablePPO.load(model_path, env=env)
            print(f"加载已有PPO模型: {model_path}")
        else:
            # 创建新模型，设置更适合复杂环境的超参数
            self.model = MaskablePPO(
                CustomPolicy, 
                env, 
                policy_kwargs=policy_kwargs,
                learning_rate=learning_rate, 
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1
            )
            print("创建新的PPO模型")
            print(f"特征提取器输出维度: {self.model.policy.features_dim}")
            
            # 如果提供了预训练模型路径，加载预训练参数
            if pretrained_model_path and os.path.exists(pretrained_model_path):
                self.load_pretrained_parameters(pretrained_model_path)
    
    def load_pretrained_parameters(self, pretrained_model_path):
        """从预训练的MahjongTotalSingleModel加载参数到PPO的特征提取器和策略网络"""
        try:
            # 加载预训练模型
            pretrained_model = MahjongTotalSingleModel()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(pretrained_model_path, map_location=device)
            pretrained_model.load_state_dict(checkpoint["model_state_dict"])
            print(f"加载预训练模型: {pretrained_model_path}")
            
            # 获取PPO的特征提取器
            ppo_feature_extractor = self.model.policy.features_extractor
            
            # 复制嵌入层参数 - 使用安全的方法
            try:
                ppo_feature_extractor.tile_embed.load_state_dict(pretrained_model.tile_embed.state_dict())
                print("成功复制tile_embed参数")
            except Exception as e:
                print(f"复制tile_embed参数时出错: {e}")
                
            # 复制rush嵌入参数
            try:
                ppo_feature_extractor.rush_tile_embed.load_state_dict(pretrained_model.rush_tile_embed.state_dict())
                print("成功复制rush_tile_embed参数")
            except Exception as e:
                print(f"复制rush_tile_embed参数时出错: {e}")
                
            # 复制turn嵌入参数
            try:
                ppo_feature_extractor.turn_embed.load_state_dict(pretrained_model.turn_embed.state_dict())
                print("成功复制turn_embed参数")
            except Exception as e:
                print(f"复制turn_embed参数时出错: {e}")
            
            # 复制位置嵌入参数，考虑到尺寸可能不同
            try:
                min_pos_size = min(ppo_feature_extractor.pos_embed.weight.size(0), 
                                   pretrained_model.pos_embed.weight.size(0))
                ppo_feature_extractor.pos_embed.weight.data[:min_pos_size] = pretrained_model.pos_embed.weight.data[:min_pos_size]
                print("成功复制pos_embed参数")
            except Exception as e:
                print(f"复制pos_embed参数时出错: {e}")
            
            # 复制特征类型嵌入，考虑到尺寸可能不同
            try:
                min_type_size = min(ppo_feature_extractor.feature_type_embed.weight.size(0),
                                    pretrained_model.feature_type_embed.weight.size(0))
                ppo_feature_extractor.feature_type_embed.weight.data[:min_type_size] = pretrained_model.feature_type_embed.weight.data[:min_type_size]
                print("成功复制feature_type_embed参数")
            except Exception as e:
                print(f"复制feature_type_embed参数时出错: {e}")
            
            # 复制Transformer编码器参数
            try:
                ppo_feature_extractor.encoder.load_state_dict(pretrained_model.encoder.state_dict())
                print("成功复制encoder参数")
            except Exception as e:
                print(f"复制encoder参数时出错: {e}")
            
            # 复制统一动作头参数到PPO的动作网络
            try:
                # 使用统一动作头
                if hasattr(pretrained_model, 'unified_action_head'):
                    unified_head = pretrained_model.unified_action_head
                    print(f"找到预训练的统一动作头: {type(unified_head)}")
                    
                    # 打印网络结构以便调试
                    print(f"PPO动作网络类型: {type(self.model.policy.action_net)}")
                    
                    # 无论结构如何，都按层进行匹配
                    if isinstance(self.model.policy.action_net, nn.Sequential) and isinstance(unified_head, nn.Sequential):
                        print("两者都是Sequential结构，开始逐层复制参数")
                        
                        # 打印预训练模型的层结构
                        print("预训练模型的统一动作头结构:")
                        for i, layer in enumerate(unified_head):
                            if hasattr(layer, 'weight'):
                                print(f"  层{i}: {type(layer)}, 输入={layer.in_features}, 输出={layer.out_features}")
                            else:
                                print(f"  层{i}: {type(layer)}")
                        
                        # 打印PPO模型的层结构
                        print("PPO模型的动作网络结构:")
                        for i, layer in enumerate(self.model.policy.action_net):
                            if hasattr(layer, 'weight'):
                                print(f"  层{i}: {type(layer)}, 输入={layer.in_features}, 输出={layer.out_features}")
                            else:
                                print(f"  层{i}: {type(layer)}")
                        
                        # 逐层复制参数
                        layers_copied = 0
                        for i, (ppo_layer, pretrained_layer) in enumerate(zip(self.model.policy.action_net, unified_head)):
                            if hasattr(ppo_layer, 'weight') and hasattr(pretrained_layer, 'weight'):
                                if ppo_layer.weight.shape == pretrained_layer.weight.shape:
                                    ppo_layer.weight.data.copy_(pretrained_layer.weight.data)
                                    ppo_layer.bias.data.copy_(pretrained_layer.bias.data)
                                    layers_copied += 1
                                    print(f"  已复制第{i}层参数: {ppo_layer.weight.shape}")
                                else:
                                    print(f"  第{i}层形状不匹配: PPO={ppo_layer.weight.shape}, 预训练={pretrained_layer.weight.shape}")
                        
                        print(f"总共复制了{layers_copied}层参数")
                        
                    elif isinstance(unified_head, nn.Linear) and hasattr(self.model.policy.action_net, '__getitem__'):
                        # 预训练模型是单层线性层，PPO是Sequential
                        print("预训练模型是单层线性层，PPO是Sequential结构")
                        
                        # 找到PPO中的最后一层线性层
                        for i in range(len(self.model.policy.action_net)-1, -1, -1):
                            layer = self.model.policy.action_net[i]
                            if isinstance(layer, nn.Linear) and layer.out_features == unified_head.out_features:
                                layer.weight.data.copy_(unified_head.weight.data)
                                layer.bias.data.copy_(unified_head.bias.data)
                                print(f"  已将预训练线性层复制到PPO动作网络的第{i}层")
                                break
                        else:
                            print("  未找到匹配的输出层")
                    
                    elif isinstance(self.model.policy.action_net, nn.Linear) and isinstance(unified_head, nn.Sequential):
                        # PPO是单层线性层，预训练模型是Sequential
                        print("PPO是单层线性层，预训练模型是Sequential结构")
                        
                        # 找到预训练中的最后一层线性层
                        for i in range(len(unified_head)-1, -1, -1):
                            layer = unified_head[i]
                            if isinstance(layer, nn.Linear) and layer.out_features == self.model.policy.action_net.out_features:
                                self.model.policy.action_net.weight.data.copy_(layer.weight.data)
                                self.model.policy.action_net.bias.data.copy_(layer.bias.data)
                                print(f"  已将预训练Sequential的第{i}层复制到PPO动作网络")
                                break
                        else:
                            print("  未找到匹配的输出层")
                    
                    elif isinstance(self.model.policy.action_net, nn.Linear) and isinstance(unified_head, nn.Linear):
                        # 两者都是单层线性层
                        print("两者都是单层线性层结构")
                        
                        if self.model.policy.action_net.out_features == unified_head.out_features:
                            self.model.policy.action_net.weight.data.copy_(unified_head.weight.data)
                            self.model.policy.action_net.bias.data.copy_(unified_head.bias.data)
                            print("  已将预训练线性层复制到PPO动作网络")
                        else:
                            print(f"  输出维度不匹配: PPO={self.model.policy.action_net.out_features}, 预训练={unified_head.out_features}")
                            
                    else:
                        print(f"未知的网络结构组合: PPO={type(self.model.policy.action_net)}, 预训练={type(unified_head)}")
                    
                    # 处理值网络
                    # ...existing code...
                    
                else:
                    print("无法找到统一动作头，尝试查找其他可能的输出层")
                    # 尝试查找其他可能的输出层
                    found_output_layer = False
                    for attr_name in ['action_head', 'output_layer', 'disacrd_head']:
                        if hasattr(pretrained_model, attr_name):
                            output_layer = getattr(pretrained_model, attr_name)
                            print(f"使用 {attr_name} 作为输出层")
                            found_output_layer = True
                            
                            # 对于action_net
                            if isinstance(self.model.policy.action_net, nn.Linear):
                                # 如果输出层是Sequential
                                if isinstance(output_layer, nn.Sequential):
                                    # 找到最后一个线性层
                                    for layer in reversed(list(output_layer)):
                                        if isinstance(layer, nn.Linear) and layer.out_features == self.model.policy.action_net.out_features:
                                            self.model.policy.action_net.weight.data.copy_(layer.weight.data)
                                            self.model.policy.action_net.bias.data.copy_(layer.bias.data)
                                            print(f"已将 {attr_name} 中兼容的线性层复制到PPO的action_net")
                                            break
                                # 如果输出层本身是线性层
                                elif isinstance(output_layer, nn.Linear) and output_layer.out_features == self.model.policy.action_net.out_features:
                                    self.model.policy.action_net.weight.data.copy_(output_layer.weight.data)
                                    self.model.policy.action_net.bias.data.copy_(output_layer.bias.data)
                                    print(f"已将 {attr_name} 线性层复制到PPO的action_net")
                            
                            # 对于value_net处理同上
                            # ...
                            
                            break
                    
                    if not found_output_layer:
                        print("无法找到任何合适的输出层")
                        
            except Exception as e:
                print(f"复制输出层参数时出错: {e}")
                import traceback
                traceback.print_exc()
            
            print("预训练模型参数加载完成")
            
        except Exception as e:
            print(f"加载预训练模型参数失败: {e}")
            import traceback
            traceback.print_exc()
    
    def train(self, total_timesteps=100000, log_interval=10, callback=None):
        """训练模型"""
        print(f"开始PPO训练，总时间步数: {total_timesteps}")
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval, callback=callback)
        
    def save(self, path):
        """保存模型"""
        self.model.save(path)
        print(f"模型已保存至: {path}")
    
    def predict(self, observation, deterministic=True):
        """预测动作，确保选择有效动作"""
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action