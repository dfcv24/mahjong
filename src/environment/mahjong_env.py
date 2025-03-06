# src/environment/mahjong_env.py

import gym
from gym import spaces
import numpy as np
import torch
from ..utils.tile_utils import init_tile_mapping
from ..utils.constants import NUM_TILE_TYPES

class MahjongGymEnv(gym.Env):
    """
    麻将环境的 Gym 接口，用于 PPO 训练
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, model=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(MahjongGymEnv, self).__init__()
        
        # 加载模型和设备
        self.model = model
        self.device = device
        
        # 创建牌面到ID的映射
        self.tile_to_id, self.id_to_tile = init_tile_mapping()
        
        # 动作空间：34种基本牌 + 吃碰杠胡
        self.action_space = spaces.Discrete(NUM_TILE_TYPES + 4)  # 34种牌 + 吃碰杠胡
        
        # 观察空间：
        # - 手牌 (14张牌，每张牌34种可能)
        # - 回合数 (最多50回合)
        # - 其他玩家出牌历史 (3个玩家，每人最多打出20张牌)
        # - 当前场上状态 (如上一张出牌等)
        self.observation_space = spaces.Dict({
            'hand': spaces.Box(low=0, high=NUM_TILE_TYPES, shape=(14,), dtype=np.int32),
            'turn': spaces.Discrete(50),
            'player_discards': spaces.Box(low=0, high=NUM_TILE_TYPES, shape=(3, 20), dtype=np.int32),
            'last_discard': spaces.Discrete(NUM_TILE_TYPES + 1),  # +1 表示无出牌
        })
        
        # 游戏状态
        self.hand_cards = []
        self.turn_count = 0
        self.player_discards = {0: [], 1: [], 2: [], 3: []}
        self.last_discard = None
        self.last_discard_player_index = None
        self.current_player_index = 0
        
    def reset(self):
        """重置环境到初始状态"""
        # 初始化手牌、回合计数等
        self.hand_cards = []  # 在实际使用时，这里会填充一副初始手牌
        self.turn_count = 0
        self.player_discards = {0: [], 1: [], 2: [], 3: []}
        self.last_discard = None
        self.last_discard_player_index = None
        
        return self._get_observation()
    
    def step(self, action):
        """
        执行一步动作，返回新的状态、奖励、是否结束和额外信息
        
        Parameters:
        - action: 要执行的动作索引
        
        Returns:
        - observation: 新的观察状态
        - reward: 获得的奖励
        - done: 游戏是否结束
        - info: 额外信息
        """
        # 根据行动更新状态
        if action < NUM_TILE_TYPES:  # 出牌动作
            # 从手牌中移除打出的牌
            tile_name = self.id_to_tile[action]
            if tile_name in self.hand_cards:
                self.hand_cards.remove(tile_name)
                
                # 更新出牌历史
                self.player_discards[self.current_player_index].append(tile_name)
                self.last_discard = tile_name
                self.last_discard_player_index = self.current_player_index
        else:  # 吃碰杠胡动作
            # 在实际应用中实现具体逻辑
            pass
            
        # 增加回合计数
        self.turn_count += 1
        
        # 判断是否结束
        done = self.turn_count >= 50 or len(self.hand_cards) == 0
        
        # 计算奖励
        reward = self._calculate_reward(action)
        
        return self._get_observation(), reward, done, {}
    
    def _calculate_reward(self, action):
        """根据动作计算奖励"""
        # 这是一个示例奖励函数，需要根据实际游戏规则和策略调整
        reward = 0
        
        # 使用现有的AI模型进行评估，对于好的动作给予更高的奖励
        if self.model is not None and action < NUM_TILE_TYPES:
            with torch.no_grad():
                # 将状态转换为模型输入
                hand_tensor = self._convert_hand_to_tensor(self.hand_cards)
                turn_tensor = torch.tensor([self.turn_count], dtype=torch.long)
                
                # 移动到设备
                hand_tensor = hand_tensor.to(self.device)
                turn_tensor = turn_tensor.to(self.device)
                
                # 模型预测
                logits = self.model(hand_tensor, turn_tensor)
                probs = torch.softmax(logits, dim=1)
                
                # 如果模型认为当前动作概率高，给予更高的奖励
                action_prob = probs[0, action].item()
                reward = action_prob * 2 - 1  # 将概率转换为 [-1, 1] 范围的奖励
        
        return reward
    
    def _get_observation(self):
        """构造当前观察状态"""
        # 将手牌转换为索引形式
        hand_indices = np.full(14, NUM_TILE_TYPES)  # 默认填充值表示无牌
        for i, card in enumerate(self.hand_cards):
            if i < 14 and card in self.tile_to_id:
                hand_indices[i] = self.tile_to_id[card]
        
        # 处理玩家出牌历史
        player_discard_indices = np.full((3, 20), NUM_TILE_TYPES)
        for player_idx in range(3):
            actual_idx = (self.current_player_index + player_idx + 1) % 4
            discards = self.player_discards[actual_idx]
            for i, card in enumerate(discards):
                if i < 20 and card in self.tile_to_id:
                    player_discard_indices[player_idx, i] = self.tile_to_id[card]
        
        # 处理最后出牌
        last_discard_idx = NUM_TILE_TYPES  # 默认表示无出牌
        if self.last_discard and self.last_discard in self.tile_to_id:
            last_discard_idx = self.tile_to_id[self.last_discard]
        
        return {
            'hand': hand_indices,
            'turn': self.turn_count,
            'player_discards': player_discard_indices,
            'last_discard': last_discard_idx
        }
        
    def _convert_hand_to_tensor(self, hand_cards):
        """将手牌转换为模型输入的张量形式"""
        # 创建索引张量，初始化为填充标记
        hand_indices = torch.full((1, 14), NUM_TILE_TYPES, dtype=torch.long)
        
        # 为每张手牌设置对应的索引
        for i, card in enumerate(hand_cards):
            if i < 14:  # 最多处理14张牌
                if card in self.tile_to_id:
                    hand_indices[0, i] = self.tile_to_id[card]
        
        return hand_indices
    
    def render(self, mode='human'):
        """渲染当前环境状态"""
        if mode == 'human':
            print(f"回合: {self.turn_count}")
            print(f"手牌: {self.hand_cards}")
            print(f"最后打出的牌: {self.last_discard} (玩家 {self.last_discard_player_index})")