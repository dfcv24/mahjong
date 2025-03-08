import gym
import numpy as np
import time
from gym import spaces
from src.client.client_server import MahjongClientServer
from src.utils.constants import *

class MahjongEnv(gym.Env):
    def __init__(self, server_ip="127.0.0.1", server_port=5000):
        super().__init__()
        
        # 创建麻将客户端
        self.client = MahjongClientServer(server_ip, server_port)
        
        # 定义动作空间
        self.action_space = spaces.Dict({
            'is_discard': spaces.Discrete(2),  # 是否出牌
            'discard': spaces.Discrete(37),  # 出牌动作34张牌+zimohu+angang+minggang
            'rush_action': spaces.Discrete(5),   # 吃碰杠胡pass动作
            "chi_type": spaces.Discrete(3), # 吃的类型, 0: 前吃, 1: 中吃, 2: 后吃
        })
        
        # 定义观察空间 - 按照模型输入特征定义
        # 总维度: 198 = 14*4 + 16*3 + 30*3
        self.observation_space = spaces.Dict({
            'features': spaces.Box(low=0, high=NUM_TILE_TYPES, shape=(198,), dtype=np.int32),
            'rush_tile': spaces.Box(low=0, high=NUM_TILE_TYPES, shape=(1,), dtype=np.int32),
            'turn': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            'discard_mask': spaces.Box(low=0, high=1, shape=(37,), dtype=np.int32),
            'action_mask': spaces.Box(low=0, high=1, shape=(5,), dtype=np.int32),
            'chi_mask': spaces.Box(low=0, high=1, shape=(3,), dtype=np.int32)
        })
        
        self.current_obs = None
        self.episode_reward = 0
        self.player_name = "PPOAgent"  # 设置玩家名称
        
    def reset(self):
        # 重置环境，连接到服务器并开始新游戏
        self.client.reset_game_state()
        # 清除之前的游戏结果
        self.client.last_game_result = None
        self.client.game_completed = False
        self.client.current_game_score = 0
        self.client.send_play_game(self.player_name)
        # 等待游戏开始并获取初始状态
        while not self.client.hand_cards:
            time.sleep(0.1)
        
        # 构建初始观察
        self.current_obs = self._get_observation()
        self.episode_reward = 0
        
        return self.current_obs
        
    def step(self, action):
        # 执行动作(出牌或回应)，获取状态和奖励
        reward = 0
        done = False
        info = {}
        
        # 根据动作类型执行相应操作
        if action['is_discard'] == 1:  # 出牌阶段
            discard = action['discard']
            if discard < 34:  # 普通出牌
                # 从牌ID转换为实际牌
                tile_to_discard = self.client.id_to_tile.get(discard, None)
                if tile_to_discard and tile_to_discard in self.client.hand_cards:
                    # 从手牌中移除要出的牌
                    self.client.hand_cards.remove(tile_to_discard)
                    # 将出的牌添加到自己的弃牌堆
                    player_idx = int(self.client.player_index)
                    self.client.player_discards[player_idx].append(tile_to_discard)
                    # 更新最后出牌信息
                    self.client.last_discard = tile_to_discard
                    self.client.last_discard_player_id = self.client.player_id
                    self.client.last_discard_player_index = self.client.player_index
                    # 向服务器发送出牌消息
                    self.client.hit_out_card(tile_to_discard)
            elif discard == 34:  # 自摸胡
                self.client.send_zimohu()
            elif discard == 35:  # 暗杠
                # 找到可以暗杠的牌
                for tile in set(self.client.hand_cards):
                    if self.client.hand_cards.count(tile) >= 4:
                        self.client.send_angang(tile)
                        break
            elif discard == 36:  # 明杠
                # 找到可以明杠的牌
                for tile in set(self.client.hand_cards):
                    if self.client.hand_cards.count(tile) >= 3:
                        tile_id = self.client.tile_to_id.get(tile)
                        # 检查是否有相应的碰牌组合
                        for i in range(len(self.client.player_rush[int(self.client.player_index)])-3):
                            if (self.client.player_rush[int(self.client.player_index)][i:i+3] == [tile, tile, tile] and 
                                self.client.player_rush[int(self.client.player_index)][i+3] == "null"):
                                self.client.send_minggang(tile)
                                break
        else:  # 回应阶段(吃碰杠胡pass)
            rush_action = action['rush_action']
            chi_type = action['chi_type']
            
            # 获取当前等待回应的牌
            rush_card = self.client.last_discard
            if rush_card is None:
                # 如果没有等待回应的牌，跳过
                pass
            else:
                if rush_action == 0:  # pass
                    self.client.rush_skip(rush_card)
                elif rush_action == 1:  # 吃
                    # 根据chi_type确定吃牌类型
                    rush_id = self.client.tile_to_id.get(rush_card)
                    if chi_type == 0 and rush_id < 27 and rush_id % 9 >= 2:  # 前吃
                        chi_tile1 = self.client.id_to_tile.get(rush_id - 2)
                        chi_tile2 = self.client.id_to_tile.get(rush_id - 1)
                        if chi_tile1 in self.client.hand_cards and chi_tile2 in self.client.hand_cards:
                            self.client.rush_chi(chi_tile1, chi_tile2, rush_card)
                    elif chi_type == 1 and rush_id < 27 and rush_id % 9 >= 1 and rush_id % 9 <= 7:  # 中吃
                        chi_tile1 = self.client.id_to_tile.get(rush_id - 1)
                        chi_tile2 = self.client.id_to_tile.get(rush_id + 1)
                        if chi_tile1 in self.client.hand_cards and chi_tile2 in self.client.hand_cards:
                            self.client.rush_chi(chi_tile1, chi_tile2, rush_card)
                    elif chi_type == 2 and rush_id < 27 and rush_id % 9 <= 6:  # 后吃
                        chi_tile1 = self.client.id_to_tile.get(rush_id + 1)
                        chi_tile2 = self.client.id_to_tile.get(rush_id + 2)
                        if chi_tile1 in self.client.hand_cards and chi_tile2 in self.client.hand_cards:
                            self.client.rush_chi(chi_tile1, chi_tile2, rush_card)
                elif rush_action == 2:  # 碰
                    if self.client.hand_cards.count(rush_card) >= 2:
                        self.client.rush_peng(rush_card)
                elif rush_action == 3:  # 杠
                    if self.client.hand_cards.count(rush_card) >= 3:
                        self.client.rush_gang(rush_card)
                elif rush_action == 4:  # 胡
                    self.client.rush_hu(rush_card)
        
        # 等待服务器响应或游戏状态更新
        # 这里可能需要添加一些等待逻辑，确保动作已被服务器处理
        time.sleep(0.1)  # 简单的等待
            
        # 判断游戏是否结束
        if self.client.game_completed:
            done = True
            reward = self.client.current_game_score
            self.episode_reward += reward
            info["game_result"] = self.client.last_game_result
            
        # 更新观察
        self.current_obs = self._get_observation()
        
        return self.current_obs, reward, done, info
    
    def _get_observation(self):
        """
        构建观察状态，与MahjongTotalModel输入格式保持一致
        特征向量包括:
        1. 手牌特征 (14)
        2. 每个玩家的rush牌 (16*4 = 64)
        3. 每个玩家的hitout牌 (30*4 = 120)
        总计198维特征
        """
        # 初始化198维特征向量 - 用NUM_TILE_TYPES填充作为padding值
        features = np.full(198, NUM_TILE_TYPES, dtype=np.int32)
        
        # 1. 处理手牌特征 (前14维)
        for i, tile in enumerate(self.client.hand_cards[:14]):
            if tile in self.client.tile_to_id:
                features[i] = self.client.tile_to_id[tile]
        
        # 2. 初始化rush牌和hitout牌的位置
        player_idx = int(self.client.player_index)
        rush_offset = 14  # 手牌后的rush牌偏移量
        hitout_offset = 30  # rush牌后的hitout牌偏移量
        
        # 3. 处理当前玩家的rush牌和hitout牌
        for i, tile in enumerate(self.client.player_rush[player_idx]):
            if i < 16 and tile in self.client.tile_to_id and tile != "null":
                features[rush_offset + i] = self.client.tile_to_id[tile]
        
        for i, tile in enumerate(self.client.player_discards[player_idx]):
            if i < 30 and tile in self.client.tile_to_id:
                features[hitout_offset + i] = self.client.tile_to_id[tile]
        
        # 4. 处理其他玩家的rush牌和hitout牌
        for offset, other_idx in enumerate([(player_idx + i) % 4 for i in range(1, 4)]):
            # 每个玩家占据60个位置: 16个rush牌位 + 30个hitout牌位
            player_base = 60 + offset * 60
            
            # 处理其他玩家的rush牌
            if other_idx in self.client.player_rush:
                for i, tile in enumerate(self.client.player_rush[other_idx]):
                    if i < 16 and tile in self.client.tile_to_id and tile != "null":
                        features[player_base + i] = self.client.tile_to_id[tile]
            
            # 处理其他玩家的hitout牌
            if other_idx in self.client.player_discards:
                for i, tile in enumerate(self.client.player_discards[other_idx]):
                    if i < 30 and tile in self.client.tile_to_id:
                        features[player_base + 16 + i] = self.client.tile_to_id[tile]
        
        # 获取当前轮数
        turn = np.array([self.client.turn_count], dtype=np.int32)
        
        # 获取当前等待决策的牌
        rush_tile = np.array([NUM_TILE_TYPES], dtype=np.int32)  # 默认使用填充值
        if self.client.last_discard and self.client.last_discard in self.client.tile_to_id:
            rush_tile = np.array([self.client.tile_to_id[self.client.last_discard]], dtype=np.int32)
        
        # 生成出牌掩码
        discard_mask = np.zeros(37, dtype=np.int32)
        # 填充手牌可出的掩码
        for tile in set(self.client.hand_cards):
            if tile in self.client.tile_to_id:
                tile_id = self.client.tile_to_id[tile]
                discard_mask[tile_id] = 1
        
        # 填充特殊动作掩码(胡牌、暗杠、明杠)
        if self.client.can_zimohu:
            discard_mask[34] = 1  # 自摸胡
        
        # 检查暗杠条件
        for tile in set(self.client.hand_cards):
            if self.client.hand_cards.count(tile) >= 4:
                discard_mask[35] = 1  # 可以暗杠
                break
        
        # 检查明杠条件
        for tile in set(self.client.hand_cards):
            if self.client.hand_cards.count(tile) >= 3:
                for i in range(len(self.client.player_rush[player_idx])-3):
                    if (self.client.player_rush[player_idx][i:i+3] == [tile, tile, tile] and 
                        self.client.player_rush[player_idx][i+3] == "null"):
                        discard_mask[36] = 1  # 可以明杠
                        break
        
        # 生成回应动作掩码(吃碰杠胡pass)
        action_mask = np.zeros(5, dtype=np.int32)
        action_mask[0] = 1  # 默认可以pass
        
        # 检查其他动作是否可行
        rush_card = self.client.last_discard
        if rush_card in self.client.tile_to_id:
            rush_id = self.client.tile_to_id[rush_card]
            
            # 检查吃牌条件
            if rush_id < 27:  # 只能吃万筒条，不能吃字牌
                # 检查前吃
                if (rush_id % 9 >= 2 and
                    self.client.id_to_tile.get(rush_id - 2) in self.client.hand_cards and
                    self.client.id_to_tile.get(rush_id - 1) in self.client.hand_cards):
                    action_mask[1] = 1  # 可以吃
                
                # 检查中吃
                if (rush_id % 9 >= 1 and rush_id % 9 <= 7 and
                    self.client.id_to_tile.get(rush_id - 1) in self.client.hand_cards and
                    self.client.id_to_tile.get(rush_id + 1) in self.client.hand_cards):
                    action_mask[1] = 1  # 可以吃
                
                # 检查后吃
                if (rush_id % 9 <= 6 and
                    self.client.id_to_tile.get(rush_id + 1) in self.client.hand_cards and
                    self.client.id_to_tile.get(rush_id + 2) in self.client.hand_cards):
                    action_mask[1] = 1  # 可以吃
            
            # 检查碰牌条件
            if self.client.hand_cards.count(rush_card) >= 2:
                action_mask[2] = 1  # 可以碰
            
            # 检查杠牌条件
            if self.client.hand_cards.count(rush_card) >= 3:
                action_mask[3] = 1  # 可以杠
            
            # 检查胡牌条件
            if self.client.can_dianhu(rush_card):
                action_mask[4] = 1  # 可以胡
        
        # 生成吃牌类型掩码
        chi_mask = np.zeros(3, dtype=np.int32)
        if rush_card in self.client.tile_to_id:
            rush_id = self.client.tile_to_id[rush_card]
            if rush_id < 27:  # 只能吃万筒条
                # 检查前吃
                if (rush_id % 9 >= 2 and
                    self.client.id_to_tile.get(rush_id - 2) in self.client.hand_cards and
                    self.client.id_to_tile.get(rush_id - 1) in self.client.hand_cards):
                    chi_mask[0] = 1
                
                # 检查中吃
                if (rush_id % 9 >= 1 and rush_id % 9 <= 7 and
                    self.client.id_to_tile.get(rush_id - 1) in self.client.hand_cards and
                    self.client.id_to_tile.get(rush_id + 1) in self.client.hand_cards):
                    chi_mask[1] = 1
                
                # 检查后吃
                if (rush_id % 9 <= 6 and
                    self.client.id_to_tile.get(rush_id + 1) in self.client.hand_cards and
                    self.client.id_to_tile.get(rush_id + 2) in self.client.hand_cards):
                    chi_mask[2] = 1
        
        # 构建观察字典
        obs = {
            'features': features,
            'rush_tile': rush_tile,
            'turn': turn,
            'discard_mask': discard_mask,
            'action_mask': action_mask,
            'chi_mask': chi_mask
        }
        
        return obs
    
    def close(self):
        if self.client.socket:
            self.client.close()