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
        self.client.connect_to_server()
        # 定义动作空间 - 统一整合为单一Discrete空间
        # 0-33: 出牌 (34张牌)
        # 34-36: 特殊动作 (自摸胡, 暗杠, 明杠)
        # 37: 过
        # 38: 碰
        # 39: 杠
        # 40: 胡
        # 41-43: 吃牌 (前吃, 中吃, 后吃)
        self.action_space = spaces.Discrete(44)
        
        # 定义观察空间 - 整合所有特征到单一Box中
        # 198维特征向量 + 1维当前可决策牌 + 1维当前轮数 + 44维动作掩码
        self.observation_space = spaces.Box(
            low=0, 
            high=NUM_TILE_TYPES, 
            shape=(244,), 
            dtype=np.int32
        )
        
        self.current_obs = None
        self.episode_reward = 0
        self.player_name = "PPOAgent"  # 设置玩家名称
        self.auto_restart = True  # 添加自动重启标志
        self.games_played = 0  # 跟踪已完成游戏数量
        
    def reset(self):
        # 重置环境，连接到服务器并开始新游戏
        
        # 如果是首次reset或者游戏已经完成，则重置游戏状态并开始新游戏
        if not self.client.hand_cards or self.client.game_completed:
            self.client.reset_game_state()
            # 清除之前的游戏结果
            self.client.last_game_result = None
            self.client.game_completed = False
            self.client.current_game_score = 0
            # 发送开始新游戏请求
            self.client.send_play_game(self.player_name)
            
            # 等待游戏开始并获取初始状态，最多等待10秒
            wait_time = 0
            while not self.client.hand_cards and wait_time < 10:
                time.sleep(0.2)
                wait_time += 0.2
                
            if not self.client.hand_cards:
                print("警告: 游戏未能成功启动，手牌为空")
        
        # 构建初始观察
        self.current_obs = self._get_observation()
        self.episode_reward = 0
        
        return self.current_obs
    
    def response_by_action(self, action):
        # 根据动作ID执行相应操作
        if action <= 36:  # 出牌阶段 (出牌、自摸胡、暗杠、明杠)
            if action < 34:  # 普通出牌
                # 从牌ID转换为实际牌
                tile_to_discard = self.client.id_to_tile.get(action, None)
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
            elif action == 34:  # 自摸胡
                self.client.send_zimohu()
            elif action == 35:  # 暗杠
                # 找到可以暗杠的牌
                for tile in set(self.client.hand_cards):
                    if self.client.hand_cards.count(tile) >= 4:
                        self.client.send_angang(tile)
                        break
            elif action == 36:  # 明杠
                # 找到可以明杠的牌
                for tile in set(self.client.hand_cards):
                    if self.client.hand_cards.count(tile) >= 3:
                        tile_id = self.client.tile_to_id.get(tile)
                        # 检查是否有相应的碰牌组合
                        player_idx = int(self.client.player_index)
                        for i in range(len(self.client.player_rush[player_idx])-3):
                            if (self.client.player_rush[player_idx][i:i+3] == [tile, tile, tile] and 
                                self.client.player_rush[player_idx][i+3] == "null"):
                                self.client.send_minggang(tile)
                                break
        else:  # 回应阶段 (pass、碰、杠、胡、吃)
            # 获取当前等待回应的牌
            rush_card = self.client.rush_card
            if rush_card is None:
                # 如果没有等待回应的牌，跳过
                pass
            else:
                if action == 37:  # pass
                    self.client.rush_skip()
                elif action == 38:  # 碰
                    if self.client.hand_cards.count(rush_card) >= 2:
                        self.client.rush_peng(rush_card)
                elif action == 39:  # 杠
                    if self.client.hand_cards.count(rush_card) >= 3:
                        self.client.rush_gang(rush_card)
                elif action == 40:  # 胡
                    self.client.rush_hu(rush_card)
                elif 41 <= action <= 43:  # 吃 (前吃、中吃、后吃)
                    chi_type = action - 41  # 0: 前吃, 1: 中吃, 2: 后吃
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
            
    def step(self, action):
        # 确保动作有效
        valid_action = self._validate_action(action)
        if not valid_action:
            # 如果动作无效，选择一个有效的动作
            action = self._get_fallback_action()
        
        reward = 0
        done = False
        info = {}
        
        # 判断游戏是否结束
        if self.client.game_completed:
            done = True
            reward = self.client.current_game_score
            self.episode_reward += reward
            self.games_played += 1  # 增加已完成游戏计数
            
            info["game_result"] = self.client.last_game_result
            info["win_player"] = self.client.win_player
            info["score_changes"] = self.client.score_changes
            info["hand_score"] = self.client.hand_score
            info["episode_reward"] = self.episode_reward
            info["games_played"] = self.games_played
            
            # 记录游戏统计数据
            print(f"游戏结束! 结果: {self.client.last_game_result}")
            print(f"获胜玩家: {self.client.win_player}")
            print(f"本局得分: {self.client.current_game_score}")
            print(f"累计得分: {self.episode_reward}")
            print(f"已完成游戏数: {self.games_played}")
            
            if hasattr(self.client, 'score_changes') and self.client.score_changes:
                print(f"各玩家分数变化: {self.client.score_changes}")
            if hasattr(self.client, 'hand_score') and self.client.hand_score:
                print(f"手牌分数: {self.client.hand_score}")
                
            # 如果设置了自动重启，则在游戏结束后延迟一段时间再开始新游戏
            if self.auto_restart:
                print("准备开始新游戏...")
                time.sleep(1)  # 短暂延迟，让服务器有时间处理
                self.reset()  # 自动重置并开始新游戏
        else:
            # 获取动作掩码
            action_mask = self.current_obs[200:244]
            
            # 添加中间奖励以引导学习
            if action < 34:  # 出牌
                valid_discard = False
                tile_to_discard = self.client.id_to_tile.get(action, None)
                if tile_to_discard and tile_to_discard in self.client.hand_cards:
                    valid_discard = True
                
                if valid_discard:
                    reward += 0.01  # 为有效出牌提供小的奖励
            
            # 为合法的特殊动作提供更大的奖励
            elif action >= 34 and action_mask[action] == 1:  # 特殊动作且合法
                reward += 0.05
    
        self.response_by_action(action)
        
        self.client.need_discard = False  # 重置出牌标志
        self.client.need_rush = False  # 重置回应标志
        self.rush_card = None  # 重置等待回应的牌
        while not self.client.need_discard and not self.client.need_rush and not self.client.game_completed:
            time.sleep(0.1)
        # 新任务来了，更新观察
        self.current_obs = self._get_observation()
        
        return self.current_obs, reward, done, info
    
    def _validate_action(self, action):
        """验证动作是否有效"""
        # 获取当前观察
        obs = self._get_observation()
        # 提取动作掩码
        action_mask = obs[200:244]
        # 检查该动作是否在有效范围内且被允许
        return 0 <= action < len(action_mask) and action_mask[action] == 1
    
    def _get_fallback_action(self):
        """当选择的动作无效时，返回一个有效的动作"""
        action_mask = self._get_observation()[200:244]
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) > 0:
            return np.random.choice(valid_actions)
        else:
            return 37
    
    def _get_observation(self):
        """
        构建整合后的观察状态，包含：
        1. 198维特征向量（手牌、各玩家的rush牌和hitout牌）
        2. 1维当前等待决策的牌
        3. 1维当前轮数
        4. 44维动作掩码
        总计244维特征
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
        hitout_offset = 14 + 16  # rush牌后的hitout牌偏移量
        
        # 3. 处理当前玩家的rush牌和hitout牌
        for i, tile in enumerate(self.client.player_rush[player_idx]):
            if i < 16 and tile in self.client.tile_to_id and tile != "null":
                features[rush_offset + i] = self.client.tile_to_id[tile]
        for i, tile in enumerate(self.client.player_discards[player_idx]):
            if i < 30 and tile in self.client.tile_to_id:
                features[hitout_offset + i] = self.client.tile_to_id[tile]
        
        # 4. 处理其他玩家的rush牌和hitout牌
        for offset, other_idx in enumerate([(player_idx + i) % 4 for i in range(1, 4)]):
            # 每个玩家占据46个位置: 16个rush牌位 + 30个hitout牌位
            player_base = 14 + 16 + 30 + offset * 46
            rush_base = player_base
            hitout_base = player_base + 16
            
            # 处理其他玩家的rush牌
            if other_idx in self.client.player_rush:
                for i, tile in enumerate(self.client.player_rush[other_idx]):
                    if i < 16 and tile in self.client.tile_to_id and tile != "null":
                        features[rush_base + i] = self.client.tile_to_id[tile]
            
            # 处理其他玩家的hitout牌
            if other_idx in self.client.player_discards:
                for i, tile in enumerate(self.client.player_discards[other_idx]):
                    if i < 30 and tile in self.client.tile_to_id:
                        features[hitout_base + i] = self.client.tile_to_id[tile]
        
        # 获取当前轮数,分桶，10桶
        turn = min(self.client.turn_count // 4, 9)
        turn = np.array([turn], dtype=np.int32)
        
        # 获取当前等待决策的牌
        rush_tile = np.array([NUM_TILE_TYPES], dtype=np.int32)  # 默认使用填充值
        if self.client.rush_card and self.client.rush_card in self.client.tile_to_id:
            rush_tile = np.array([self.client.tile_to_id[self.client.rush_card]], dtype=np.int32)
        
        # 生成44维动作掩码 (整合了所有可能的动作)
        action_mask = np.zeros(44, dtype=np.int32)
        
        if self.client.need_discard:  # discard阶段
            # 填充可出的牌的掩码 (0-33)
            for tile in set(self.client.hand_cards):
                if tile in self.client.tile_to_id:
                    tile_id = self.client.tile_to_id[tile]
                    action_mask[tile_id] = 1
            
            # 填充特殊动作掩码
            if self.client.can_zimohu:
                action_mask[34] = 1  # 自摸胡
            
            # 检查暗杠条件
            for tile in set(self.client.hand_cards):
                if self.client.hand_cards.count(tile) >= 4:
                    action_mask[35] = 1  # 可以暗杠
                    break
            
            # 检查明杠条件
            for tile in set(self.client.hand_cards):
                tile_count = self.client.hand_cards.count(tile)
                if tile_count >= 1:  # 需要至少一张牌才能补杠
                    player_idx = int(self.client.player_index)
                    for i in range(len(self.client.player_rush[player_idx])-2):
                        if (self.client.player_rush[player_idx][i:i+3] == [tile, tile, tile] and 
                            self.client.player_rush[player_idx][i+3] == "null"):
                            action_mask[36] = 1  # 可以明杠
                            break
        elif self.client.need_rush:  # rush阶段
            # 默认可以pass
            action_mask[37] = 1
            
            rush_card = self.client.last_discard
            if rush_card in self.client.tile_to_id:
                rush_id = self.client.tile_to_id[rush_card]
                
                # 检查碰牌条件
                if self.client.hand_cards.count(rush_card) >= 2:
                    action_mask[38] = 1  # 可以碰
                
                # 检查杠牌条件
                if self.client.hand_cards.count(rush_card) >= 3:
                    action_mask[39] = 1  # 可以杠
                
                # 检查胡牌条件
                if self.client.can_hu:
                    action_mask[40] = 1  # 可以胡
                
                # 检查吃牌条件 (41-43)
                if rush_id < 27:  # 只能吃万筒条，不能吃字牌
                    # 检查前吃
                    if (rush_id % 9 >= 2 and
                        self.client.id_to_tile.get(rush_id - 2) in self.client.hand_cards and
                        self.client.id_to_tile.get(rush_id - 1) in self.client.hand_cards):
                        action_mask[41] = 1
                    
                    # 检查中吃
                    if (rush_id % 9 >= 1 and rush_id % 9 <= 7 and
                        self.client.id_to_tile.get(rush_id - 1) in self.client.hand_cards and
                        self.client.id_to_tile.get(rush_id + 1) in self.client.hand_cards):
                        action_mask[42] = 1
                    
                    # 检查后吃
                    if (rush_id % 9 <= 6 and
                        self.client.id_to_tile.get(rush_id + 1) in self.client.hand_cards and
                        self.client.id_to_tile.get(rush_id + 2) in self.client.hand_cards):
                        action_mask[43] = 1
        
        # 构建最终整合的观察向量
        obs = np.concatenate([features, rush_tile, turn, action_mask])
        
        # 确保返回的数组是int32类型，避免类型不一致问题
        return obs.astype(np.int32)
    
    def close(self):
        if self.client.socket:
            self.client.close()