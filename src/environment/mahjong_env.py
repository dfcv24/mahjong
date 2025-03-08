import gym
import numpy as np
from gym import spaces
from src.client.client import MahjongClient
from src.utils.constants import *

class MahjongEnv(gym.Env):
    def __init__(self, server_ip="127.0.0.1", server_port=5000):
        super().__init__()
        
        # 创建麻将客户端
        self.client = MahjongClient(server_ip=server_ip, server_port=server_port)
        self.client.load_ai_model()  # 加载监督学习模型作为初始策略
        
        # 定义动作空间
        self.action_space = spaces.Dict({
            'discard': spaces.Discrete(34),  # 出牌动作
            'response': spaces.Discrete(5)   # 吃碰杠胡pass动作
        })
        
        # 定义观察空间 (根据您的特征设计调整维度)
        self.observation_space = spaces.Dict({
            'hand_cards': spaces.Box(low=0, high=4, shape=(34,), dtype=np.int32),
            'discard_history': spaces.Box(low=0, high=4, shape=(34, 4), dtype=np.int32),
            'last_discard': spaces.Box(low=0, high=33, shape=(1,), dtype=np.int32),
            'player_index': spaces.Discrete(4)
        })
        
        self.current_obs = None
        self.episode_reward = 0
        
    def reset(self):
        # 重置环境，连接到服务器并开始新游戏
        if self.client.socket:
            self.client.close()
        
        self.client.connect()
        self.client.join_game()
        
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
        
        # 判断是出牌还是回应动作
        if self.client.waiting_for_discard:
            # 出牌动作
            tile_to_discard = self.client.id_to_tile.get(action['discard'])
            self.client.discard_tile(tile_to_discard)
        elif self.client.waiting_for_response:
            # 吃碰杠胡或pass的回应动作
            self.client.respond_to_action(action['response'])
        
        # 等待下一个决策点
        while not (self.client.waiting_for_discard or self.client.waiting_for_response or self.client.game_completed):
            time.sleep(0.1)
            
        # 判断游戏是否结束
        if self.client.game_completed:
            done = True
            reward = self.client.current_game_score
            self.episode_reward += reward
            
        # 更新观察
        self.current_obs = self._get_observation()
        
        return self.current_obs, reward, done, info
    
    def _get_observation(self):
        # 构建观察状态
        # 将客户端当前状态转换为神经网络可用的特征表示
        hand_cards_array = np.zeros(34, dtype=np.int32)
        for tile in self.client.hand_cards:
            tile_id = self.client.tile_to_id[tile]
            hand_cards_array[tile_id] += 1
            
        # 构建出牌历史特征
        discard_history = np.zeros((34, 4), dtype=np.int32)
        for player_idx, tiles in self.client.player_discards.items():
            for tile in tiles:
                if tile in self.client.tile_to_id:
                    tile_id = self.client.tile_to_id[tile]
                    # 找到该牌当前计数并递增
                    count = sum(discard_history[tile_id])
                    if count < 4:
                        discard_history[tile_id][count] = 1
        
        # 最后打出的牌
        last_discard = np.array([-1]) if self.client.last_discard is None else \
                      np.array([self.client.tile_to_id.get(self.client.last_discard, -1)])
        
        obs = {
            'hand_cards': hand_cards_array,
            'discard_history': discard_history,
            'last_discard': last_discard,
            'player_index': self.client.player_index
        }
        
        return obs
    
    def close(self):
        if self.client.socket:
            self.client.close()