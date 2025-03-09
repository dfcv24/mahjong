import sys
import os
import socket
import threading
import time
import json
import torch
import traceback
import logging
import numpy as np
from collections import deque

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.model import MahjongDiscardModel, MahjongActionModel, MahjongTotalModel
from src.utils.constants import *
from src.utils.tile_utils import init_tile_mapping, sort_hand_cards
# 导入PPO相关类
from src.agents.new_ppo_agent import MahjongPPO, PPOMemory

class MahjongClient:
    def __init__(self, server_ip="127.0.0.1", server_port=5000, client_port=5001, agent=None, logger=None):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_ip = "127.0.0.1"
        self.client_port = client_port
        self.socket = None
        self.receive_thread = None
        self.running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tile_to_id = None
        self.id_to_tile = None
        self.logger = logger
        # 玩家信息
        self.player_id = None  # 玩家名称/ID
        self.player_index = None  # 玩家在牌桌上的位置
        self.hand_cards = []
        # 游戏状态跟踪
        self.game_id = None
        self.turn_count = 0  # 记录游戏回合数
        self.player_rush = {0: [], 1: [], 2: [], 3: []}  # 吃碰杠胡历史
        self.player_discards = {0: [], 1: [], 2: [], 3: []}  # 每个位置玩家的出牌历史
        self.last_discard = None  # 最后一张出牌
        self.last_discard_player_id = None  # 最后出牌的玩家id
        self.last_discard_player_index = None  # 最后出牌的玩家index
        self.current_game_score = 0  # 当前游戏得分
        self.game_completed = False  # 游戏是否结束
        
        # PPO智能体相关，eposode表示比赛数
        self.ppo_agent = agent if agent is not None else MahjongPPO(device=self.device)
        self.memory = PPOMemory()
        self.episode_reward = 0
        self.episode_count = 0
        self.update_frequency = 10
        self.save_frequency = 200
        
        # 当前状态和动作追踪
        self.current_feature = None
        self.current_rush_tile_id = None
        self.current_turn = None
        self.current_action_mask = None
        self.current_action = None
        self.current_log_prob = None
        self.current_value = None
        self.last_reward = 0
        
        # 训练监控相关
        self.episode_rewards = []
        self.episode_lengths = []
        self.recent_rewards = deque(maxlen=10)  # 保存最近10局奖励
        self.recent_win_rate = deque(maxlen=10)  # 保存最近10局胜率
        self.recent_scores = deque(maxlen=10)    # 保存最近10局分数
        self.total_wins = 0                      # 总胜局数
        self.episode_start_time = time.time()    # 记录每局开始时间
        self.total_training_time = 0             # 总训练时间
        
        # 行动统计
        self.action_counts = {
            'discard': 0,       # 打牌数
            'hu': 0,            # 胡牌数
            'pass': 0,          # 过牌数
            'peng': 0,          # 碰牌数
            'gang': 0,          # 杠牌数
            'chi': 0,           # 吃牌数
            'zimohu': 0,        # 自摸次数
            'angang': 0,        # 暗杠次数
            'minggang': 0       # 明杠次数
        }
        
        # PPO训练监控
        self.ppo_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'learning_rate': []
        }
        
        # 最佳模型管理
        self.best_reward = -float('inf')
        self.best_win_rate = 0
    
    def connect_to_server(self):
        """连接到麻将服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            self.logger.info(f"成功连接到服务器 {self.server_ip}:{self.server_port}")
            
            # 启动接收消息的线程
            self.running = True
            self.receive_thread = threading.Thread(target=self.receive_messages)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            return True
        except Exception as e:
            self.logger.error(f"连接服务器失败: {e}")
            return False
    
    def receive_messages(self):
        """接收服务器消息的线程"""
        buffer = ""
        while self.running:
            try:
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    self.logger.info("服务器连接已关闭")
                    self.running = False
                    break
                
                self.logger.debug(f"收到原始数据: {data!r}")  # 以原始格式打印，包括特殊字符
                # 将整个消息直接传递给处理函数
                self.handle_message(data)
            except Exception as e:
                self.logger.error(f"接收消息时出错: {e}")
                traceback.print_exc()  # 打印详细错误栈
                self.running = False
                break
        
    def handle_message(self, message):
        """处理从服务器接收到的消息"""
        # 假设从服务器发来永远只有一个命令
        # StartGame 游戏编号 玩家编号(ID) 玩家序号(位置)
        parts = message.split()
        if parts[0] == "StartGame":
            if len(parts) >= 4:
                self.game_id = parts[1]
                self.player_id = parts[2]  # 玩家ID
                self.player_index = parts[3]  # 玩家在牌桌上的位置
                self.logger.info(f"游戏开始: ID={self.game_id}, 玩家ID={self.player_id}, 牌桌位置={self.player_index}")
                # 自动回复准备就绪
                self.send_ready()
        elif parts[0] == "HandPaiList":
            if len(parts) > 4:  # 消息头 + 游戏信息(3) + 至少一张牌                
                # 从第5个元素开始是手牌
                cards = parts[4:]
                self.hand_cards = cards
                self.logger.info(f"收到初始手牌: {self.hand_cards}")
                self.logger.debug(f"手牌数量: {len(self.hand_cards)}")
                self.send_ready()                
        # NextPai 游戏编号 玩家编号 玩家序号 牌名 是否能胡 是否能暗杠 是否能明杠- 当前玩家摸牌
        elif parts[0] == "NextPai" or parts[0] == "GangPai":
            if len(parts) >= 8:
                new_tile = parts[4]
                can_zimohu = parts[5] == "True"
                can_angang = parts[6] == "True"
                can_minggang = parts[7] == "True"
                # 增加回合计数
                self.turn_count += 1       
                # 将新牌添加到手牌中
                if new_tile not in ["", "None"]:
                    self.hand_cards.append(new_tile)
                    self.hand_cards = sort_hand_cards(self.hand_cards)
                self.logger.debug(f"摸到新牌: {new_tile}, 可以自摸胡: {can_zimohu}, 可以暗杠: {can_angang}, 可以明杠: {can_minggang}")
                self.logger.debug(f"当前手牌: {self.hand_cards} (共{len(self.hand_cards)}张)")
                # 如果可以胡且AI模型已加载，考虑自动胡牌
                self.auto_play(new_tile, can_zimohu, can_angang, can_minggang)
        # NextPaiOther 游戏编号 玩家编号 玩家序号 拿牌玩家编号 拿牌玩家序号 - 其他玩家摸牌
        elif parts[0] == "NextPaiOther" or parts[0] == "GangPaiOther":
            if len(parts) >= 6:
                draw_player_id = parts[4]
                draw_player_index = parts[5]
                if parts[0] == "NextPaiOther":
                    self.logger.debug(f"玩家 {draw_player_id} (位置 {draw_player_index}) 摸了一张牌")
                else:
                    self.logger.debug(f"玩家 {draw_player_id} (位置 {draw_player_index}) 杠了一张牌")
            self.send_ready()
        # NotifyOtherHitOut 游戏编号 玩家编号 玩家序号 牌名 出牌玩家编号 出牌玩家序号
        elif parts[0] == "NotifyOtherHitOut":
            if len(parts) >= 7:
                card = parts[4]
                hit_player_id = parts[5]
                hit_player_index = parts[6]
                
                # 记录特定玩家的出牌历史
                player_pos = int(hit_player_index)
                if player_pos in self.player_discards:
                    self.player_discards[player_pos].append(card)
                
                # 记录最后出牌信息
                self.last_discard = card
                self.last_discard_player_id = hit_player_id
                self.last_discard_player_index = hit_player_index
                
                self.logger.debug(f"玩家 {hit_player_id} (位置 {hit_player_index}) 打出了: {card}")
                
                # 回复服务器已准备好
                self.send_ready()
        # ChiPengGangHu 游戏编号 玩家编号 玩家序号 是否有吃 是否有碰 是否有杠 是否有胡 牌名
        elif parts[0] == "ChiPengGangHu":
            if len(parts) >= 9:
                can_chi = parts[4] == "True"
                can_peng = parts[5] == "True"
                can_gang = parts[6] == "True"
                can_hu = parts[7] == "True"
                rush_card = parts[8]
                
                self.logger.debug(f"收到吃碰杠胡请求: {rush_card}, 可吃: {can_chi}, 可碰: {can_peng}, 可杠: {can_gang}, 可胡: {can_hu}")
                self.logger.debug(f"当前手牌: {self.hand_cards}")
                
                # 使用PPO智能体决策
                self.auto_rush(rush_card, can_chi, can_peng, can_gang, can_hu)
        # NotifyRushPeng 游戏编号 玩家编号 玩家序号 碰牌玩家编号 碰牌玩家序号 碰牌
        elif parts[0] == "NotifyRushPeng":
            if len(parts) >= 7:
                peng_player_id = parts[4]
                peng_player_index = parts[5]
                peng_tile = parts[6]
                
                self.logger.debug(f"玩家 {peng_player_id} (位置 {peng_player_index}) 碰了牌: {peng_tile}")
                self.player_rush[int(peng_player_index)].extend([peng_tile, peng_tile, peng_tile, "null"])
                # 如果是自己碰牌，先移除碰牌，再出牌
                if peng_player_id == self.player_id:
                    self.hand_cards.remove(peng_tile)
                    self.hand_cards.remove(peng_tile)
                    self.auto_play()
                else:
                    self.send_ready()
        # NotifyRushGang 游戏编号 玩家编号 玩家序号 杠牌玩家编号 杠牌玩家序号 杠牌1 抽牌2
        elif parts[0] == "NotifyRushGang":
            if len(parts) >= 7:
                gang_player_id = parts[4]
                gang_player_index = parts[5]
                rush_tile = parts[6]
                if len(parts) >= 8:
                    gang_tile = parts[7]
                    self.logger.debug(f"玩家 {gang_player_id} (位置 {gang_player_index}) 杠了牌: {rush_tile} (抽牌: {gang_tile})")
                else:
                    self.logger.debug(f"玩家 {gang_player_id} (位置 {gang_player_index}) 杠了牌: {rush_tile})")
                
                self.player_rush[int(gang_player_index)].extend([rush_tile, rush_tile, rush_tile, rush_tile])
                # 如果是自己杠牌，先移除rush牌，杠来的牌加入手牌，再出牌
                if gang_player_id == self.player_id:
                    self.hand_cards.remove(rush_tile)
                    self.hand_cards.remove(rush_tile)
                    self.hand_cards.remove(rush_tile)
                    self.hand_cards.append(gang_tile)
                    self.hand_cards = sort_hand_cards(self.hand_cards)
                    self.auto_play()
                else:
                    self.send_ready()
        # NotifyRushChi 游戏编号 玩家编号 玩家序号 吃牌玩家编号 吃牌玩家序号 吃牌1 吃牌2 吃牌3
        elif parts[0] == "NotifyRushChi":
            if len(parts) >= 9:
                chi_player_id = parts[4]
                chi_player_index = parts[5]
                chi_tile1 = parts[6]
                chi_tile2 = parts[7]
                chi_tile3 = parts[8]
                
                self.logger.debug(f"玩家 {chi_player_id} (位置 {chi_player_index}) 吃了牌组合: {chi_tile1} {chi_tile2} {chi_tile3}")
                self.player_rush[int(chi_player_index)].extend([chi_tile1, chi_tile2, chi_tile3, "null"])
                # 如果是自己吃牌，移除吃牌，再出牌
                if chi_player_id == self.player_id:
                    self.hand_cards.remove(chi_tile1)
                    self.hand_cards.remove(chi_tile2)
                    self.auto_play()
                else:
                    self.send_ready()
        # NotifyRushHu 游戏编号 玩家编号 玩家序号 胡牌玩家编号 胡牌玩家序号 手牌数 手牌列表
        elif parts[0] == "NotifyRushHu":
            if len(parts) >= 7:
                hu_player_id = parts[4]
                hu_player_index = parts[5]
                hand_count = int(parts[6])
                winning_hand = []
                if len(parts) > 7:
                    winning_hand = parts[7:7+hand_count]
                
                self.logger.info(f"玩家 {hu_player_id} (位置 {hu_player_index}) 胡牌了!")
                self.logger.debug(f"胡牌玩家手牌: {winning_hand}")
                
                # 玩家得分等信息
                if hu_player_id == self.player_id:
                    self.current_game_score += 8  # 胡牌得分
                    self.action_counts['hu'] += 1  # 记录胡牌次数
                    self.total_wins += 1          # 增加胜局数
                elif self.last_discard_player_id == self.player_id:
                    self.current_game_score -= 6  # 放炮扣分
                else:
                    self.current_game_score -= 1
                
                # 回复服务器已准备好
                self.send_ready()
        elif parts[0] == "NotifyZimoHu":
            hu_player_id = parts[4]
            hu_player_index = parts[5]
            if len(parts) >= 7:  
                hand_count = int(parts[6])
                winning_hand = []
                if len(parts) > 7:
                    winning_hand = parts[7:7+hand_count]
                
                self.logger.info(f"玩家 {hu_player_id} (位置 {hu_player_index}) 胡牌了!")
                self.logger.debug(f"胡牌玩家手牌: {winning_hand}")
            # 玩家得分等信息
            if hu_player_id == self.player_id:
                self.current_game_score = 9
                self.action_counts['zimohu'] += 1  # 记录自摸次数
                self.total_wins += 1              # 增加胜局数
            else:
                self.current_game_score = -3
            
            self.send_ready()
        elif parts[0] == "NotifyZimoAnGang":
            gang_player_id = parts[4]
            gang_player_index = parts[5]
            if len(parts) >= 7 and gang_player_id == self.player_id:
                gang_tile = parts[6]
                self.logger.debug(f"玩家 {gang_player_id} (位置 {gang_player_index}) 暗杠了牌: {gang_tile}")
                self.player_rush[int(gang_player_index)].extend([gang_tile, gang_tile, gang_tile, gang_tile])
                self.hand_cards.remove(gang_tile)
                self.hand_cards.remove(gang_tile)
                self.hand_cards.remove(gang_tile)
            else:
                self.logger.debug(f"玩家 {gang_player_id} (位置 {gang_player_index}) 暗杠了牌")
            self.send_ready()
        elif parts[0] == "NotifyZimoMingGang":
            gang_player_id = parts[4]
            gang_player_index = parts[5]
            gang_tile = parts[6]
            self.logger.debug(f"玩家 {gang_player_id} (位置 {gang_player_index}) 明杠了牌: {gang_tile}")
            # 找到原来3张gang_tile的位置，后面加上gang_tile，替换null牌
            for i in range(len(self.player_rush[int(gang_player_index)])):
                if self.player_rush[int(gang_player_index)][i] == gang_tile and self.player_rush[int(gang_player_index)][i+1] == "null":
                    self.player_rush[int(gang_player_index)][i+1] = gang_tile
                    break
            self.send_ready()
        # 处理游戏结束消息
        elif parts[0] == "GameFinished":
            self.logger.info("游戏结束")
            # 计算最终奖励并存储转换
            reward = self.calculate_reward({"game_over": True})
            self.store_transition(reward, True)
            
            # 更新episode计数
            self.episode_count += 1
            
            # 记录本局游戏统计数据
            episode_time = time.time() - self.episode_start_time
            self.total_training_time += episode_time
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.turn_count)
            self.recent_rewards.append(self.episode_reward)
            self.recent_scores.append(self.current_game_score)
            
            # 计算胜率并记录
            win_in_this_episode = 1 if self.current_game_score > 0 else 0
            self.recent_win_rate.append(win_in_this_episode)
            
            # 打印本局游戏统计信息
            self.print_episode_stats()
            
            # 如果达到更新频率，更新智能体
            if self.episode_count % self.update_frequency == 0:
                self.update_agent()
                
                # 打印更新后的智能体统计信息
                self.print_training_stats()
                
                if self.episode_count % self.save_frequency == 0:
                    # 如果当前模型表现超过最佳模型，保存为最佳模型
                    avg_recent_reward = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0
                    avg_win_rate = sum(self.recent_win_rate) / len(self.recent_win_rate) if self.recent_win_rate else 0
                    
                    if avg_recent_reward > self.best_reward:
                        self.best_reward = avg_recent_reward
                        self.save_model(f"mahjong_ppo_best_reward_ep{self.episode_count}.pt")
                        self.logger.info(f"保存新的最佳奖励模型，平均奖励: {avg_recent_reward:.2f}")
                    
                    if avg_win_rate > self.best_win_rate:
                        self.best_win_rate = avg_win_rate
                        self.save_model(f"mahjong_ppo_best_winrate_ep{self.episode_count}.pt")
                        self.logger.info(f"保存新的最佳胜率模型，胜率: {avg_win_rate:.2f}")
                    
                    # 保存定期检查点
                    self.save_model(f"mahjong_ppo_model_ep{self.episode_count}.pt")
            
            self.send_ready()
            self.reset_game_state()
            
            # 重置奖励相关记录
            self.episode_reward = 0
            self.last_reward = 0
            self.game_completed = True
            
            # 重置本局开始时间
            self.episode_start_time = time.time()
    
    def auto_play(self, new_tile="null", can_zimohu=False, can_angang=False, can_minggang=False):
        """AI自动出牌 - 使用PPO智能体"""
        # 计算奖励并存储转换, 保证obs为最新。
        reward = self.calculate_reward()
        self.store_transition(reward, False)

        # 初始化牌名映射（如果需要）
        if self.tile_to_id is None or self.id_to_tile is None:
            self.tile_to_id, self.id_to_tile = init_tile_mapping()
        
        # 处理游戏状态，为PPO准备输入
        state_tensor, rush_tile_tensor, turn_tensor = self.process_game_state_for_ppo()
        
        # 修改动作掩码以反映特殊动作，长度44，需要初始化
        action_mask = torch.zeros(44, dtype=torch.bool)  # 全部初始化为False
        if NUM_TILE_TYPES < len(action_mask):  # 确保有足够的长度设置特殊动作
            for tile in self.hand_cards:
                if self.tile_to_id[tile] < NUM_TILE_TYPES:  # 确保是有效的牌ID
                    action_mask[self.tile_to_id[tile]] = True
            action_mask[NUM_TILE_TYPES] = can_zimohu
            action_mask[NUM_TILE_TYPES + 1] = can_angang
            action_mask[NUM_TILE_TYPES + 2] = can_minggang
        action_mask_tensor = action_mask.unsqueeze(0).to(self.device)
        self.current_action_mask = action_mask_tensor
        # 使用PPO智能体选择动作
        with torch.no_grad():
            # 获取特征
            combined_features = self.ppo_agent.feature_extractor(
                state_tensor.unsqueeze(0),
                rush_tile_tensor.unsqueeze(0),
                turn_tensor.unsqueeze(0)
            )
            self.current_feature = combined_features
            
            # 从策略中采样动作
            action, log_prob, probs = self.ppo_agent.policy.get_action_and_log_prob(
                combined_features, 
                action_mask_tensor
            )
            
            # 估计状态值
            state_value = self.ppo_agent.value(combined_features)
            
            # 保存当前动作信息
            self.current_action = action.item()
            self.current_log_prob = log_prob
            self.current_value = state_value
            
            # 执行选择的动作
            if self.current_action == NUM_TILE_TYPES and can_zimohu:
                self.logger.debug("PPO智能体选择自摸胡牌")
                self.action_counts['zimohu'] += 1
                self.send_zimohu()
            elif self.current_action == NUM_TILE_TYPES + 1 and can_angang:
                self.logger.debug("PPO智能体选择暗杠")
                self.action_counts['angang'] += 1
                self.send_angang(new_tile)
            elif self.current_action == NUM_TILE_TYPES + 2 and can_minggang:
                self.logger.debug("PPO智能体选择明杠")
                self.action_counts['minggang'] += 1
                self.send_minggang(new_tile)
            else:
                card_to_play = self.hand_cards[-1]
                if self.current_action < NUM_TILE_TYPES and self.current_action >= 0:
                    card_to_play = self.id_to_tile.get(self.current_action, None)
                    if card_to_play in self.hand_cards:
                        self.logger.debug(f"PPO智能体选择打出牌 {card_to_play}")
                    else:
                        self.logger.info("PPO智能体选择的牌不在手牌中，选择默认出牌")
                        card_to_play = self.hand_cards[-1]     
                else:
                    self.logger.info("PPO智能体决策出错，选择默认出牌")
                    card_to_play = self.hand_cards[-1]
                self.action_counts['discard'] += 1
                self.hand_cards.remove(card_to_play)
                self.last_discard = card_to_play
                self.last_discard_player_id = self.player_id
                self.last_discard_player_index = self.player_index
                self.player_discards[int(self.player_index)].append(card_to_play)
                self.hit_out_card(card_to_play)
    
    def auto_rush(self, rush_card, can_chi, can_peng, can_gang, can_hu):
        # 计算奖励并存储转换, 保证obs为最新。
        reward = self.calculate_reward()
        self.store_transition(reward, False)

        # 初始化牌名映射（如果需要）
        if self.tile_to_id is None or self.id_to_tile is None:
            self.tile_to_id, self.id_to_tile = init_tile_mapping()
        
        # 处理游戏状态，为PPO准备输入
        state_tensor, rush_tile_tensor, turn_tensor = self.process_game_state_for_ppo(
            is_rush_action=True, 
            rush_card=rush_card
        )
        
        # 创建动作掩码 [过, 吃, 碰, 杠, 胡]
        action_mask = torch.zeros(44, dtype=torch.bool)
        action_mask[37] = True  # 过总是可以的
        action_mask[38] = can_peng
        action_mask[39] = can_gang
        action_mask[40] = can_hu
        if can_chi:
            hand_nums = [self.tile_to_id.get(item, NUM_TILE_TYPES) for item in self.hand_cards]
            rush_num = self.tile_to_id[rush_card]
            
            # 检查前吃所需的牌是否在手牌中
            if rush_num < 27 and rush_num % 9 >= 2:
                action_mask[41] = ((rush_num-2) in hand_nums) and ((rush_num-1) in hand_nums)
            
            # 检查中吃所需的牌是否在手牌中
            if rush_num < 27 and rush_num % 9 <= 8 and rush_num % 9 >= 1:
                action_mask[42] = ((rush_num-1) in hand_nums) and ((rush_num+1) in hand_nums)
            
            # 检查后吃所需的牌是否在手牌中
            if rush_num < 27 and rush_num % 9 <= 7:
                action_mask[43] = ((rush_num+1) in hand_nums) and ((rush_num+2) in hand_nums)
        
        action_mask_tensor = action_mask.unsqueeze(0).to(self.device)
        self.current_action_mask = action_mask_tensor
        # 使用PPO智能体选择动作
        with torch.no_grad():
            # 获取特征
            combined_features = self.ppo_agent.feature_extractor(
                state_tensor.unsqueeze(0),
                rush_tile_tensor.unsqueeze(0),
                turn_tensor.unsqueeze(0)
            )
            self.current_feature = combined_features
            
            # 采样动作
            # 注意：这里假设policy网络的输出适用于吃碰杠胡决策
            # 可能需要根据实际情况调整
            action, log_prob, probs = self.ppo_agent.policy.get_action_and_log_prob(
                combined_features, 
                action_mask_tensor
            )
            
            # 估计状态值
            state_value = self.ppo_agent.value(combined_features)
            
            # 保存当前动作信息
            self.current_action = action.item()
            self.current_log_prob = log_prob
            self.current_value = state_value
            
            self.logger.debug(f"PPO智能体动作选择: {self.current_action}, 动作掩码: {action_mask}")
            
            # 执行选择的动作
            if self.current_action == 40 and can_hu:  # 胡
                self.logger.debug("PPO智能体选择胡牌")
                self.action_counts['hu'] += 1
                self.rush_hu(rush_card)
            elif self.current_action == 39 and can_gang:  # 杠
                self.logger.debug("PPO智能体选择杠牌")
                self.action_counts['gang'] += 1
                self.rush_gang(rush_card)
            elif self.current_action == 38 and can_peng:  # 碰
                self.logger.debug("PPO智能体选择碰牌")
                self.action_counts['peng'] += 1
                self.rush_peng(rush_card)
            elif self.current_action in [41, 42, 43] and can_chi:  # 吃
                self.action_counts['chi'] += 1
                hand_nums = [self.tile_to_id.get(card, NUM_TILE_TYPES) for card in self.hand_cards]
                rush_num = self.tile_to_id.get(rush_card, NUM_TILE_TYPES)
                
                if self.current_action == 41 and self.id_to_tile[rush_num-1] in hand_nums and self.id_to_tile[rush_num-2] in hand_nums:
                    self.logger.debug(f"PPO智能体选择前吃: {self.id_to_tile[rush_num-2]}, {self.id_to_tile[rush_num-1]}, {rush_card}")
                    self.rush_chi(self.id_to_tile[rush_num-2], self.id_to_tile[rush_num-1], rush_card)
                elif self.current_action == 42 and self.id_to_tile[rush_num-1] in hand_nums and self.id_to_tile[rush_num+1] in hand_nums:
                    self.logger.debug(f"PPO智能体选择中吃: {self.id_to_tile[rush_num-1]}, {self.id_to_tile[rush_num+1]}, {rush_card}")
                    self.rush_chi(self.id_to_tile[rush_num-1], self.id_to_tile[rush_num+1], rush_card)
                elif self.current_action == 43 and self.id_to_tile[rush_num+1] in hand_nums and self.id_to_tile[rush_num+2] in hand_nums:
                    self.logger.debug(f"PPO智能体选择后吃: {self.id_to_tile[rush_num+1]}, {self.id_to_tile[rush_num+2]}, {rush_card}")
                    self.rush_chi(self.id_to_tile[rush_num+1], self.id_to_tile[rush_num+2], rush_card)
                else:
                    self.logger.info("PPO智能体选择的吃牌组合不存在，选择过")
                    self.rush_skip(rush_card)
            else:  # 过
                self.logger.info("PPO智能体选择过")
                self.action_counts['pass'] += 1
                self.rush_skip(rush_card)
    
    def reset_game_state(self):
        """重置游戏状态"""
        self.game_id = None
        self.player_id = None
        self.player_index = None
        self.hand_cards = []
        self.turn_count = 0
        self.player_discards = {0: [], 1: [], 2: [], 3: []}
        self.player_rush = {0: [], 1: [], 2: [], 3: []}
        self.last_discard = None
        self.last_discard_player = None
        self.last_discard_player_index = None
        self.current_game_score = 0
        # 重置行动统计（只在需要时，比如按局统计时重置）
        # self.action_counts = {key: 0 for key in self.action_counts}
    
        # 重置当前局相关的变量，但保留统计数据
        self.current_feature = None
        self.current_rush_tile_id = None
        self.current_turn = None
        self.current_action_mask = None
        self.current_action = None
        self.current_log_prob = None
        self.current_value = None
        
        # 记录本局的回合数
        if not hasattr(self, 'turns_per_game'):
            self.turns_per_game = []
        self.turns_per_game.append(self.turn_count)

    def send_play_game(self, player_name="Player1"):
        """发送PlayGame命令启动单人游戏"""
        message = f"PlayGame {player_name} Single"
        return self.send_message(message)
    
    def send_ready(self):
        """发送准备就绪信号"""
        message = f"Ready {self.game_id} {self.player_id} {self.player_index}"
        return self.send_message(message)

    def rush_skip(self, card):
        """跳过吃碰杠胡"""
        message = f"RushSkip {self.game_id} {self.player_id} {self.player_index} {card}"
        return self.send_message(message)

    def rush_chi(self, in_card1, in_card2, out_card):
        """吃牌"""
        message = f"RushChi {self.game_id} {self.player_id} {self.player_index} {in_card1} {in_card2} {out_card}"
        return self.send_message(message)

    def rush_peng(self, card):
        """碰牌"""
        message = f"RushPeng {self.game_id} {self.player_id} {self.player_index} {card}"
        return self.send_message(message)
        
    def rush_gang(self, card):
        """杠牌"""
        message = f"RushGang {self.game_id} {self.player_id} {self.player_index} {card}"
        return self.send_message(message)

    def rush_hu(self, card):
        """胡牌"""
        message = f"RushHu {self.game_id} {self.player_id} {self.player_index} {card}"
        return self.send_message(message)
    
    def hit_out_card(self, card):
        """打出一张牌"""
        message = f"HitOut {self.game_id} {self.player_id} {self.player_index} {card}"            
        return self.send_message(message)
    
    def send_zimohu(self):
        """自摸胡牌"""
        message = f"ZimoHu {self.game_id} {self.player_id} {self.player_index}"
        return self.send_message(message)
    
    def send_angang(self, card):
        """暗杠"""
        message = f"AnGang {self.game_id} {self.player_id} {self.player_index} {card}"
        return self.send_message(message)

    def send_minggang(self, card):
        """明杠"""
        message = f"MingGang {self.game_id} {self.player_id} {self.player_index} {card}"
        return self.send_message(message)

    def pad_sequence(self, seq, max_len, pad_value=NUM_TILE_TYPES):
        """将序列填充或截断到指定长度"""
        if len(seq) > max_len:
            return seq[:max_len]
        elif len(seq) < max_len:
            return seq + [pad_value] * (max_len - len(seq))
        else:
            return seq   

    def send_message(self, message):
        """向服务器发送消息"""
        try:
            self.socket.sendall(message.encode('utf-8'))
            logging.debug(f"发送消息: {message}")
            return True
        except Exception as e:
            logging.error(f"发送消息失败:")
            return False
     
    def close(self):
        """关闭连接"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
                logging.info("连接已关闭")
            except:
                pass

    def process_game_state_for_ppo(self, is_rush_action=False, rush_card=None):
        """处理当前游戏状态，生成PPO智能体需要的输入格式"""
        # 准备特征输入
        feature = []
        hand_card_ids = [self.tile_to_id.get(card, NUM_TILE_TYPES) for card in self.hand_cards]
        feature.extend(self.pad_sequence(hand_card_ids, MAX_HAND_SIZE))
        for i in range(4):
            player_rush_ids = [self.tile_to_id.get(card, NUM_TILE_TYPES) for card in self.player_rush[i]]
            player_discard_ids = [self.tile_to_id.get(card, NUM_TILE_TYPES) for card in self.player_discards[i]]
            feature.extend(self.pad_sequence(player_rush_ids, MAX_RUSH_SIZE))
            feature.extend(self.pad_sequence(player_discard_ids, MAX_DISCARD_SIZE))
        
        # 准备上牌ID (rush_tile_id)
        if is_rush_action and rush_card:
            rush_tile_id = self.tile_to_id.get(rush_card, NUM_TILE_TYPES)
        else:
            rush_tile_id = NUM_TILE_TYPES  # 使用填充值表示没有上牌
        
        # 转换为张量
        state_tensor = torch.tensor(feature, dtype=torch.long).to(self.device)
        rush_tile_tensor = torch.tensor(rush_tile_id, dtype=torch.long).to(self.device)
        turn_tensor = torch.tensor(self.turn_count, dtype=torch.long).to(self.device)
        
        # 保存当前状态
        self.current_rush_tile_id = rush_tile_tensor
        self.current_turn = turn_tensor
        
        return state_tensor, rush_tile_tensor, turn_tensor
    
    def calculate_reward(self, game_state=None):
        """计算当前状态的奖励"""
        # 基础奖励 - 游戏分数变化
        reward = self.current_game_score - self.last_reward
        self.last_reward = self.current_game_score
        
        # 记录奖励到当前动作类型
        if hasattr(self, 'recent_action_type') and self.recent_action_type is not None:
            if self.recent_action_type in self.rewards_by_action:
                self.rewards_by_action[self.recent_action_type].append(reward)
            self.recent_action_type = None
        
        # 额外奖励
        if game_state:
            # 游戏结束额外奖励
            if game_state.get("game_over", True):
                # 记录最终游戏分数
                if not hasattr(self, 'final_game_scores'):
                    self.final_game_scores = []
                self.final_game_scores.append(self.current_game_score)
        
        return reward
    
    def store_transition(self, reward, done=False):
        """存储当前转换到内存"""
        if self.current_feature is not None and self.current_action is not None:
            self.memory.push(
                self.current_feature.detach(),  # 特征
                self.current_action,            # 动作
                self.current_log_prob.detach(), # 对数概率
                reward,                         # 奖励
                self.current_value.detach(),    # 状态值
                1 - float(done),                # 掩码 (not done)
                self.current_action_mask        # 动作掩码
            )
            self.episode_reward += reward
            
            # 重置当前状态
            if done:
                self.current_feature = None
                self.current_action = None
                self.current_log_prob = None
                self.current_value = None
    
    def update_agent(self):
        """更新智能体的策略"""
        if len(self.memory) > 0:
            self.logger.info(f"使用 {len(self.memory)} 个样本更新智能体")
            # 获取训练统计数据
            stats = self.ppo_agent.update_policy(self.memory, batch_size=64, epochs=10, return_stats=True)
            self.ppo_agent.copy_policy_to_old_policy()
            
            # 保存训练统计数据
            for key, value in stats.items():
                if key in self.ppo_stats:
                    self.ppo_stats[key].append(value)
                    
            self.memory.clear()
            self.logger.info("智能体更新完成")
    
    def save_model(self, filepath):
        """保存当前智能体模型"""
        try:
            torch.save({
                'feature_extractor': self.ppo_agent.feature_extractor.state_dict(),
                'policy': self.ppo_agent.policy.state_dict(),
                'value': self.ppo_agent.value.state_dict(),
                'episode_count': self.episode_count
            }, filepath)
            self.logger.info(f"模型已保存到 {filepath}")
        except Exception as e:
            self.logger.info(f"保存模型失败: {e}")
    
    def load_model(self, filepath):
        """加载智能体模型"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.ppo_agent.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
            self.ppo_agent.policy.load_state_dict(checkpoint['policy'])
            self.ppo_agent.value.load_state_dict(checkpoint['value'])
            self.ppo_agent.copy_policy_to_old_policy()
            self.episode_count = checkpoint.get('episode_count', 0)
            self.logger.info(f"成功加载模型: {filepath}, 已训练回合: {self.episode_count}")
            return True
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            return False

    def print_episode_stats(self):
        """打印单局游戏的统计信息"""
        avg_recent_reward = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0
        avg_recent_score = sum(self.recent_scores) / len(self.recent_scores) if self.recent_scores else 0
        avg_win_rate = sum(self.recent_win_rate) / len(self.recent_win_rate) if self.recent_win_rate else 0
        
        episode_time = time.time() - self.episode_start_time
        
        self.logger.info(f"===== 游戏 #{self.episode_count} 统计 =====")
        self.logger.info(f"本局奖励: {self.episode_reward:.2f}, 得分: {self.current_game_score}")
        self.logger.info(f"本局回合数: {self.turn_count}, 用时: {episode_time:.1f}秒")
        self.logger.info(f"最近{len(self.recent_rewards)}局平均奖励: {avg_recent_reward:.2f}")
        self.logger.info(f"最近{len(self.recent_scores)}局平均分数: {avg_recent_score:.2f}")
        self.logger.info(f"最近{len(self.recent_win_rate)}局胜率: {avg_win_rate:.2f}")
        
        # 打印本局行动统计
        self.logger.info(f"行动统计: 打牌:{self.action_counts['discard']}, "
                         f"胡:{self.action_counts['hu']}, 自摸:{self.action_counts['zimohu']}, "
                         f"碰:{self.action_counts['peng']}, 杠:{self.action_counts['gang']}, "
                         f"吃:{self.action_counts['chi']}, 过:{self.action_counts['pass']}")
    
    def print_training_stats(self):
        """打印训练统计信息"""
        self.logger.info(f"===== 训练统计 (游戏 #{self.episode_count}) =====")
        self.logger.info(f"总训练时间: {self.total_training_time/3600:.2f}小时")
        
        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0
        avg_length = sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0
        win_rate = self.total_wins / self.episode_count if self.episode_count > 0 else 0
        
        self.logger.info(f"总游戏局数: {self.episode_count}, 胜局: {self.total_wins}, 胜率: {win_rate:.2f}")
        self.logger.info(f"平均每局奖励: {avg_reward:.2f}, 平均回合数: {avg_length:.1f}")
        
        # 打印PPO损失统计
        if self.ppo_stats['policy_loss']:
            recent_policy_loss = self.ppo_stats['policy_loss'][-1] if self.ppo_stats['policy_loss'] else 0
            recent_value_loss = self.ppo_stats['value_loss'][-1] if self.ppo_stats['value_loss'] else 0
            recent_entropy = self.ppo_stats['entropy'][-1] if self.ppo_stats['entropy'] else 0
            
            self.logger.info(f"PPO损失 - 策略: {recent_policy_loss:.4f}, "
                             f"价值: {recent_value_loss:.4f}, 熵: {recent_entropy:.4f}")
    
    def get_training_stats(self):
        """获取所有训练统计数据，用于可视化"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'win_rate': self.total_wins / self.episode_count if self.episode_count > 0 else 0,
            'action_counts': self.action_counts,
            'ppo_stats': self.ppo_stats,
            'total_games': self.episode_count,
            'total_wins': self.total_wins,
            'total_training_time': self.total_training_time
        }
        return stats

    def start(self, load_model_path=None):
        """启动客户端和训练过程"""
        # 加载模型（如果指定）
        if load_model_path:
            self.load_model(load_model_path)
            
        # 初始化牌名映射
        self.tile_to_id, self.id_to_tile = init_tile_mapping()
        
        # 连接服务器和启动游戏
        if self.connect_to_server():
            self.logger.info("客户端已连接到服务器，准备开始游戏")
            return True
        return False

# 在文件末尾添加运行函数
def run_mahjong_client(server_ip="127.0.0.1", server_port=5000, load_model_path=None, save_interval=10, player_name="PPOPlayer"):
    """
    运行麻将客户端进行训练
    
    参数:
        server_ip: 服务器IP地址
        server_port: 服务器端口
        load_model_path: 要加载的模型路径，若为None则从头开始训练
        save_interval: 保存模型的回合间隔
        player_name: 玩家名称
    """
    try:
        # 创建智能体
        agent = MahjongPPO(device="cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建客户端
        client = MahjongClient(server_ip, server_port, client_port=5001, agent=agent)
        client.update_frequency = save_interval
        
        # 自动连接到服务器
        if client.connect_to_server():
            client.logger.info("客户端已连接到服务器，准备开始游戏")
            
            # 启动游戏
            client.send_play_game(player_name)
            
            # 主循环 - 保持程序运行
            try:
                while True:
                    if client.game_completed:
                        client.logger.info("游戏已完成，准备开始下一局")
                        client.send_play_game(player_name)
                        client.game_completed = False
                    time.sleep(1)
            except KeyboardInterrupt:
                client.logger.info("接收到停止信号，正在关闭...")
            finally:
                # 保存最终模型
                client.save_model(f"mahjong_ppo_model_final.pt")
                client.close()
        else:
            client.logger.error("连接服务器失败")
            return False
        
        return True
    except Exception as e:
        client.logger.error(f"运行麻将客户端时发生错误: {e}")
        traceback.print_exc()
        return False

# 直接调用方便测试
# if __name__ == "__main__":
    # run_mahjong_client()
