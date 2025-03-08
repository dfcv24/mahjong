import sys
import os
import socket
import threading
import time
import json
import torch
import traceback

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.model import MahjongDiscardModel, MahjongActionModel, MahjongTotalModel
from src.utils.constants import *
from src.utils.tile_utils import init_tile_mapping, sort_hand_cards

class MahjongClientServer:
    def __init__(self, server_ip="127.0.0.1", server_port=5000, client_port=5001):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_ip = "127.0.0.1"
        self.client_port = client_port
        self.player_name = "luu"
        self.game_id = None
        self.player_id = None  # 玩家名称/ID
        self.player_index = None  # 玩家在牌桌上的位置
        self.hand_cards = []
        self.need_discard = False
        self.need_rush = False
        self.socket = None
        self.receive_thread = None
        self.running = False
        self.turn_count = 0  # 记录游戏回合数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tile_to_id = None  # 将在load_ai_model中初始化
        self.id_to_tile = None  # 将在load_ai_model中初始化
        self.game_completed = False
        self.last_game_result = None
        
        # 游戏状态跟踪
        self.player_rush = {0: [], 1: [], 2: [], 3: []}  # 吃碰杠胡历史
        self.player_discards = {0: [], 1: [], 2: [], 3: []}  # 每个位置玩家的出牌历史
        self.last_discard = None  # 最后一张出牌
        self.last_discard_player_id = None  # 最后出牌的玩家id
        self.last_discard_player_index = None  # 最后出牌的玩家index

        self.current_game_score = 0  # 当前游戏得分
        # 初始化牌名映射
        self.tile_to_id, self.id_to_tile = init_tile_mapping()
        self.can_zimohu = False
        self.can_hu = False
        self.rush_card = None
    
    def connect_to_server(self):
        """连接到麻将服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            print(f"成功连接到服务器 {self.server_ip}:{self.server_port}")
            
            # 启动接收消息的线程
            self.running = True
            self.receive_thread = threading.Thread(target=self.receive_messages)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            return True
        except Exception as e:
            print(f"连接服务器失败: {e}")
            return False
    
    def receive_messages(self):
        """接收服务器消息的线程"""
        buffer = ""
        while self.running:
            try:
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    print("服务器连接已关闭")
                    self.running = False
                    break
                
                print(f"收到原始数据: {data!r}")  # 以原始格式打印，包括特殊字符
                # 将整个消息直接传递给处理函数
                self.handle_message(data)
            except Exception as e:
                print(f"接收消息时出错: {e}")
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
                print(f"游戏开始: ID={self.game_id}, 玩家ID={self.player_id}, 牌桌位置={self.player_index}")
                # 自动回复准备就绪
                self.send_ready()
        elif parts[0] == "HandPaiList":
            if len(parts) > 4:  # 消息头 + 游戏信息(3) + 至少一张牌                
                # 从第5个元素开始是手牌
                cards = parts[4:]
                self.hand_cards = cards
                print(f"收到初始手牌: {self.hand_cards}")
                print(f"手牌数量: {len(self.hand_cards)}")
                self.send_ready()                
        # NextPai 游戏编号 玩家编号 玩家序号 牌名 是否能胡 是否能暗杠 是否能明杠- 当前玩家摸牌
        elif parts[0] == "NextPai" or parts[0] == "GangPai":
            if len(parts) >= 8:
                new_tile = parts[4]
                self.can_zimohu = parts[5] == "True"
                can_angang = parts[6] == "True"
                can_minggang = parts[7] == "True"
                # 增加回合计数
                self.turn_count += 1
                # 将新牌添加到手牌中
                if new_tile not in ["", "None"]:
                    self.hand_cards.append(new_tile)
                    self.hand_cards = sort_hand_cards(self.hand_cards)
                
                print(f"摸到新牌: {new_tile}, 可以自摸胡: {self.can_zimohu}, 可以暗杠: {can_angang}, 可以明杠: {can_minggang}")
                print(f"当前手牌: {self.hand_cards} (共{len(self.hand_cards)}张)")
                
                self.need_discard = True
                # 等待处理
        # NextPaiOther 游戏编号 玩家编号 玩家序号 拿牌玩家编号 拿牌玩家序号 - 其他玩家摸牌
        elif parts[0] == "NextPaiOther" or parts[0] == "GangPaiOther":
            self.turn_count += 1
            if len(parts) >= 6:
                draw_player_id = parts[4]
                draw_player_index = parts[5]
                if parts[0] == "NextPaiOther":
                    print(f"玩家 {draw_player_id} (位置 {draw_player_index}) 摸了一张牌")
                else:
                    print(f"玩家 {draw_player_id} (位置 {draw_player_index}) 杠了一张牌")
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
                
                print(f"玩家 {hit_player_id} (位置 {hit_player_index}) 打出了: {card}")
                
                # 回复服务器已准备好
                self.send_ready()
        # ChiPengGangHu 游戏编号 玩家编号 玩家序号 是否有吃 是否有碰 是否有杠 是否有胡 牌名
        elif parts[0] == "ChiPengGangHu":
            if len(parts) >= 9:
                can_chi = parts[4] == "True"
                can_peng = parts[5] == "True"
                can_gang = parts[6] == "True"
                self.can_hu = parts[7] == "True"
                self.rush_card = parts[8]
                
                print(f"收到吃碰杠胡请求: {self.rush_card}, 可吃: {can_chi}, 可碰: {can_peng}, 可杠: {can_gang}, 可胡: {self.can_hu}")
                print(f"当前手牌: {self.hand_cards}")
                self.need_rush = True
                # 等待处理
        # NotifyRushPeng 游戏编号 玩家编号 玩家序号 碰牌玩家编号 碰牌玩家序号 碰牌
        elif parts[0] == "NotifyRushPeng":
            if len(parts) >= 7:
                peng_player_id = parts[4]
                peng_player_index = parts[5]
                peng_tile = parts[6]
                
                print(f"玩家 {peng_player_id} (位置 {peng_player_index}) 碰了牌: {peng_tile}")
                self.player_rush[int(peng_player_index)].extend([peng_tile, peng_tile, peng_tile, "null"])
                # 如果是自己碰牌，先移除碰牌，再出牌
                if peng_player_id == self.player_id:
                    self.hand_cards.remove(peng_tile)
                    self.hand_cards.remove(peng_tile)
                    self.need_discard = True
                    # 等待处理
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
                    print(f"玩家 {gang_player_id} (位置 {gang_player_index}) 杠了牌: {rush_tile} (抽牌: {gang_tile})")
                else:
                    print(f"玩家 {gang_player_id} (位置 {gang_player_index}) 杠了牌: {rush_tile})")
                
                self.player_rush[int(gang_player_index)].extend([rush_tile, rush_tile, rush_tile, rush_tile])
                # 如果是自己杠牌，先移除rush牌，杠来的牌加入手牌，再出牌
                if gang_player_id == self.player_id:
                    self.hand_cards.remove(rush_tile)
                    self.hand_cards.remove(rush_tile)
                    self.hand_cards.remove(rush_tile)
                    self.hand_cards.append(gang_tile)
                    self.hand_cards = sort_hand_cards(self.hand_cards)
                    self.need_discard = True
                    # 等待处理
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
                
                print(f"玩家 {chi_player_id} (位置 {chi_player_index}) 吃了牌组合: {chi_tile1} {chi_tile2} {chi_tile3}")
                self.player_rush[int(chi_player_index)].extend([chi_tile1, chi_tile2, chi_tile3, "null"])
                # 如果是自己吃牌，移除吃牌，再出牌
                if chi_player_id == self.player_id:
                    self.hand_cards.remove(chi_tile1)
                    self.hand_cards.remove(chi_tile2)
                    self.need_discard = True
                    # 等待处理
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
                
                print(f"玩家 {hu_player_id} (位置 {hu_player_index}) 胡牌了!")
                print(f"胡牌玩家手牌: {winning_hand}")
                
                # 玩家得分等信息
                if hu_player_id == self.player_id:
                    self.current_game_score += 8  # 胡牌得分
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
                
                print(f"玩家 {hu_player_id} (位置 {hu_player_index}) 胡牌了!")
                print(f"胡牌玩家手牌: {winning_hand}")
            # 玩家得分等信息
            if hu_player_id == self.player_id:
                self.current_game_score = 9
            else:
                self.current_game_score = -3
            self.send_ready()
        elif parts[0] == "NotifyZimoAnGang":
            gang_player_id = parts[4]
            gang_player_index = parts[5]
            if len(parts) >= 7 and gang_player_id == self.player_id:
                gang_tile = parts[6]
                print(f"玩家 {gang_player_id} (位置 {gang_player_index}) 暗杠了牌: {gang_tile}")
                self.player_rush[int(gang_player_index)].extend([gang_tile, gang_tile, gang_tile, gang_tile])
                self.hand_cards.remove(gang_tile)
                self.hand_cards.remove(gang_tile)
                self.hand_cards.remove(gang_tile)
            else:
                print(f"玩家 {gang_player_id} (位置 {gang_player_index}) 暗杠了牌")
            self.send_ready()
        elif parts[0] == "NotifyZimoMingGang":
            gang_player_id = parts[4]
            gang_player_index = parts[5]
            gang_tile = parts[6]
            print(f"玩家 {gang_player_id} (位置 {gang_player_index}) 明杠了牌: {gang_tile}")
            # 找到原来3张gang_tile的位置，后面加上gang_tile，替换null牌
            for i in range(len(self.player_rush[int(gang_player_index)])):
                if self.player_rush[int(gang_player_index)][i] == gang_tile and self.player_rush[int(gang_player_index)][i+1] == "null":
                    self.player_rush[int(gang_player_index)][i+1] = gang_tile
                    break
            self.send_ready()
        # 处理游戏结束消息
        elif parts[0] == "GameFinished" or parts[0] == "GameOver":
            print("游戏结束")
            result_info = {}
            
            # 尝试解析更多结束信息
            if len(parts) > 4:  # 至少包含游戏ID、玩家ID、位置和结果
                result_info["winner"] = parts[4] if len(parts) > 4 else "unknown"
                
                # 如果有更多信息，如分数等
                if len(parts) > 5:
                    try:
                        scores = [int(s) for s in parts[5:] if s.lstrip('-').isdigit()]
                        result_info["scores"] = scores
                    except:
                        pass
            
            self.last_game_result = result_info
            self.game_completed = True
            
            # 设置获胜玩家
            self.win_player = result_info.get("winner", "unknown")
            
            # 如果有分数信息，设置分数变化
            if "scores" in result_info:
                self.score_changes = result_info["scores"]
                # 找出当前玩家的分数
                if self.player_index is not None and 0 <= int(self.player_index) < len(self.score_changes):
                    self.current_game_score = self.score_changes[int(self.player_index)]
            
            print(f"游戏结束详情: {result_info}")
            self.send_ready()
            # self.reset_game_state()
    
    def reset_game_state(self):
        """重置游戏状态，清除之前游戏的所有数据"""
        # 重置基本游戏状态
        self.hand_cards = []
        self.player_rush = {0: [], 1: [], 2: [], 3: []}
        self.player_discards = {0: [], 1: [], 2: [], 3: []}
        self.last_discard = None
        self.last_discard_player_id = None
        self.last_discard_player_index = None
        self.turn_count = 0
        self.is_discard = True
        
        # 重置游戏结果相关变量
        self.game_completed = False
        self.last_game_result = None
        self.current_game_score = 0
        self.win_player = None
        self.score_changes = None
        self.hand_score = None
        
        # 状态变化标志
        self._last_state_hash = None
        
        # 游戏能力标志
        self.can_zimohu = False
        self.can_hu = False
        
        print("游戏状态已重置")
    
    def handle_game_over(self, data):
        """处理游戏结束消息"""
        self.game_completed = True
        
        # 提取游戏结果信息
        if 'result' in data:
            self.last_game_result = data['result']
        
        # 提取获胜玩家
        if 'winPlayer' in data:
            self.win_player = data['winPlayer']
            
        # 提取分数变化
        if 'scoreChanges' in data:
            self.score_changes = data['scoreChanges']
            
            # 找到当前玩家的分数变化
            if self.player_index is not None and 0 <= int(self.player_index) < len(self.score_changes):
                self.current_game_score = self.score_changes[int(self.player_index)]
                
        # 提取手牌分数
        if 'handScore' in data:
            self.hand_score = data['handScore']
            
        print(f"游戏结束，结果: {self.last_game_result}, 当前玩家得分: {self.current_game_score}")
    
    def has_state_changed(self):
        """检查游戏状态是否有变化"""
        # 计算当前状态的哈希值
        current_state = (
            tuple(sorted(self.hand_cards)),
            tuple(tuple(self.player_rush[i]) for i in range(4)),
            tuple(tuple(self.player_discards[i]) for i in range(4)),
            self.last_discard,
            self.last_discard_player_id,
            self.turn_count
        )
        current_hash = hash(current_state)
        
        # 检查状态是否变化
        changed = current_hash != self._last_state_hash
        
        # 更新状态哈希
        self._last_state_hash = current_hash
        
        return changed

    def send_play_game(self, player_name="Player1"):
        """发送PlayGame命令启动单人游戏"""
        message = f"PlayGame {player_name} Single"
        result = self.send_message(message)
        if result:
            print(f"已请求开始新游戏，玩家名: {player_name}")
        return result
    
    def send_ready(self):
        """发送准备就绪信号"""
        message = f"Ready {self.game_id} {self.player_id} {self.player_index}"
        return self.send_message(message)

    def rush_skip(self):
        """跳过吃碰杠胡"""
        message = f"RushSkip {self.game_id} {self.player_id} {self.player_index}"
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
            print(f"已发送: {message}")
            return True
        except Exception as e:
            print(f"发送消息失败: {e}")
            return False
     
    def close(self):
        """关闭连接"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
                print("已关闭与服务器的连接")
            except:
                pass
