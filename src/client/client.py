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

from src.models.model import MahjongDiscardModel, MahjongActionModel
from src.utils.constants import *
from src.utils.tile_utils import init_tile_mapping, sort_hand_cards

class MahjongClient:
    def __init__(self, server_ip="127.0.0.1", server_port=5000, client_port=5001):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_ip = "127.0.0.1"
        self.client_port = client_port
        self.game_id = None
        self.player_id = None  # 玩家名称/ID
        self.player_index = None  # 玩家在牌桌上的位置
        self.hand_cards = []
        self.discard_model = None
        self.action_model = None  # 吃碰杠胡决策模型
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
        # self.discard_history = []  # 所有玩家的出牌历史
        self.player_rush = {0: [], 1: [], 2: [], 3: []}  # 吃碰杠胡历史
        self.player_discards = {0: [], 1: [], 2: [], 3: []}  # 每个位置玩家的出牌历史
        self.last_discard = None  # 最后一张出牌
        self.last_discard_player_id = None  # 最后出牌的玩家id
        self.last_discard_player_index = None  # 最后出牌的玩家index

        self.current_game_score = 0  # 当前游戏得分
    
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
    
    def load_ai_models(self):
        """加载AI模型和动作决策模型"""
        # 读取配置文件
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config", "model_config.json")
           
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            discard_model_path = config["discard_model_path"]
            action_model_path = config["action_model_path"]
            
            # 初始化牌名映射
            self.tile_to_id, self.id_to_tile = init_tile_mapping()
            
            # 加载出牌决策模型
            print(f"加载出牌模型: {discard_model_path}")
            checkpoint = torch.load(discard_model_path, map_location=self.device)
            discard_model = MahjongDiscardModel().to(self.device)
            discard_model.load_state_dict(checkpoint['model_state_dict'])
            discard_model.eval()  # 设置为评估模式
            self.discard_model = discard_model
            
            # 加载吃碰杠胡决策模型
            print(f"加载动作决策模型: {action_model_path}")
            action_checkpoint = torch.load(action_model_path, map_location=self.device)
            action_model = MahjongActionModel().to(self.device)
            action_model.load_state_dict(action_checkpoint['model_state_dict'])
            action_model.eval()  # 设置为评估模式
            self.action_model = action_model
            
            print("所有AI模型加载成功")
            return True
        except Exception as e:
            print(f"加载AI模型失败: {e}")
            return False
        
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
                can_zimohu = parts[5] == "True"
                can_angang = parts[6] == "True"
                can_minggang = parts[7] == "True"
                # 增加回合计数
                self.turn_count += 1       
                # 将新牌添加到手牌中
                if new_tile not in ["", "None"]:
                    self.hand_cards.append(new_tile)
                    self.hand_cards = sort_hand_cards(self.hand_cards)
                
                print(f"摸到新牌: {new_tile}, 可以自摸胡: {can_zimohu}, 可以暗杠: {can_angang}, 可以明杠: {can_minggang}")
                print(f"当前手牌: {self.hand_cards} (共{len(self.hand_cards)}张)")
                
                # 如果可以胡且AI模型已加载，考虑自动胡牌
                self.auto_play(new_tile, can_zimohu, can_angang, can_minggang)
        # NextPaiOther 游戏编号 玩家编号 玩家序号 拿牌玩家编号 拿牌玩家序号 - 其他玩家摸牌
        elif parts[0] == "NextPaiOther" or parts[0] == "GangPaiOther":
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
                
                # 记录出牌历史
                # self.discard_history.append((hit_player_id, card))
                
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
                can_hu = parts[7] == "True"
                rush_card = parts[8]
                
                print(f"收到吃碰杠胡请求: {rush_card}, 可吃: {can_chi}, 可碰: {can_peng}, 可杠: {can_gang}, 可胡: {can_hu}")
                print(f"当前手牌: {self.hand_cards}")
                
                # 如果已加载AI动作模型，使用模型自动响应
                if self.action_model:
                    self.auto_rush(rush_card, can_chi, can_peng, can_gang, can_hu)
                else:
                    # 使用简单规则
                    self.auto_rush_simple(rush_card, can_chi, can_peng, can_gang, can_hu)
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
                
                print(f"玩家 {chi_player_id} (位置 {chi_player_index}) 吃了牌组合: {chi_tile1} {chi_tile2} {chi_tile3}")
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
        elif parts[0] == "GameFinished":
            print("游戏结束")
            self.last_game_result = {
                "winner": parts[-1],
                "scores": self.current_game_score
            }
            self.game_completed = True
            self.send_ready()
            self.reset_game_state()
            # 回复服务器已准备好
    
    def auto_play(self, new_tile="null", can_zimohu=False, can_angang=False, can_minggang=False):
        # 将牌转换为模型输入格式
        feature = []
        hand_card_ids = [self.tile_to_id.get(card, NUM_TILE_TYPES) for card in self.hand_cards]
        feature.extend(self.pad_sequence(hand_card_ids, MAX_HAND_SIZE))
        for i in range(4):
            player_rush_ids = [self.tile_to_id.get(card, NUM_TILE_TYPES) for card in self.player_rush[i]]
            player_discard_ids = [self.tile_to_id.get(card, NUM_TILE_TYPES) for card in self.player_discards[i]]
            feature.extend(self.pad_sequence(player_rush_ids, MAX_RUSH_SIZE))
            feature.extend(self.pad_sequence(player_discard_ids, MAX_DISCARD_SIZE))
        feature_tensor = torch.tensor([feature], dtype=torch.long)

        # 准备回合数输入 - 修改这里，确保是整数类型
        turn_tensor = torch.tensor([self.turn_count], dtype=torch.long)
        
        # 创建动作掩码张量
        action_mask = torch.zeros((1, NUM_TILE_TYPES + 3), dtype=torch.bool)
        # 对手牌中的每张牌设置掩码为True
        for tile in self.hand_cards:
            tile_id = self.tile_to_id.get(tile, NUM_TILE_TYPES)  # 如果牌不在映射中，使用填充值
            if tile_id < NUM_TILE_TYPES:  # 确保是有效的牌ID
                action_mask[0, tile_id] = True
        
        # 设置特殊动作掩码
        action_mask[0, NUM_TILE_TYPES] = can_zimohu
        action_mask[0, NUM_TILE_TYPES + 1] = can_angang
        action_mask[0, NUM_TILE_TYPES + 2] = can_minggang

        # 移动到设备
        feature_tensor = feature_tensor.to(self.device)
        turn_tensor = turn_tensor.to(self.device)
        action_mask = action_mask.to(self.device)
        
        # 打印调试信息
        print(f"牌索引张量类型: {feature_tensor.dtype}, 形状: {feature_tensor.shape}")
        print(f"回合张量类型: {turn_tensor.dtype}, 形状: {turn_tensor.shape}")
        print(f"动作掩码张量类型: {action_mask.dtype}, 形状: {action_mask.shape}")
        
        # 模型预测
        with torch.no_grad():
            logits = self.discard_model(feature_tensor, turn_tensor, action_mask)
            _, predicted = torch.max(logits, 1)
            if can_zimohu and predicted[0].item() == NUM_TILE_TYPES:
                print("AI选择自摸胡牌")
                self.send_zimohu()
            elif can_angang and predicted[0].item() == NUM_TILE_TYPES + 1:
                print("AI选择暗杠")
                self.send_angang(new_tile)
            elif can_minggang and predicted[0].item() == NUM_TILE_TYPES + 2:
                print("AI选择明杠")
                self.send_minggang(new_tile)
            else:
                if predicted[0] < NUM_TILE_TYPES:
                    card_to_play = self.id_to_tile[predicted[0].item()]
                else:
                    print("AI决策出错, 选择默认出牌")
                    card_to_play = self.hand_cards[-1]
                print("AI选择打出牌", card_to_play)
                self.hand_cards.remove(card_to_play)
                self.last_discard = card_to_play
                self.last_discard_player_id = self.player_id
                self.last_discard_player_index = self.player_index
                self.player_discards[int(self.player_index)].append(card_to_play)
                self.hit_out_card(card_to_play)
    
    def auto_rush(self, rush_card, can_chi, can_peng, can_gang, can_hu):
        """AI自动决策吃碰杠胡"""
        if not self.action_model:
            # 如果没有加载动作决策模型，使用简单规则
            return self.auto_rush_simple(rush_card, can_chi, can_peng, can_gang, can_hu)
        # try:
        # 准备模型输入
        feature = []
        hand_card_ids = [self.tile_to_id.get(card, NUM_TILE_TYPES) for card in self.hand_cards]
        feature.extend(self.pad_sequence(hand_card_ids, MAX_HAND_SIZE))
        for i in range(4):
            player_rush_ids = [self.tile_to_id.get(card, NUM_TILE_TYPES) for card in self.player_rush[i]]
            player_discard_ids = [self.tile_to_id.get(card, NUM_TILE_TYPES) for card in self.player_discards[i]]
            feature.extend(self.pad_sequence(player_rush_ids, MAX_RUSH_SIZE))
            feature.extend(self.pad_sequence(player_discard_ids, MAX_DISCARD_SIZE))
        feature_tensor = torch.tensor([feature], dtype=torch.long)
        
        # 准备rush牌
        rush_id = self.tile_to_id.get(rush_card, NUM_TILE_TYPES)  # 如果牌不在映射中，使用填充值
        rush_tensor = torch.tensor([rush_id], dtype=torch.long)
        # 准备回合数
        turn_tensor = torch.tensor([self.turn_count], dtype=torch.long)
        
        # 创建允许动作的掩码向量 [过, 吃, 碰, 杠, 胡]
        # 注意：顺序变更为与模型训练时一致
        action_mask_tensor = torch.tensor([[
            True,       # 过总是允许的
            can_chi,    # 吃
            can_peng,   # 碰 
            can_gang,   # 杠
            can_hu      # 胡
        ]], dtype=torch.bool)  # 确保使用布尔类型

        # 准备吃牌掩码
        # chi_type的mask - 检查手牌中是否包含吃牌所需的两张牌
        chi_mask = torch.zeros((1,3), dtype=torch.bool)  # 对应前吃、中吃、后吃三种类型
        
        if can_chi:
            hand_ids = [self.tile_to_id.get(card, NUM_TILE_TYPES) for card in self.hand_cards]
            
            # 检查前吃所需的牌是否在手牌中 (需要rush_id-2和rush_id-1)
            if rush_id < 27 and rush_id % 9 >= 2:
                chi_mask[0,0] = ((rush_id-2) in hand_ids) and ((rush_id-1) in hand_ids)
            
            # 检查中吃所需的牌是否在手牌中 (需要rush_id-1和rush_id+1)
            if rush_id < 27 and rush_id % 9 <= 8 and rush_id % 9 >=1:  # 确保不跨花色
                chi_mask[0,1] = ((rush_id-1) in hand_ids) and ((rush_id+1) in hand_ids)
            
            # 检查后吃所需的牌是否在手牌中 (需要rush_id+1和rush_id+2)
            if rush_id < 27 and rush_id % 9 <= 7:  # 确保不跨花色
                chi_mask[0,2] = ((rush_id+1) in hand_ids) and ((rush_id+2) in hand_ids)
        chi_mask_tensor = torch.tensor(chi_mask, dtype=torch.bool)

        # 打印调试信息
        print(f"牌索引张量类型: {feature_tensor.dtype}, 形状: {feature_tensor.shape}")
        print(f"Rush牌张量类型: {rush_tensor.dtype}, 形状: {rush_tensor.shape}")
        print(f"回合张量类型: {turn_tensor.dtype}, 形状: {turn_tensor.shape}")
        print(f"动作掩码张量类型: {action_mask_tensor.dtype}, 形状: {action_mask_tensor.shape}")
        print(f"动作掩码内容: [过={action_mask_tensor[0,0]}, 吃={action_mask_tensor[0,1]}, 碰={action_mask_tensor[0,2]}, 杠={action_mask_tensor[0,3]}, 胡={action_mask_tensor[0,4]}]")
        print(f"吃牌掩码张量类型: {chi_mask_tensor.dtype}, 形状: {chi_mask_tensor.shape}")
        print(f"吃牌掩码内容: [前吃={chi_mask_tensor[0,0]}, 中吃={chi_mask_tensor[0,1]}, 后吃={chi_mask_tensor[0,2]}]")

        # 移动到设备
        feature_tensor = feature_tensor.to(self.device)
        rush_tensor = rush_tensor.to(self.device)
        turn_tensor = turn_tensor.to(self.device)
        action_mask_tensor = action_mask_tensor.to(self.device)
        chi_mask_tensor = chi_mask_tensor.to(self.device)
        
        # 模型预测
        with torch.no_grad():
            action_logits, chi_logits = self.action_model(feature_tensor, rush_tensor, turn_tensor, action_mask_tensor, chi_mask_tensor)
            
            # 获取动作概率
            action_probs = torch.softmax(action_logits, dim=1)
            
            # 打印动作概率，便于调试
            print(f"动作概率: [过={action_probs[0,0].item():.4f}, 吃={action_probs[0,1].item():.4f}, 碰={action_probs[0,2].item():.4f}, 杠={action_probs[0,3].item():.4f}, 胡={action_probs[0,4].item():.4f}]")
            
            # 动作索引: 0-过, 1-吃, 2-碰, 3-杠, 4-胡
            action_idx = torch.argmax(action_probs, dim=1).item()
            print(f"选择的动作索引: {action_idx}")
            
            # 如果选择的是吃，还需要决定用哪两张牌
            if action_idx == 1 and can_chi:  # 选择吃
                # 从chi_logits预测两张牌的位置
                print("处理吃牌逻辑...")
                _, chi_index = torch.max(chi_logits, dim=1)
                chi_index = chi_index.item()
                print(f"选择的吃牌方式0:前吃,1:中吃,2:后吃 {chi_index}")
                if chi_index == 0:
                    chi_tile_ids = [rush_id-2, rush_id-1]
                elif chi_index == 1:
                    chi_tile_ids = [rush_id-1, rush_id+1]
                else:
                    chi_tile_ids = [rush_id+1, rush_id+2]
                chi_tiles = [self.id_to_tile.get(tile_id, "null") for tile_id in chi_tile_ids]
                print(f"选择的吃牌: {chi_tiles}")
                if all(tile in self.hand_cards for tile in chi_tiles):
                    self.rush_chi(chi_tiles[0], chi_tiles[1], rush_card)
                else:
                    print("AI选择吃牌,但是没有给出正确的牌,选择跳过")
                    self.rush_skip(rush_card)
            # 基于动作索引执行相应操作
            if action_idx == 4 and can_hu:  # 胡
                print("AI选择胡牌")
                self.rush_hu(rush_card)
            elif action_idx == 3 and can_gang:  # 杠
                print("AI选择杠牌")
                # self.hand_cards.remove(rush_card)
                # self.hand_cards.remove(rush_card)
                # self.hand_cards.remove(rush_card)
                self.rush_gang(rush_card)
            elif action_idx == 2 and can_peng:  # 碰
                print("AI选择碰牌")
                # self.hand_cards.remove(rush_card)
                # self.hand_cards.remove(rush_card)
                self.rush_peng(rush_card)
            else:  # 过
                print("AI选择跳过")
                self.rush_skip(rush_card)
                
        # except Exception as e:
        #     print(f"AI决策吃碰杠胡出错: {e}")
        #     # 如果出错，使用简单规则
        #     return self.auto_rush_simple(rush_card, can_chi, can_peng, can_gang, can_hu)
    
    def auto_rush_simple(self, rush_card, can_chi, can_peng, can_gang, can_hu):
        """使用简单规则决策吃碰杠胡"""
        # 简单逻辑：优先级 胡 > 杠 > 碰 > 吃 > 跳过
        time.sleep(1)  # 模拟思考时间
        
        if can_hu:
            print("AI选择胡牌(简单规则)")
            self.rush_hu(rush_card)
        elif can_gang:
            print("AI选择杠牌(简单规则)") 
            self.rush_gang(rush_card)
        elif can_peng:
            print("AI选择碰牌(简单规则)")
            self.rush_peng(rush_card)
        elif can_chi:
            print("AI选择吃牌(简单规则)=跳过")
            self.rush_skip(rush_card)
        else:
            print("AI选择跳过(简单规则)")
            self.rush_skip(rush_card)
    
    def reset_game_state(self):
        """重置游戏状态"""
        self.game_id = None
        self.player_id = None
        self.player_index = None
        self.hand_cards = []
        self.turn_count = 0
        # self.discard_history = []
        self.player_discards = {0: [], 1: [], 2: [], 3: []}
        self.player_rush = {0: [], 1: [], 2: [], 3: []}
        self.last_discard = None
        self.last_discard_player = None
        self.last_discard_player_index = None

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
