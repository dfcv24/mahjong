from torch.utils.data import Dataset
import os
import glob
import torch
import numpy as np
from torch.utils.data import Subset, random_split
from .constants import *
from .tile_utils import chinese_tile_to_id, parse_chinese_hand

class BaseMahjongDataset(Dataset):
    """
    麻将数据集的基类，用于处理数据文件的共同逻辑
    """
    def __init__(self, data_folder="/home/luzhiwei/data/a/mahjong_data", max_samples=None, debug=False):
        """
        初始化基础麻将数据集
        
        参数:
        data_folder: 包含游戏记录的文件夹路径
        max_samples: 最大加载样本数 (None表示加载全部)
        debug: 是否打印调试信息
        """
        self.data_folder = data_folder
        self.debug = debug
        self.data = []
        
        # 检查数据目录
        if not self._check_data_folder():
            return
    
    def _check_data_folder(self):
        """检查数据文件夹是否存在且有效"""
        if not os.path.exists(self.data_folder):
            print(f"错误: 数据文件夹 '{self.data_folder}' 不存在")
            return False
        
        if not os.path.isdir(self.data_folder):
            print(f"错误: '{self.data_folder}' 不是文件夹")
            return False
        
        try:
            self.game_folders = [d for d in os.listdir(self.data_folder) 
                                if os.path.isdir(os.path.join(self.data_folder, d))]
            if not self.game_folders:
                print(f"警告: 在 '{self.data_folder}' 中未找到子文件夹")
                return False
        except OSError as e:
            print(f"访问数据文件夹出错: {e}")
            return False
            
        return True
    
    def _parse_hand(self, hand_line):
        """解析手牌行
        解析手牌行,将中文转为数字ID,将null转为NUM_TILE_TYPES
        """
        if not hand_line or hand_line.strip() == "":
            return []
    
        parts = hand_line.split()
        hand = []
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if part.lower() == "null":
                # null转换为NUM_TILE_TYPES
                hand.append(NUM_TILE_TYPES)
            else:
                tile_id = chinese_tile_to_id(part)
                if tile_id >= 0:  # 有效的中文牌名
                    hand.append(tile_id)
                else:
                    # 无法解析的部分用填充值替代
                    if self.debug:
                        print(f"警告: 无法解析牌 '{part}'，使用填充值替代")
                    hand.append(NUM_TILE_TYPES)
        return hand
    
    def _pad_sequence(self, seq, max_len, pad_value=NUM_TILE_TYPES):
        """将序列填充或截断到指定长度"""
        if len(seq) > max_len:
            return seq[:max_len]
        elif len(seq) < max_len:
            return seq + [pad_value] * (max_len - len(seq))
        else:
            return seq
    
    def _process_player_data(self, lines):
        """处理玩家数据，返回包含所有玩家手牌、rush牌和hitout牌的字典"""
        try:
            player_data = {}
            # 只读第一个玩家的手牌，
            hand_idx = 1
            hand_line = lines[hand_idx].strip()
            # 跳过第一个标识符(HandPaiList等)，只取牌列表部分
            hand_parts = hand_line.split(maxsplit=1)
            if len(hand_parts) > 1:
                hand = self._parse_hand(hand_parts[1])
            else:
                hand = []
            hand = self._pad_sequence(hand, MAX_HAND_SIZE, NUM_TILE_TYPES)
            player_data[f'player{0}_hand'] = hand
            # 处理4个玩家的数据
            for i in range(4):
                # 计算每个玩家数据的起始行索引
                # 玩家0: 第2-5行 (索引1-4)
                # 玩家1: 第6-9行 (索引5-8) 
                # 玩家2: 第10-13行 (索引9-12)
                # 玩家3: 第14-17行 (索引13-16)
                player_start_idx = i * 4 + 1  # 第一行是元数据，从索引1开始
                
                # rush牌 (该玩家数据的第二行)
                rush_idx = player_start_idx + 1
                rush_line = lines[rush_idx].strip()
                # 跳过第一个标识符(RushPaiList等)，只取牌列表部分
                rush_parts = rush_line.split(maxsplit=1)
                if len(rush_parts) > 1:
                    rush = self._parse_hand(rush_parts[1])
                else:
                    rush = []
                rush = self._pad_sequence(rush, MAX_RUSH_SIZE, NUM_TILE_TYPES)
                player_data[f'player{i}_rush'] = rush
                
                # hitout牌 (该玩家数据的第三行)
                hitout_idx = player_start_idx + 2
                hitout_line = lines[hitout_idx].strip()
                # 跳过第一个标识符(HitoutPaiList等)，只取牌列表部分
                hitout_parts = hitout_line.split(maxsplit=1)
                if len(hitout_parts) > 1:
                    hitout = self._parse_hand(hitout_parts[1])
                else:
                    hitout = []
                hitout = self._pad_sequence(hitout, MAX_DISCARD_SIZE, NUM_TILE_TYPES)
                player_data[f'player{i}_discard'] = hitout
                
                # 胡牌信息在第四行，但我们不需要处理它
            
            return player_data
                
        except Exception as e:
            if self.debug:
                print(f"处理玩家数据时出错: {e}")
            return None

    def __len__(self):
        return len(self.data)

class MahjongDiscardDataset(BaseMahjongDataset):
    """
    麻将打牌决策数据集
    文件格式:
    - 第一行: 牌局ID 回合序号 玩家名 剩余牌 是否能胡 是否能暗杠 是否能明杠
    - 第2-5行: 玩家0的手牌、rush牌、hitout牌、胡牌
    - 第6-9行: 玩家1的手牌、rush牌、hitout牌、胡牌
    - 第10-13行: 玩家2的手牌、rush牌、hitout牌、胡牌
    - 第14-17行: 4位玩家的强制动作
    - 最后一行: action hitout/zimohu/ZimoAnGang/ZimoMingGang 打出的牌(中文)
    """
    
    def __init__(self, data_folder="/home/luzhiwei/data/a/mahjong_data", max_samples=None, debug=False):
        """
        初始化打牌决策数据集
        新的输入格式：
        - 当前玩家手牌14 + 当前玩家rush牌16 + 当前玩家hitout牌30 +
        - 玩家1rush牌16 + 玩家1hitout牌30 +
        - 玩家2rush牌16 + 玩家2hitout牌30 +
        - 玩家3rush牌16 + 玩家3hitout牌30 +
        - 剩余牌
        
        输出：
        - 34张手牌 + 胡牌 + 暗杠 + 明杠
        """
        print(f"初始化麻将打牌决策数据集，加载目录: {data_folder}")
        super().__init__(data_folder, max_samples, debug)
        self._load_data(max_samples)
        print(f"加载完成，共 {len(self.data)} 个样本")
    
    def _load_data(self, max_samples):
        """加载打牌决策数据"""
        sample_count = 0
        
        for game_folder in self.game_folders:
            game_path = os.path.join(self.data_folder, game_folder)
            step_files = sorted(glob.glob(os.path.join(game_path, "*.txt")))
            for step_file in step_files:
                try:
                    # 如果文件名带a，则跳过（这是吃碰杠胡动作数据）
                    if '_a.txt' in step_file:
                        continue
                        
                    # 读取文件
                    with open(step_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # 确保文件有足够的行
                    if len(lines) < 18:
                        continue
                    
                    # 解析第一行：牌局ID 回合序号 玩家名 剩余牌 是否能胡 是否能暗杠 是否能明杠
                    first_line = lines[0].strip().split()
                    if len(first_line) < 7:
                        # 数据格式不符合要求，跳过
                        continue
                    # 解析基本信息
                    game_id = first_line[0]                # 牌局ID
                    turn = int(first_line[1])              # 回合序号
                    remaining_tiles = int(first_line[3])   # 剩余牌数
                    
                    # 解析特殊动作可用性
                    can_hu = first_line[4].lower() in ('true', '1', 't', 'yes', 'y')
                    can_angang = first_line[5].lower() in ('true', '1', 't', 'yes', 'y')
                    can_minggang = first_line[6].lower() in ('true', '1', 't', 'yes', 'y')
                    
                    # 解析最后一行，获取目标动作
                    last_line = lines[-1].strip().lower()
                    if "zimohu" in last_line:
                        target_action = NUM_TILE_TYPES  # 胡牌
                    elif "zimoangang" in last_line:
                        target_action = NUM_TILE_TYPES + 1  # 暗杠
                    elif "zimominggang" in last_line:
                        target_action = NUM_TILE_TYPES + 2  # 明杠
                    else:
                        # 处理普通打牌
                        parts = last_line.split()
                        if len(parts) < 3 or parts[0] != "action" or parts[1] != "hitout":
                            continue
                        
                        # 根据中文牌名获取牌ID
                        card_name = parts[2]
                        target_action = chinese_tile_to_id(card_name)
                    
                    # 处理玩家数据
                    player_data = self._process_player_data(lines)
                    if player_data is None:
                        continue
                    
                    # 创建样本
                    sample = {
                        'turn': turn,
                        'target': target_action,
                        'game_id': game_id,
                        'can_hu': can_hu,
                        'can_angang': can_angang,
                        'can_minggang': can_minggang,
                        'remaining_tiles': remaining_tiles
                    }
                    
                    # 添加玩家数据
                    sample.update(player_data)
                    
                    # 添加到数据集
                    self.data.append(sample)
                    sample_count += 1
                    
                    # 如果达到最大样本数，则停止加载
                    if max_samples is not None and sample_count >= max_samples:
                        return
                    
                except Exception as e:
                    if self.debug:
                        print(f"处理文件 {step_file} 时出错: {e}")
                    continue
    
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        sample = self.data[idx]
        
        # 构建输入特征
        features = []
        
        # 当前玩家数据
        features.extend(sample['player0_hand'])
        features.extend(sample['player0_rush'])
        features.extend(sample['player0_discard'])
        
        # 其他玩家数据
        for i in range(1, 4):
            features.extend(sample[f'player{i}_rush'])
            features.extend(sample[f'player{i}_discard'])
        
        # 创建输入张量
        features_tensor = torch.tensor(features, dtype=torch.long)
        
        # 创建目标张量
        target_tensor = torch.tensor(sample['target'], dtype=torch.long)
        
        # 创建回合张量
        turn_tensor = torch.tensor(sample['turn'], dtype=torch.long)
        
        # 创建动作掩码张量
        action_mask = torch.zeros(NUM_TILE_TYPES + 3, dtype=torch.bool)
        
        # 对手牌中的每张牌设置掩码为True
        for tile_id in sample['player0_hand']:
            if tile_id < NUM_TILE_TYPES:  # 确保是有效的牌ID
                action_mask[tile_id] = True
        
        # 设置特殊动作掩码
        action_mask[NUM_TILE_TYPES] = sample['can_hu']
        action_mask[NUM_TILE_TYPES + 1] = sample['can_angang']
        action_mask[NUM_TILE_TYPES + 2] = sample['can_minggang']
        
        return {
            'features': features_tensor,
            'target': target_tensor,
            'turn': turn_tensor,
            'action_mask': action_mask,
            'game_id': sample['game_id']
        }

class MahjongActionDataset(BaseMahjongDataset):
    """
    用于加载麻将吃碰杠胡请求决策数据的数据集类
    
    文件格式:
    - 第一行: 牌局ID 回合序号 玩家名 剩余牌 rush牌 可吃 可碰 可杠 可胡
    - 第二行: 玩家名 剩余手牌列表(中文)
    - 第三行: 玩家名 吃碰杠出去(Rush)的牌(中文)
    - 第四行: 玩家名 打出的牌(中文)
    - 第五行: 玩家名 可胡的牌(中文)
    - 其他三个玩家信息：第六-九，十-十三行， 十四-十七行: 剩余手牌列表(中文) 吃碰杠出去(Rush)的牌(中文) 打出的牌(中文) 可胡的牌(中文)
    - 最后一行: action 动作 (过/吃/碰/杠/胡), 吃牌时还有两张额外的牌
    """
    def __init__(self, data_folder="/home/luzhiwei/data/a/mahjong_data", max_samples=None, debug=False):
        print(f"初始化麻将吃碰杠胡请求数据集，加载目录: {data_folder}")
        super().__init__(data_folder, max_samples, debug)
        self._load_data(max_samples)
        print(f"加载完成，共 {len(self.data)} 个样本")
    
    def _load_data(self, max_samples):
        """
        加载吃碰杠胡决策数据
        从原MahjongActionDataset中移植过来的加载逻辑
        """
        sample_count = 0
        
        for game_folder in self.game_folders:
            game_path = os.path.join(self.data_folder, game_folder)
            # 只处理带a后缀的文件
            action_files = sorted(glob.glob(os.path.join(game_path, "*_a.txt")))
            
            for action_file in action_files:
                try:
                    # 读取文件
                    with open(action_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    if len(lines) < 18:  # 至少需要18行: 第一行信息, 四个玩家信息(每个4行), 最后一行动作
                        if self.debug:
                            print(f"警告: 文件行数不足 - {action_file}")
                        continue
                    
                    # 解析第一行 - 获取回合信息和Rush牌
                    first_line = lines[0].strip().split()
                    if len(first_line) < 9:  # 至少需要9个元素
                        if self.debug:
                            print(f"警告: 首行格式不正确 - {action_file}")
                        continue
                    
                    try:
                        game_id = first_line[0]  # 牌局ID
                        turn = int(first_line[1])  # 回合序号
                        player_name = first_line[2]  # 玩家名
                        remaining_tiles = int(first_line[3])  # 剩余牌数
                        
                        # 解析rush牌 (别人打出的牌)
                        rush_tile_str = first_line[4]
                        rush_tile = chinese_tile_to_id(rush_tile_str)
                        
                        if rush_tile < 0:
                            if self.debug:
                                print(f"警告: 无法解析Rush牌 '{rush_tile_str}' in {action_file}")
                            continue
                            
                        # 解析服务端发来的可能动作布尔值
                        can_chi = first_line[5].lower() == "true"
                        can_peng = first_line[6].lower() == "true"
                        can_gang = first_line[7].lower() == "true"
                        can_hu = first_line[8].lower() == "true"
                        
                        # 可执行动作的掩码 (5个动作: 过, 吃, 碰, 杠, 胡)
                        # 过/skip总是可选的
                        action_mask = [True, can_chi, can_peng, can_gang, can_hu]
                        
                    except (ValueError, IndexError) as e:
                        if self.debug:
                            print(f"解析第一行出错: {action_file} - {e}")
                        continue

                    # 处理玩家数据
                    player_data = self._process_player_data(lines)
                    if player_data is None:
                        continue

                    # 解析最后一行的动作
                    last_line = lines[-1].strip()
                    action_tokens = last_line.split()
                    
                    if len(action_tokens) < 2:
                        continue  # 跳过不完整数据
                    
                    action_type = action_tokens[1].lower()
                    
                    # 解析动作类型
                    if action_type in ["过", "skip", "pass"]:
                        action = ACTION_SKIP
                        chi_tiles = []
                    elif action_type in ["吃", "chi"]:
                        action = ACTION_CHI
                        # 吃牌时还有额外的两张牌
                        if len(action_tokens) >= 4:
                            try:
                                chi_tiles = []
                                for i in range(2, 4):
                                    tile_id = chinese_tile_to_id(action_tokens[i])
                                    if tile_id is not None:
                                        chi_tiles.append(tile_id)
                                    else:
                                        if self.debug:
                                            print(f"警告: 无法解析吃牌 '{action_tokens[i]}' in {action_file}")
                            except Exception:
                                chi_tiles = []
                        else:
                            chi_tiles = []
                    elif action_type in ["碰", "peng"]:
                        action = ACTION_PENG
                        chi_tiles = []
                    elif action_type in ["杠", "gang"]:
                        action = ACTION_GANG
                        chi_tiles = []
                    elif action_type in ["胡", "hu"]:
                        action = ACTION_HU
                        chi_tiles = []
                    else:
                        if self.debug:
                            print(f"未知动作类型: {action_type} 在 {action_file}")
                        continue
                    
                    # 检查动作是否在可执行动作范围内
                    if not action_mask[action]:
                        if self.debug:
                            print(f"警告: 动作 {action_type} 不在服务器允许的动作列表中: {action_file}")
                        continue
                    
                    # 添加到样本列表
                    # 创建样本
                    sample = {
                        'turn': turn,
                        'game_id': game_id,
                        'rush_tile': rush_tile,
                        'action_mask': action_mask,
                        'action': action,
                        'chi_tiles': chi_tiles
                    }
                    
                    # 添加玩家数据
                    sample.update(player_data)
                    self.data.append(sample)
                    
                    sample_count += 1
                    if max_samples is not None and sample_count >= max_samples:
                        return
                        
                except Exception as e:
                    if self.debug:
                        print(f"Error processing file {action_file}: {e}")
                    continue
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 构建输入特征
        features = []
        
        # 当前玩家数据
        features.extend(sample['player0_hand'])
        features.extend(sample['player0_rush'])
        features.extend(sample['player0_discard'])
        
        # 其他玩家数据
        for i in range(1, 4):
            features.extend(sample[f'player{i}_rush'])
            features.extend(sample[f'player{i}_discard'])
        
        # 创建输入张量
        features_tensor = torch.tensor(features, dtype=torch.long)
        
        # Rush牌信息
        rush_tile = torch.tensor(sample['rush_tile'], dtype=torch.long)
        
        # 回合数
        turn_tensor = torch.tensor(sample['turn'], dtype=torch.long)
        
        # 动作掩码
        action_mask = torch.tensor(sample['action_mask'], dtype=torch.bool)
        
        # 目标动作
        action = torch.tensor(sample['action'], dtype=torch.long)
        
        # 吃牌类型 (0=前吃, 1=中吃, 2=后吃)
        if sample['action'] == ACTION_CHI and sample['chi_tiles']:
            # 获取rush牌和chi牌的序号
            rush_num = sample['rush_tile']
            chi_nums = sample['chi_tiles']
            
            # 根据序号关系确定吃牌类型
            if all(n < rush_num for n in chi_nums):  # 所有吃牌序号都小于rush牌
                chi_type = 0  # 前吃
            elif all(n > rush_num for n in chi_nums):  # 所有吃牌序号都大于rush牌
                chi_type = 2  # 后吃
            else:  # rush牌在中间
                chi_type = 1  # 中吃
            
            chi_tensor = torch.tensor(chi_type, dtype=torch.long)
        else:
            chi_tensor = torch.tensor(-1, dtype=torch.long)  # 用-1表示非吃牌样本
        
        # chi_type的mask - 检查手牌中是否包含吃牌所需的两张牌
        chi_mask = torch.zeros(3, dtype=torch.bool)  # 对应前吃、中吃、后吃三种类型
        
        if sample['action'] == ACTION_CHI:
            hand_nums = sample['player0_hand']
            rush_num = sample['rush_tile']
            
            # 检查前吃所需的牌是否在手牌中 (需要rush_num-2和rush_num-1)
            if rush_num < 27 and rush_num % 9 >= 2:
                chi_mask[0] = ((rush_num-2) in hand_nums) and ((rush_num-1) in hand_nums)
            
            # 检查中吃所需的牌是否在手牌中 (需要rush_num-1和rush_num+1)
            if rush_num < 27 and rush_num % 9 <= 8 and rush_num % 9 >=1:  # 确保不跨花色
                chi_mask[1] = ((rush_num-1) in hand_nums) and ((rush_num+1) in hand_nums)
            
            # 检查后吃所需的牌是否在手牌中 (需要rush_num+1和rush_num+2)
            if rush_num < 27 and rush_num % 9 <= 7:  # 确保不跨花色
                chi_mask[2] = ((rush_num+1) in hand_nums) and ((rush_num+2) in hand_nums)
             
        return {
            'features': features_tensor,
            'rush_tile': rush_tile,
            'turn': turn_tensor,
            'action_mask': action_mask,
            'action': action,
            'chi_type': chi_tensor,  # 新的吃牌类型标签
            'chi_mask': chi_mask,
            'game_id': sample['game_id']
        }
    
def split_dataset(dataset, train_ratio=0.8, val_ratio=0.2, seed=None):
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    
    # 获取数据集大小
    dataset_size = len(dataset)
    
    # 计算各部分大小
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size
    
    # 使用PyTorch的random_split函数分割数据集
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    print(f"数据集已分割: 训练集 {len(train_dataset)}样本, "
          f"验证集 {len(val_dataset)}样本")
    
    return train_dataset, val_dataset
