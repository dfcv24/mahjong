import torch
import torch.nn as nn
from .config import D_MODEL, NHEAD, NUM_ENCODER_LAYERS
from src.utils.constants import MAX_HAND_SIZE, NUM_TILE_TYPES

class MahjongDiscardModel(nn.Module):
    """
    初始化打牌决策模型
    
    参数:
    input_size: 输入特征维度，默认198
        当前玩家手牌(14) + 当前玩家rush牌(16) + 当前玩家hitout牌(30) +
        其他三个玩家各自的rush牌(16*3) + hitout牌(30*3)
    hidden_size: 隐藏层大小
    output_size: 输出维度，默认37 (34种牌 + 胡牌 + 暗杠 + 明杠)
    """
    def __init__(self, dropout_rate=0.1, input_size=198, output_size=NUM_TILE_TYPES+3):
        super().__init__()
        self.input_size = input_size
        # 输入嵌入
        self.tile_embed = nn.Embedding(NUM_TILE_TYPES+1, D_MODEL, padding_idx=NUM_TILE_TYPES)  # +1用于填充值
        self.turn_embed = nn.Embedding(10, D_MODEL)  # 回合分10个桶
        
        # 位置编码
        self.pos_embed = nn.Embedding(input_size+1, D_MODEL)  # +1用于回合token

        # 特征类型编码 (区分手牌、rush牌、hitout牌)
        self.feature_type_embed = nn.Embedding(4, D_MODEL)  # 0=手牌, 1=rush牌, 2=hitout牌, 3=回合

        # 添加嵌入后的dropout
        self.embed_dropout = nn.Dropout(dropout_rate)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, 
            nhead=NHEAD,
            dim_feedforward=4*D_MODEL,
            batch_first=True,
            dropout=dropout_rate
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, NUM_ENCODER_LAYERS)
        
        # 修改决策头，删除一个隐藏层
        self.action_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(D_MODEL*2, output_size)  # 输出34种牌 + 胡牌 + 暗杠 + 明杠
        )
        
    def forward(self, features, turn, action_mask=None):
        """
        前向传播
        
        参数:
        features: [batch_size, input_size] 包含当前玩家和其他玩家的手牌、rush牌和hitout牌信息
        turn: [batch_size] 当前回合数
        
        返回:
        logits: [batch_size, output_size] 打牌/胡牌/杠牌的概率对数
        """
        batch_size = features.size(0)
        
        # 输入嵌入
        tile_embeddings = self.tile_embed(features)  # (B, input_size, D)
        
        # 回合处理（分桶）
        boundaries = torch.tensor([4,8,12,16,20,24,28,32], device=features.device)
        turn_bucket = torch.bucketize(turn, boundaries)
        turn_embeddings = self.turn_embed(turn_bucket).unsqueeze(1)  # (B, 1, D)
        
        # 拼接回合token
        # combined = torch.cat([tile_embeddings, turn_embeddings], dim=1)  # (B, 15, D)
        
        # 准备位置编码
        positions = torch.arange(self.input_size, device=features.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_embed(positions)  # [batch, input_size, d_model]
        
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
        
        # 组合所有嵌入
        combined = tile_embeddings + pos_embeddings + feature_type_embeddings  # [batch, input_size, d_model]
        
        # 将回合嵌入添加到序列末尾
        turn_pos_embed = self.pos_embed(torch.full((batch_size, 1), self.input_size, 
                                                  device=features.device))  # [batch, 1, d_model]
        turn_type_embed = self.feature_type_embed(torch.full((batch_size, 1), 3, 
                                                            device=features.device))  # [batch, 1, d_model]
        turn_full_embed = turn_embeddings + turn_pos_embed + turn_type_embed  # [batch, 1, d_model]
        
        # 合并特征和回合
        sequence = torch.cat([combined, turn_full_embed], dim=1)  # [batch, input_size+1, d_model]
        sequence = self.embed_dropout(sequence)

        # 生成注意力掩码，处理填充值(NUM_TILE_TYPES)
        # 创建填充掩码，值为NUM_TILE_TYPES的位置为True
        padding_mask = (features == NUM_TILE_TYPES)  # [batch, input_size]
        
        # 为回合token添加False（不掩盖）
        padding_mask = torch.cat([
            padding_mask, 
            torch.zeros(batch_size, 1, dtype=torch.bool, device=features.device)
        ], dim=1)  # [batch, input_size+1]
        
        # 通过Transformer编码器
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)  # [batch, input_size+1, d_model]
        
        # 使用回合位置的输出作为决策基础
        decision_features = encoded[:, -1, :]  # [batch, d_model]
        
        # 输出动作概率
        logits = self.action_head(decision_features)  # [batch, output_size]
        
        if action_mask is not None:
            masked_logits = logits.clone()
            masked_logits[~action_mask] = -1e9  # 将不可选动作的logits设为很小的值
            return masked_logits
        
        return logits

# 添加一个简化的打牌决策模型
class SimpleMahjongDiscardModel(nn.Module):
    """
    简化版麻将打牌决策模型，只使用当前玩家手牌和回合信息
    
    参数:
    input_size: 输入特征维度，默认14（仅包含手牌）
    output_size: 输出维度，默认37 (34种牌 + 胡牌 + 暗杠 + 明杠)
    """
    def __init__(self, dropout_rate=0.1, input_size=14, output_size=NUM_TILE_TYPES+3):
        super(SimpleMahjongDiscardModel, self).__init__()
        
        self.input_size = input_size
        
        # 输入嵌入
        self.tile_embed = nn.Embedding(NUM_TILE_TYPES+1, D_MODEL, padding_idx=NUM_TILE_TYPES)  # +1用于填充值
        self.turn_embed = nn.Embedding(10, D_MODEL)  # 支持50个回合
        
        # 位置编码
        self.pos_embed = nn.Embedding(input_size+1, D_MODEL)  # +1用于回合token
        
        # 添加嵌入后的dropout
        self.embed_dropout = nn.Dropout(dropout_rate)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, 
            nhead=NHEAD,
            dim_feedforward=4*D_MODEL,
            batch_first=True,
            dropout=dropout_rate
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, NUM_ENCODER_LAYERS)
        
        # 决策头
        self.action_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(D_MODEL*2, output_size)  # 输出34种牌 + 胡牌 + 暗杠 + 明杠
        )
    
    def forward(self, features, turn, action_mask=None):
        """
        前向传播
        
        参数:
        features: [batch_size, input_size] 当前玩家的手牌
        turn: [batch_size] 当前回合数
        
        返回:
        logits: [batch_size, output_size] 打牌/胡牌/杠牌的概率对数
        """
        batch_size = features.size(0)
        
        # 嵌入手牌
        tile_embeddings = self.tile_embed(features)  # [batch, input_size, d_model]
        
        # 嵌入回合（修改分桶方式）
        boundaries = torch.tensor([4,8,12,16,20,24,28,32], device=features.device)
        turn_bucket = torch.bucketize(turn, boundaries)
        turn_embeddings = self.turn_embed(turn_bucket)  # [batch, d_model]
        turn_embeddings = turn_embeddings.unsqueeze(1)  # [batch, 1, d_model]
        
        # 准备位置编码
        positions = torch.arange(self.input_size, device=features.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_embed(positions)  # [batch, input_size, d_model]
        
        # 组合所有嵌入
        combined = tile_embeddings + pos_embeddings  # [batch, input_size, d_model]
        
        # 将回合嵌入添加到序列末尾
        turn_pos_embed = self.pos_embed(torch.full((batch_size, 1), self.input_size, 
                                                  device=features.device))  # [batch, 1, d_model]
        turn_full_embed = turn_embeddings + turn_pos_embed  # [batch, 1, d_model]
        
        # 合并手牌和回合
        sequence = torch.cat([combined, turn_full_embed], dim=1)  # [batch, input_size+1, d_model]
        sequence = self.embed_dropout(sequence)
        
        # 生成注意力掩码，处理填充值(NUM_TILE_TYPES)
        # 创建填充掩码，值为NUM_TILE_TYPES的位置为True
        padding_mask = (features == NUM_TILE_TYPES)  # [batch, input_size]
        
        # 为回合token添加False（不掩盖）
        padding_mask = torch.cat([
            padding_mask, 
            torch.zeros(batch_size, 1, dtype=torch.bool, device=features.device)
        ], dim=1)  # [batch, input_size+1]
        
        # 通过Transformer编码器
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)  # [batch, input_size+1, d_model]
        
        # 使用回合位置的输出作为决策基础
        decision_features = encoded[:, -1, :]  # [batch, d_model]
        
        # 输出动作概率
        logits = self.action_head(decision_features)  # [batch, output_size]
        
        if action_mask is not None:
            masked_logits = logits.clone()
            masked_logits[~action_mask] = -1e9  # 将不可选动作的logits设为很小的值
            return masked_logits
        
        return logits
    
# 吃碰杠胡决策模型
class MahjongActionModel(torch.nn.Module):
    """
    麻将吃碰杠胡决策模型
    
    参数:
    input_size: 输入特征维度，默认198
        当前玩家手牌(14) + 当前玩家rush牌(16) + 当前玩家hitout牌(30) +
        其他三个玩家各自的rush牌(16*3) + hitout牌(30*3)
    dropout_rate: dropout比率
    """
    def __init__(self, dropout_rate=0.1, input_size=198, output_size=5):
        super().__init__()
        self.input_size = input_size
        # 输入嵌入
        self.tile_embed = nn.Embedding(NUM_TILE_TYPES+1, D_MODEL, padding_idx=NUM_TILE_TYPES)
        self.rush_tile_embed = nn.Embedding(NUM_TILE_TYPES+1, D_MODEL, padding_idx=NUM_TILE_TYPES)  # rush牌的嵌入
        self.turn_embed = nn.Embedding(10, D_MODEL)
        self.pos_embed = nn.Embedding(input_size+2, D_MODEL)  # +2是rush牌和回合token
        
        # 特征类型编码 (区分手牌、rush牌、hitout牌)
        self.feature_type_embed = nn.Embedding(5, D_MODEL)  # 0=手牌, 1=rush牌, 2=hitout牌, 3=回合, 4=rush_tile
        
        # 添加嵌入后的dropout
        self.embed_dropout = nn.Dropout(dropout_rate)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, 
            nhead=NHEAD,
            dim_feedforward=4*D_MODEL,
            batch_first=True,
            dropout=dropout_rate
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, NUM_ENCODER_LAYERS)
        
        # 动作类型预测头 (5种动作)
        self.action_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(D_MODEL*2, 5)  # 5种动作: 过, 吃, 碰, 杠, 胡
        )
        
        # 吃牌组合预测头 - 预测手牌索引
        # 对于吃牌，我们需要选择手牌中的两张牌
        self.chi_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(D_MODEL*2, 14 * 2)  # 预测两张牌的位置，每个位置范围是0-13
        )
    
    def forward(self, features, rush_tile, turn, action_mask=None):
        """
        前向传播
        
        参数:
        features: [batch_size, input_size] 包含当前玩家和其他玩家的手牌、rush牌和hitout牌信息
        rush_tile: [batch_size] rush牌（别人打出的牌）
        turn: [batch_size] 当前回合数
        action_mask: [batch_size, 5] 动作掩码，指示哪些动作可用 (过,吃,碰,杠,胡)
        
        返回:
        action_logits: [batch_size, 5] 5种动作的概率对数
        chi_logits: [batch_size, 28] 用于吃牌的两张牌位置的预测
        """
        batch_size = features.size(0)
        actual_input_size = features.size(1)
    
        # 检查输入尺寸是否与预期不同，如果不同则打印警告
        if actual_input_size != self.input_size:
            print(f"警告: 输入特征维度 {actual_input_size} 与模型预期维度 {self.input_size} 不匹配")
    
        # 嵌入所有特征
        tile_embeddings = self.tile_embed(features)  # [batch, input_size, d_model]
        rush_tile_embedding = self.rush_tile_embed(rush_tile).unsqueeze(1)  # [batch, 1, d_model]
        
        # 回合处理（分桶）
        boundaries = torch.tensor([4,8,12,16,20,24,28,32], device=features.device)
        turn_bucket = torch.bucketize(turn, boundaries)
        turn_embeddings = self.turn_embed(turn_bucket).unsqueeze(1)  # (B, 1, D)
        
        # 准备位置编码 - 使用actual_input_size而不是self.input_size
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
        
        # 组合所有嵌入
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
        # 创建填充掩码，值为NUM_TILE_TYPES的位置为True
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
        
        # 输出动作类型概率
        action_logits = self.action_head(decision_features)  # [batch, 5]
        
        # 如果提供了动作掩码，应用掩码
        if action_mask is not None:
            # 确保动作掩码是布尔值
            if action_mask.dtype != torch.bool:
                action_mask = action_mask.bool()
            
            # 将不可执行动作的logits设为很小的值
            action_logits = action_logits.clone()
            action_logits[~action_mask] = -1e9
        
        # 输出吃牌组合概率
        chi_logits = self.chi_head(decision_features)  # [batch, 28]
        
        return action_logits, chi_logits
    

# 添加一个简化的吃碰杠胡决策模型
class SimpleMahjongActionModel(nn.Module):
    """
    简化版麻将吃碰杠胡决策模型，只使用当前玩家手牌、单个rush牌和回合信息
    
    参数:
    input_size: 输入特征维度，默认14（14张手牌 ）
    output_size: 输出维度，默认5 (不操作 + 吃 + 碰 + 明杠 + 胡)
    """
    def __init__(self, dropout_rate=0.1, input_size=14, output_size=5):
        super(SimpleMahjongActionModel, self).__init__()
        
        self.input_size = input_size
        
        # 输入嵌入
        self.tile_embed = nn.Embedding(NUM_TILE_TYPES+1, D_MODEL, padding_idx=NUM_TILE_TYPES)  # +1用于填充值
        self.turn_embed = nn.Embedding(10, D_MODEL)  # 回合分10个桶
        self.rush_tile_embed = nn.Embedding(NUM_TILE_TYPES+1, D_MODEL, padding_idx=NUM_TILE_TYPES)  # rush牌的嵌入
        
        # 位置编码
        self.pos_embed = nn.Embedding(input_size+2, D_MODEL)  # +2用于回合token和rush牌token
        
        # 添加嵌入后的dropout
        self.embed_dropout = nn.Dropout(dropout_rate)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, 
            nhead=NHEAD,
            dim_feedforward=4*D_MODEL,
            batch_first=True,
            dropout=dropout_rate
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, NUM_ENCODER_LAYERS)
        
        # 决策头
        self.action_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(D_MODEL*2, output_size)  # 输出：不操作 + 吃 + 碰 + 明杠 + 胡
        )
        
        # 用于吃牌决策的额外输出头
        self.chi_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(D_MODEL, 2 * MAX_HAND_SIZE)  # 预测两张用于吃的牌的位置
        )
    
    def forward(self, features, rush_tile, turn, action_mask=None):
        """
        前向传播
        
        参数:
        features: [batch_size, input_size] 当前玩家的手牌
        rush_tile: [batch_size] 当前的rush牌
        turn: [batch_size] 当前回合数
        action_mask: [batch_size, output_size] 动作掩码，指示哪些动作是合法的
        
        返回:
        logits: [batch_size, output_size] 各操作的概率对数
        chi_logits: [batch_size, MAX_HAND_SIZE] 各手牌位置用于吃的概率对数
        """
        batch_size = features.size(0)
        
        # 嵌入手牌
        tile_embeddings = self.tile_embed(features)  # [batch, input_size, d_model]
        
        # 嵌入rush牌
        rush_embeddings = self.rush_tile_embed(rush_tile).unsqueeze(1)  # [batch, 1, d_model]
        
        # 嵌入回合（分桶）
        boundaries = torch.tensor([4,8,12,16,20,24,28,32], device=features.device)
        turn_bucket = torch.bucketize(turn, boundaries)
        turn_embeddings = self.turn_embed(turn_bucket).unsqueeze(1)  # [batch, 1, d_model]
        
        # 准备位置编码
        positions = torch.arange(self.input_size, device=features.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_embed(positions)  # [batch, input_size, d_model]
        
        # 组合手牌嵌入和位置编码
        combined = tile_embeddings + pos_embeddings  # [batch, input_size, d_model]
        
        # 添加rush牌的位置编码
        rush_pos = torch.full((batch_size, 1), self.input_size, device=features.device)
        rush_pos_embed = self.pos_embed(rush_pos)  # [batch, 1, d_model]
        rush_full_embed = rush_embeddings + rush_pos_embed  # [batch, 1, d_model]
        
        # 添加回合的位置编码
        turn_pos = torch.full((batch_size, 1), self.input_size + 1, device=features.device)
        turn_pos_embed = self.pos_embed(turn_pos)  # [batch, 1, d_model]
        turn_full_embed = turn_embeddings + turn_pos_embed  # [batch, 1, d_model]
        
        # 合并所有特征：手牌、rush牌和回合
        sequence = torch.cat([combined, rush_full_embed, turn_full_embed], dim=1)  # [batch, input_size+1, d_model]
        sequence = self.embed_dropout(sequence)
        
        # 生成注意力掩码，处理填充值(NUM_TILE_TYPES)
        # 创建填充掩码，值为NUM_TILE_TYPES的位置为True
        padding_mask = (features == NUM_TILE_TYPES)  # [batch, input_size-1]
        
        # 为rush牌和回合token添加掩码
        rush_padding = (rush_tile == NUM_TILE_TYPES).unsqueeze(1)  # [batch, 1]
        padding_mask = torch.cat([
            padding_mask, 
            rush_padding,
            torch.zeros(batch_size, 1, dtype=torch.bool, device=features.device)  # 回合token不掩码
        ], dim=1)  # [batch, input_size+1]
        
        # 通过Transformer编码器
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)  # [batch, input_size+1, d_model]
        
        # 使用rush_tile和回合位置的输出作为决策基础
        # 这里使用了一个简单的平均池化来融合这两个token的信息
        decision_features = (encoded[:, -2, :] + encoded[:, -1, :]) / 2  # [batch, d_model]
        # 使用回合位置的输出作为主要决策特征
        # decision_features = encoded[:, -1, :]  # [batch, d_model]
        
        # 输出动作概率
        logits = self.action_head(decision_features)  # [batch, output_size]
        
        # 输出动作类型概率
        action_logits = self.action_head(decision_features)  # [batch, 5]
        
        # 如果提供了动作掩码，应用掩码
        if action_mask is not None:
            # 确保动作掩码是布尔值
            if action_mask.dtype != torch.bool:
                action_mask = action_mask.bool()
            
            # 将不可执行动作的logits设为很小的值
            action_logits = action_logits.clone()
            action_logits[~action_mask] = -1e9
        
        # 输出吃牌组合概率
        chi_logits = self.chi_head(decision_features)  # [batch, 28]
        
        return action_logits, chi_logits