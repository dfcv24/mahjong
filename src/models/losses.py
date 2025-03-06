import torch
import torch.nn as nn
from .config import EARLY_ROUND, LATE_ROUND
from src.utils.constants import NUM_TILE_TYPES

class FuzzyLabelLoss(nn.Module):
    """更简单、更稳定的模糊标签损失"""
    def __init__(self, num_classes=NUM_TILE_TYPES+1, early_round=EARLY_ROUND, late_round=LATE_ROUND):
        super().__init__()
        self.num_classes = num_classes
        self.early_round = early_round
        self.late_round = late_round
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, targets, turns):
        batch_size = logits.size(0)
        device = logits.device
        losses = self.ce_loss(logits, targets)
        
        # 根据回合数调整损失权重
        weights = torch.ones_like(losses)
        
        for i in range(batch_size):
            turn = turns[i].item()
            # 早期回合损失权重降低
            if turn <= self.early_round:
                weights[i] = 0.7  # 早期回合的权重
            elif turn >= self.late_round:
                weights[i] = 1.0  # 后期回合的权重
            else:
                # 中间回合线性过渡
                weights[i] = 0.7 + 0.3 * (turn - self.early_round) / (self.late_round - self.early_round)
        
        # 应用权重
        weighted_loss = losses * weights
        return weighted_loss.mean()