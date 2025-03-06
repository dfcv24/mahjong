"""
模型和训练的配置参数
"""

# 模型超参数
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 6

# 训练超参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

# 早期/晚期回合定义
EARLY_ROUND = 8
LATE_ROUND = 24