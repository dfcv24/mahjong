import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse

from src.models.config import BATCH_SIZE
from src.models.model import MahjongActionModel, SimpleMahjongActionModel
from src.utils.constants import *
from src.utils.evaluation import evaluate_action_model, evaluate_simple_action_model
from src.utils.train_utils import create_results_dir
from src.utils.data_loader import MahjongActionDataset, split_dataset
from src.utils.train_utils import setup_chinese_font

def train_simple_action_model(num_epochs=30, lr=1e-4, results_dir=None):
    """
    使用简化数据(只有手牌、rush牌和回合信息)训练基础动作模型
    
    参数:
    num_epochs: 训练轮数
    lr: 学习率
    results_dir: 结果保存目录
    
    返回:
    model: 训练好的模型
    results_dir: 结果保存目录
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建结果目录
    results_dir = create_results_dir("simple_action_models")
    print(f"训练结果将保存在: {results_dir}")
    
    # 创建数据集
    full_dataset = MahjongActionDataset()
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8)
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"划分数据集完成: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
    
    # 保存训练配置
    config = {
        "batch_size": BATCH_SIZE,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "train_samples": train_size,
        "val_samples": val_size,
        "device": str(device),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(results_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # 创建模型
    print("初始化吃碰杠胡决策模型...")
    model = SimpleMahjongActionModel(
        dropout_rate=0.1,
        input_size=14,  
    ).to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总数 {total_params/1e6:.2f}M, 可训练 {trainable_params/1e6:.2f}M")
    
    # 损失函数和优化器
    action_criterion = nn.CrossEntropyLoss()
    chi_criterion = nn.CrossEntropyLoss(ignore_index=-1) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )
    
    # 训练记录
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_action_accuracy': [],
        'val_chi_accuracy': []
    }
    
    # 最佳模型跟踪
    best_accuracy = 0.0
    best_model_path = os.path.join(results_dir, "mahjong_action_best.pth")
    
    print(f"开始训练，共 {num_epochs} 个周期...")
    # 在开始训练循环前添加
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用同步CUDA操作，便于定位错误
    try:
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for i, batch in enumerate(train_loader):
                # 获取输入和目标
                features = batch['features'][:, :14].to(device)
                rush_tile = batch['rush_tile'].to(device)
                turn = batch['turn'].to(device)
                action_targets = batch['action'].to(device)
                action_masks = batch['action_mask'].to(device)
                
                # 创建吃牌目标
                chi_targets = batch['chi_indices'].to(device)  # [batch_size, 2]
                
                # 前向传播
                action_logits, chi_logits = model(features, rush_tile, turn, action_masks)
                
                # 计算动作损失
                action_loss = action_criterion(action_logits, action_targets)
                
                # 仅对需要吃牌的样本计算吃牌损失
                chi_mask = (action_targets == ACTION_CHI)
                chi_loss = 0.0
                
                if chi_mask.sum() > 0:
                    # 从chi_logits中选择需要吃的样本
                    selected_chi_logits = chi_logits[chi_mask]
                    selected_chi_targets = chi_targets[chi_mask]
                    
                    # 计算两张牌的损失（分别针对第一张和第二张牌的位置）
                    chi_loss_1 = chi_criterion(
                        selected_chi_logits[:, :14], 
                        selected_chi_targets[:, 0]
                    )
                    chi_loss_2 = chi_criterion(
                        selected_chi_logits[:, 14:], 
                        selected_chi_targets[:, 1]
                    )
                    chi_loss = chi_loss_1 + chi_loss_2
                
                # 总损失
                loss = action_loss + chi_loss
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 打印进度
                if (i + 1) % 50 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Step [{i+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}, "
                          f"Action Loss: {action_loss.item():.4f}, "
                          f"Chi Loss: {chi_loss:.4f}")
            
            # 计算平均训练损失
            train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            val_loss, action_accuracy, chi_accuracy = evaluate_simple_action_model(
                model, val_loader, action_criterion, chi_criterion, device)
            
            # 更新学习率
            scheduler.step(action_accuracy)
            
            # 记录训练历史
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_action_accuracy'].append(action_accuracy)
            history['val_chi_accuracy'].append(chi_accuracy)
            
            # 打印本轮结果
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Action Accuracy: {action_accuracy:.4f}, "
                  f"Chi Accuracy: {chi_accuracy:.4f}")
            
            # 保存最佳模型 (基于动作准确率)
            if action_accuracy > best_accuracy:
                best_accuracy = action_accuracy
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'action_accuracy': action_accuracy,
                    'chi_accuracy': chi_accuracy,
                    'loss': val_loss,
                }
                torch.save(checkpoint, best_model_path)
                print(f"保存新的最佳模型，准确率: {action_accuracy:.4f}")
        
        # 绘制训练历史图
        setup_chinese_font()
        plot_training_history(history, results_dir)
        
        # 保存最终模型
        final_model_path = os.path.join(results_dir, "mahjong_action_final.pth")
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'action_accuracy': action_accuracy,
            'chi_accuracy': chi_accuracy,
            'loss': val_loss,
        }, final_model_path)
        print(f"保存最终模型")
        
        print(f"\n训练完成! 最佳验证动作准确率: {best_accuracy:.4f}")
        print(f"结果保存在: {results_dir}")
        
        return model, history, results_dir
    
    except KeyboardInterrupt:
        print("\n训练被中断")
        
        # 保存当前模型
        interrupted_model_path = os.path.join(results_dir, "mahjong_action_interrupted.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, interrupted_model_path)
        print(f"已保存中断时的模型: {interrupted_model_path}")
        
        # 绘制训练历史图
        setup_chinese_font()
        plot_training_history(history, results_dir)
        
        return model, history, results_dir

def fine_tune_action_model(base_model_path, num_epochs=10, lr=5e-5, results_dir=None):
    """
    基于简化模型，使用完整数据进行微调
    
    参数:
    base_model_path: 预训练的简化模型路径
    num_epochs: 微调轮数
    lr: 学习率
    results_dir: 结果保存目录
    
    返回:
    model: 微调后的模型
    results_dir: 结果保存目录
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建结果目录
    results_dir = create_results_dir("finetuned_action_models")
    print(f"训练结果将保存在: {results_dir}")
    
    # 加载数据集
    full_dataset = MahjongActionDataset()
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8)
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"划分数据集完成: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False)
    
    # 加载预训练的简化模型
    print(f"加载预训练模型: {base_model_path}")
    simple_checkpoint = torch.load(base_model_path, map_location=device)
    
    # 创建完整模型
    print("创建完整吃碰杠胡决策模型进行微调...")
    model = MahjongActionModel(
        dropout_rate=0.1,
        input_size=198  # 完整输入大小
    ).to(device)
    
    # 保存训练配置
    config = {
        "model_type": "finetuned_action",
        "base_model": base_model_path,
        "batch_size": BATCH_SIZE,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "train_samples": train_size,
        "val_samples": val_size,
        "device": str(device),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(results_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # 从简化模型加载权重到完整模型
    print("将简化模型的权重转移到完整模型...")
    
    # 加载简化模型
    simple_model = SimpleMahjongActionModel(
        input_size=14 + 1,  # 仅使用当前手牌(14)和rush牌(1)
        output_size=NUM_ACTIONS
    ).to(device)
    
    # 加载权重
    simple_model.load_state_dict(simple_checkpoint['model_state_dict'])
    
    # 复制嵌入层权重
    model.tile_embed.weight.data[:NUM_TILE_TYPES+1] = simple_model.tile_embed.weight.data[:NUM_TILE_TYPES+1]
    model.turn_embed.weight.data = simple_model.turn_embed.weight.data
    
    # 复制Transformer编码器的部分层
    try:
        # 假设两个模型都使用nn.TransformerEncoder，可以按层复制
        for i in range(min(2, len(model.transformer_encoder.layers))):
            if i < len(simple_model.transformer_encoder.layers):
                # 复制自注意力权重
                model.transformer_encoder.layers[i].self_attn.in_proj_weight.data = \
                    simple_model.transformer_encoder.layers[i].self_attn.in_proj_weight.data
                model.transformer_encoder.layers[i].self_attn.in_proj_bias.data = \
                    simple_model.transformer_encoder.layers[i].self_attn.in_proj_bias.data
                model.transformer_encoder.layers[i].self_attn.out_proj.weight.data = \
                    simple_model.transformer_encoder.layers[i].self_attn.out_proj.weight.data
                model.transformer_encoder.layers[i].self_attn.out_proj.bias.data = \
                    simple_model.transformer_encoder.layers[i].self_attn.out_proj.bias.data
                
                # 复制前馈网络权重
                model.transformer_encoder.layers[i].linear1.weight.data = \
                    simple_model.transformer_encoder.layers[i].linear1.weight.data
                model.transformer_encoder.layers[i].linear1.bias.data = \
                    simple_model.transformer_encoder.layers[i].linear1.bias.data
                model.transformer_encoder.layers[i].linear2.weight.data = \
                    simple_model.transformer_encoder.layers[i].linear2.weight.data
                model.transformer_encoder.layers[i].linear2.bias.data = \
                    simple_model.transformer_encoder.layers[i].linear2.bias.data
                
                # 复制层归一化权重
                model.transformer_encoder.layers[i].norm1.weight.data = \
                    simple_model.transformer_encoder.layers[i].norm1.weight.data
                model.transformer_encoder.layers[i].norm1.bias.data = \
                    simple_model.transformer_encoder.layers[i].norm1.bias.data
                model.transformer_encoder.layers[i].norm2.weight.data = \
                    simple_model.transformer_encoder.layers[i].norm2.weight.data
                model.transformer_encoder.layers[i].norm2.bias.data = \
                    simple_model.transformer_encoder.layers[i].norm2.bias.data
    except Exception as e:
        print(f"警告: 无法复制Transformer层权重: {e}")
        print("继续使用随机初始化的Transformer层...")
    
    # 复制决策头部分权重
    try:
        # 复制共享部分
        if hasattr(model.action_head[0], 'weight') and hasattr(simple_model.action_head[0], 'weight'):
            if model.action_head[0].weight.shape[1] == simple_model.action_head[0].weight.shape[1]:
                model.action_head[0].weight.data = simple_model.action_head[0].weight.data
                model.action_head[0].bias.data = simple_model.action_head[0].bias.data
            
        # 复制输出层
        if hasattr(model.action_head[-1], 'weight') and hasattr(simple_model.action_head[-1], 'weight'):
            if model.action_head[-1].weight.shape == simple_model.action_head[-1].weight.shape:
                model.action_head[-1].weight.data = simple_model.action_head[-1].weight.data
                model.action_head[-1].bias.data = simple_model.action_head[-1].bias.data
    except Exception as e:
        print(f"警告: 无法复制决策头权重: {e}")
    
    print("权重转移完成")
    
    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调整器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # 训练记录
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_top3_accuracy': []
    }
    
    # 最佳模型跟踪
    best_accuracy = 0.0
    best_model_path = os.path.join(results_dir, "mahjong_action_best.pth")
    
    print(f"开始微调动作模型，共 {num_epochs} 个周期...")
    
    try:
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i, batch in enumerate(train_loader):
                # 获取完整特征
                features = batch['features'].to(device)
                rush_tile = batch['rush_tile'].to(device)
                turn = batch['turn'].to(device)
                action = batch['action'].to(device)
                
                # 获取动作掩码
                if 'action_mask' in batch:
                    action_mask = batch['action_mask'].to(device)
                else:
                    action_mask = torch.ones(features.size(0), NUM_ACTIONS, dtype=torch.bool, device=device)
                
                # 前向传播
                outputs = model(features, rush_tile, turn, action_mask)
                loss = criterion(outputs, action)
                
                # 计算训练准确率
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == action).sum().item()
                train_total += action.size(0)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 打印进度
                if (i + 1) % 50 == 0:
                    batch_accuracy = (preds == action).sum().item() / action.size(0)
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Step [{i+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}, "
                          f"Accuracy: {batch_accuracy:.4f}")
            
            # 计算平均训练损失和准确率
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_accuracy = train_correct / train_total if train_total > 0 else 0
            
            # 记录训练历史
            history['train_loss'].append(epoch_train_loss)
            history['train_accuracy'].append(epoch_train_accuracy)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            correct = 0
            top3_correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device)
                    rush_tile = batch['rush_tile'].to(device)
                    turn = batch['turn'].to(device)
                    action = batch['action'].to(device)
                    
                    # 获取动作掩码
                    if 'action_mask' in batch:
                        action_mask = batch['action_mask'].to(device)
                    else:
                        action_mask = torch.ones(features.size(0), NUM_ACTIONS, dtype=torch.bool, device=device)
                    
                    outputs = model(features, rush_tile, turn, action_mask)
                    loss = criterion(outputs, action)
                    val_loss += loss.item()
                    
                    # 计算准确率
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == action).sum().item()
                    
                    # 计算Top-3准确率
                    _, top3_preds = torch.topk(outputs, 3, dim=1)
                    for j in range(len(action)):
                        if action[j] in top3_preds[j]:
                            top3_correct += 1
                    
                    total += action.size(0)
            
            # 计算平均验证损失和准确率
            epoch_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
            val_accuracy = correct / total if total > 0 else 0
            top3_accuracy = top3_correct / total if total > 0 else 0
            
            # 记录验证历史
            history['val_loss'].append(epoch_val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_top3_accuracy'].append(top3_accuracy)
            
            # 更新学习率
            scheduler.step(val_accuracy)
            
            # 打印本轮结果
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {epoch_train_loss:.4f}, "
                  f"Train Accuracy: {epoch_train_accuracy:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, "
                  f"Accuracy: {val_accuracy:.4f}, "
                  f"Top-3 Accuracy: {top3_accuracy:.4f}")
            
            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_accuracy,
                    'loss': epoch_val_loss,
                }
                torch.save(checkpoint, best_model_path)
                print(f"保存新的最佳模型，准确率: {val_accuracy:.4f}")
        
        # 绘制训练历史图
        setup_chinese_font()
        plot_training_history(history, results_dir)
        
        # 保存最终模型
        final_model_path = os.path.join(results_dir, "mahjong_action_final.pth")
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': val_accuracy,
            'loss': epoch_val_loss,
        }, final_model_path)
        print(f"保存最终模型")
        
        # 测试最佳模型
        print("\n测试最佳模型...")
        load_and_test_action_model(best_model_path, val_dataset, results_dir)
        
        print(f"\n微调完成! 最佳验证准确率: {best_accuracy:.4f}")
        print(f"结果保存在: {results_dir}")
        
        return model, results_dir
        
    except KeyboardInterrupt:
        print("\n训练被中断")
        
        # 保存当前模型
        interrupted_model_path = os.path.join(results_dir, "mahjong_action_interrupted.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, interrupted_model_path)
        print(f"已保存中断时的模型: {interrupted_model_path}")
        
        # 绘制训练历史图
        if any(len(v) > 0 for v in history.values()):
            setup_chinese_font()
            plot_training_history(history, results_dir)
        
        return model, results_dir

# 训练吃碰杠胡模型
def train_full_action_model(num_epochs=20, lr=1e-4, results_dir=None):
    """
    训练吃碰杠胡请求模型
    
    参数:
    data_folder: 数据文件夹路径
    batch_size: 批次大小
    num_epochs: 训练轮数
    lr: 学习率
    results_dir: 结果保存目录，如果为None则自动创建
    
    返回:
    model: 训练好的模型
    history: 训练历史记录
    results_dir: 结果保存目录
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建结果目录
    results_dir = create_results_dir("action_models")
    print(f"训练结果将保存在: {results_dir}")
    
    # 创建数据集
    full_dataset = MahjongActionDataset()
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8)
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"划分数据集完成: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
    
    # 保存训练配置
    config = {
        "batch_size": BATCH_SIZE,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "train_samples": train_size,
        "val_samples": val_size,
        "device": str(device),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(results_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # 创建模型
    print("初始化吃碰杠胡决策模型...")
    model = MahjongActionModel(
        dropout_rate=0.1,
        input_size=198,  # 新的输入尺寸: 14*4 + 16*3 + 30*3 = 198
    ).to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总数 {total_params/1e6:.2f}M, 可训练 {trainable_params/1e6:.2f}M")
    
    # 损失函数和优化器
    action_criterion = nn.CrossEntropyLoss()
    chi_criterion = nn.CrossEntropyLoss(ignore_index=-1) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )
    
    # 训练记录
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_action_accuracy': [],
        'val_chi_accuracy': []
    }
    
    # 最佳模型跟踪
    best_accuracy = 0.0
    best_model_path = os.path.join(results_dir, "mahjong_action_best.pth")
    
    print(f"开始训练，共 {num_epochs} 个周期...")
    # 在开始训练循环前添加
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用同步CUDA操作，便于定位错误
    try:
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for i, batch in enumerate(train_loader):
                # 获取输入和目标
                features = batch['features'].to(device)
                rush_tile = batch['rush_tile'].to(device)
                turn = batch['turn'].to(device)
                action_targets = batch['action'].to(device)
                action_masks = batch['action_mask'].to(device)
                
                # 创建吃牌目标
                chi_targets = batch['chi_indices'].to(device)  # [batch_size, 2]
                
                # 前向传播
                action_logits, chi_logits = model(features, rush_tile, turn, action_masks)
                
                # 计算动作损失
                action_loss = action_criterion(action_logits, action_targets)
                
                # 仅对需要吃牌的样本计算吃牌损失
                chi_mask = (action_targets == ACTION_CHI)
                chi_loss = 0.0
                
                if chi_mask.sum() > 0:
                    # 从chi_logits中选择需要吃的样本
                    selected_chi_logits = chi_logits[chi_mask]
                    selected_chi_targets = chi_targets[chi_mask]
                    
                    # 计算两张牌的损失（分别针对第一张和第二张牌的位置）
                    chi_loss_1 = chi_criterion(
                        selected_chi_logits[:, :14], 
                        selected_chi_targets[:, 0]
                    )
                    chi_loss_2 = chi_criterion(
                        selected_chi_logits[:, 14:], 
                        selected_chi_targets[:, 1]
                    )
                    chi_loss = chi_loss_1 + chi_loss_2
                
                # 总损失
                loss = action_loss + chi_loss
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 打印进度
                if (i + 1) % 50 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Step [{i+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}, "
                          f"Action Loss: {action_loss.item():.4f}, "
                          f"Chi Loss: {chi_loss:.4f}")
            
            # 计算平均训练损失
            train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            val_loss, action_accuracy, chi_accuracy = evaluate_action_model(
                model, val_loader, action_criterion, chi_criterion, device)
            
            # 更新学习率
            scheduler.step(action_accuracy)
            
            # 记录训练历史
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_action_accuracy'].append(action_accuracy)
            history['val_chi_accuracy'].append(chi_accuracy)
            
            # 打印本轮结果
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Action Accuracy: {action_accuracy:.4f}, "
                  f"Chi Accuracy: {chi_accuracy:.4f}")
            
            # 保存最佳模型 (基于动作准确率)
            if action_accuracy > best_accuracy:
                best_accuracy = action_accuracy
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'action_accuracy': action_accuracy,
                    'chi_accuracy': chi_accuracy,
                    'loss': val_loss,
                }
                torch.save(checkpoint, best_model_path)
                print(f"保存新的最佳模型，准确率: {action_accuracy:.4f}")
        
        # 绘制训练历史图
        setup_chinese_font()
        plot_training_history(history, results_dir)
        
        # 保存最终模型
        final_model_path = os.path.join(results_dir, "mahjong_action_final.pth")
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'action_accuracy': action_accuracy,
            'chi_accuracy': chi_accuracy,
            'loss': val_loss,
        }, final_model_path)
        print(f"保存最终模型")
        
        print(f"\n训练完成! 最佳验证动作准确率: {best_accuracy:.4f}")
        print(f"结果保存在: {results_dir}")
        
        return model, history, results_dir
    
    except KeyboardInterrupt:
        print("\n训练被中断")
        
        # 保存当前模型
        interrupted_model_path = os.path.join(results_dir, "mahjong_action_interrupted.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, interrupted_model_path)
        print(f"已保存中断时的模型: {interrupted_model_path}")
        
        # 绘制训练历史图
        setup_chinese_font()
        plot_training_history(history, results_dir)
        
        return model, history, results_dir

def plot_training_history(history, results_dir):
    """绘制训练历史图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(history['train_loss'], label='训练损失')
    ax1.plot(history['val_loss'], label='验证损失')
    ax1.set_xlabel('轮数')
    ax1.set_ylabel('损失')
    ax1.set_title('损失曲线')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(history['val_action_accuracy'], label='动作准确率')
    ax2.plot(history['val_chi_accuracy'], label='吃牌准确率')
    ax2.set_xlabel('轮数')
    ax2.set_ylabel('准确率')
    ax2.set_title('验证准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_history.png"))
    plt.close()

# 绘制训练历史记录图表
def plot_action_request_history(history, results_dir):
    """绘制吃碰杠胡模型的训练历史图表"""
    # 确保matplotlib可以使用中文
    setup_chinese_font()
    
    # 1. 损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history['epoch'], history['train_loss'], 'b-', label='训练损失')
    plt.plot(history['epoch'], history['val_loss'], 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(results_dir, 'loss_curve.png')
    plt.savefig(loss_path, dpi=300)
    plt.close()
    
    # 2. 动作准确率曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history['epoch'], history['train_action_acc'], 'b-', label='训练动作准确率')
    plt.plot(history['epoch'], history['val_action_acc'], 'r-', label='验证动作准确率')
    plt.title('动作预测准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    action_acc_path = os.path.join(results_dir, 'action_accuracy.png')
    plt.savefig(action_acc_path, dpi=300)
    plt.close()
    
    # 3. 吃牌准确率曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history['epoch'], history['train_chi_acc'], 'b-', label='训练吃牌准确率')
    plt.plot(history['epoch'], history['val_chi_acc'], 'r-', label='验证吃牌准确率')
    plt.title('吃牌索引预测准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    chi_acc_path = os.path.join(results_dir, 'chi_accuracy.png')
    plt.savefig(chi_acc_path, dpi=300)
    plt.close()
    
    # 4. 学习率曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history['epoch'], history['learning_rates'], 'g-')
    plt.title('学习率变化')
    plt.xlabel('Epoch')
    plt.ylabel('学习率')
    plt.yscale('log')
    plt.grid(True)
    lr_path = os.path.join(results_dir, 'learning_rate.png')
    plt.savefig(lr_path, dpi=300)
    plt.close()
    
    # 5. 组合图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 5.1 损失
    axes[0, 0].plot(history['epoch'], history['train_loss'], 'b-', label='训练损失')
    axes[0, 0].plot(history['epoch'], history['val_loss'], 'r-', label='验证损失')
    axes[0, 0].set_title('训练和验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 5.2 动作准确率
    axes[0, 1].plot(history['epoch'], history['train_action_acc'], 'b-', label='训练动作准确率')
    axes[0, 1].plot(history['epoch'], history['val_action_acc'], 'r-', label='验证动作准确率')
    axes[0, 1].set_title('动作预测准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 5.3 吃牌准确率
    axes[1, 0].plot(history['epoch'], history['train_chi_acc'], 'b-', label='训练吃牌准确率')
    axes[1, 0].plot(history['epoch'], history['val_chi_acc'], 'r-', label='验证吃牌准确率')
    axes[1, 0].set_title('吃牌索引预测准确率')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 5.4 学习率
    axes[1, 1].plot(history['epoch'], history['learning_rates'], 'g-')
    axes[1, 1].set_title('学习率变化')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('学习率')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    combined_path = os.path.join(results_dir, 'combined_metrics.png')
    plt.savefig(combined_path, dpi=300)
    plt.close()

# if __name__ == "__main__":
#     model, history, results_dir = train_action_request_model(num_epochs=20, lr=LEARNING_RATE)
#     print(f"训练完成! 结果保存在: {results_dir}")
#     plot_action_request_history(history, results_dir)
'''
# 训练简化模型
python train_action_model.py --mode simple --epochs 30 --lr 1e-4

# 微调模型
python train_action_model.py --mode finetune --base_model path/to/model.pth --epochs 15 --lr 5e-5

# 直接训练完整模型
python train_action_model.py --mode full --epochs 50 --lr 1e-4
'''
if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='训练麻将吃碰杠胡决策模型')
    parser.add_argument('--mode', type=str, choices=['simple', 'finetune', 'full'], default='simple',
                      help='训练模式: simple-训练简化模型, finetune-微调模型, full-直接训练完整模型')
    parser.add_argument('--base_model', type=str, default=None,
                      help='用于微调的基础模型路径')
    parser.add_argument('--epochs', type=int, default=30,
                      help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='学习率')
    parser.add_argument('--debug', action='store_true',
                      help='开启调试模式')
    
    args = parser.parse_args()
    
    # 根据模式选择训练方法
    if args.mode == 'simple':
        print("==== 训练简化吃碰杠胡决策模型 ====")
        simple_model, simple_dir = train_simple_action_model(
            num_epochs=args.epochs,
            lr=args.lr
        )
    elif args.mode == 'finetune':
        print("==== 微调吃碰杠胡决策模型 ====")
        if args.base_model is None:
            print("错误: 微调模式需要指定基础模型路径 (--base_model)")
            sys.exit(1)
        
        fine_tuned_model, ft_dir = fine_tune_action_model(
            base_model_path=args.base_model,
            num_epochs=args.epochs,
            lr=args.lr
        )
    elif args.mode == 'full':
        print("==== 直接训练完整吃碰杠胡决策模型 ====")
        full_model, full_dir = train_full_action_model(
            num_epochs=args.epochs,
            lr=args.lr
        )
    else:
        print(f"错误: 未知的训练模式 {args.mode}")
        sys.exit(1)
    
    print("训练完成!")