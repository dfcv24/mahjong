import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from datetime import datetime
import json
import argparse

from src.models.config import BATCH_SIZE, LEARNING_RATE
from src.models.model import MahjongDiscardModel, SimpleMahjongDiscardModel
from src.models.losses import FuzzyLabelLoss
from src.utils.evaluation import evaluate_discard_model, load_and_test_discard_model
from src.utils.train_utils import create_results_dir
from src.utils.data_loader import MahjongDiscardDataset, split_dataset
from src.utils.constants import NUM_TILE_TYPES
from src.utils.train_utils import plot_training_history, setup_chinese_font

def train_simple_model(num_epochs=30, lr=1e-4, results_dir=None):
    """
    使用简化数据(只有手牌和回合信息)训练基础模型
    
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
    results_dir = create_results_dir("simple_discard_models")
    print(f"训练结果将保存在: {results_dir}")
    
    # 加载数据集
    full_dataset = MahjongDiscardDataset()  # 可以使用更多样本，因为模型更简单
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8)
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"划分数据集完成: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False)
    
    # 保存训练配置
    config = {
        "model_type": "simple",
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
    
    # 创建简化模型
    print("初始化简化打牌决策模型...")
    model = SimpleMahjongDiscardModel(
        input_size=14,  # 只使用当前玩家手牌
        output_size=NUM_TILE_TYPES + 3  # 输出：34种牌 + 胡牌 + 暗杠 + 明杠
    ).to(device)
    
    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss()
    criterion = FuzzyLabelLoss()
    val_criterion = nn.CrossEntropyLoss()
    
    # 学习率调整器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # 训练记录
    history = {
        'train_loss': [],
        'train_accuracy': [],  # 添加训练准确率记录
        'val_loss': [],
        'val_accuracy': [],
        'val_top3_accuracy': []
    }
    
    # 最佳模型跟踪
    best_accuracy = 0.0
    best_model_path = os.path.join(results_dir, "simple_mahjong_discard_best.pth")
    
    print(f"开始训练简化模型，共 {num_epochs} 个周期...")
    
    try:
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0  # 记录训练正确预测数
            train_total = 0    # 记录总训练样本数
            
            for i, batch in enumerate(train_loader):
                # 只获取手牌和回合信息
                features = batch['features'][:, :14].to(device)  # 只取前14个元素(手牌)
                targets = batch['target'].to(device)
                turn = batch['turn'].to(device)
                # 获取动作掩码
                if 'action_mask' in batch:
                    action_mask = batch['action_mask'].to(device)
                else:
                    # 如果没有掩码，创建一个默认的（所有动作可用）
                    action_mask = torch.ones(features.size(0), NUM_TILE_TYPES + 3, dtype=torch.bool, device=device)
                
                # 前向传播
                outputs = model(features, turn, action_mask)
                loss = criterion(outputs, targets, turn)

                # 计算训练准确率
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 打印进度
                if (i + 1) % 50 == 0:
                    # 计算当前批次的准确率
                    batch_accuracy = (preds == targets).sum().item() / targets.size(0)
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Step [{i+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}, "
                          f"Accuracy: {batch_accuracy:.4f}")
            
            # 计算平均训练损失
            train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            correct = 0
            top3_correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'][:, :14].to(device)  # 只取前14个元素(手牌)
                    targets = batch['target'].to(device)
                    turn = batch['turn'].to(device)
                    # 获取动作掩码
                    if 'action_mask' in batch:
                        action_mask = batch['action_mask'].to(device)
                    else:
                        action_mask = torch.ones(features.size(0), NUM_TILE_TYPES + 3, dtype=torch.bool, device=device)
                    
                    outputs = model(features, turn, action_mask)
                    loss = val_criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    # 计算准确率
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    
                    # 计算Top-3准确率
                    _, top3_preds = torch.topk(outputs, 3, dim=1)
                    for j in range(len(targets)):
                        if targets[j] in top3_preds[j]:
                            top3_correct += 1
                    
                    total += targets.size(0)
            
            val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            top3_accuracy = top3_correct / total
            
            # 更新学习率
            scheduler.step(val_accuracy)
            
            # 记录训练历史
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)  # 添加训练准确率记录
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_top3_accuracy'].append(top3_accuracy)
            
            # 打印本轮结果，增加训练准确率输出
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Accuracy: {train_accuracy:.4f}, "  # 添加训练准确率输出
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}, "
                  f"Top-3 Accuracy: {top3_accuracy:.4f}")
            
            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_accuracy,
                    'loss': val_loss,
                }
                torch.save(checkpoint, best_model_path)
                print(f"保存新的最佳模型，准确率: {val_accuracy:.4f}")
        
        # 绘制训练历史图
        setup_chinese_font()
        plot_training_history(history, results_dir)
        
        # 保存最终模型
        final_model_path = os.path.join(results_dir, "simple_mahjong_discard_final.pth")
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': val_accuracy,
            'loss': val_loss,
        }, final_model_path)
        print(f"保存最终模型")
        
        print(f"\n简化模型训练完成! 最佳验证准确率: {best_accuracy:.4f}")
        print(f"结果保存在: {results_dir}")
        
        return model, results_dir
    
    except KeyboardInterrupt:
        print("\n训练被中断")
        
        # 保存当前模型
        interrupted_model_path = os.path.join(results_dir, "simple_mahjong_discard_interrupted.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, interrupted_model_path)
        print(f"已保存中断时的模型: {interrupted_model_path}")
        
        # 绘制训练历史图
        setup_chinese_font()
        plot_training_history(history, results_dir)
        
        return model, results_dir

def fine_tune_model(base_model_path, num_epochs=10, lr=5e-5, results_dir=None):
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
    results_dir = create_results_dir("finetuned_discard_models")
    print(f"训练结果将保存在: {results_dir}")
    
    # 加载数据集
    full_dataset = MahjongDiscardDataset()
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
    print("创建完整打牌决策模型进行微调...")
    model = MahjongDiscardModel(
        input_size=198,  # 完整输入大小
        output_size=NUM_TILE_TYPES + 3  # 输出：34种牌 + 胡牌 + 暗杠 + 明杠
    ).to(device)
    
    # 保存训练配置
    config = {
        "model_type": "finetuned",
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
    
    # 将简化模型的权重转移到完整模型的相应部分
    # (这需要根据具体模型架构进行调整)
    # 例如，可以复制嵌入层和部分Transformer编码器层
    
    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss()
    criterion = FuzzyLabelLoss()    
    val_criterion = nn.CrossEntropyLoss()
    
    # 学习率调整器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # 训练记录
    history = {
        'train_loss': [],
        'train_accuracy': [],  # 添加训练准确率记录
        'val_loss': [],
        'val_accuracy': [],
        'val_top3_accuracy': []
    }
    
    # 最佳模型跟踪
    best_accuracy = 0.0
    best_model_path = os.path.join(results_dir, "mahjong_discard_best.pth")
    
    print(f"开始微调模型，共 {num_epochs} 个周期...")
    
    try:
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0  # 记录训练正确预测数
            train_total = 0    # 记录总训练样本数
            
            for i, batch in enumerate(train_loader):
                # 获取完整特征
                features = batch['features'].to(device)
                targets = batch['target'].to(device)
                turn = batch['turn'].to(device)
                # 获取动作掩码
                if 'action_mask' in batch:
                    action_mask = batch['action_mask'].to(device)
                else:
                    action_mask = torch.ones(features.size(0), NUM_TILE_TYPES + 3, dtype=torch.bool, device=device)
                
                # 前向传播
                outputs = model(features, turn, action_mask)
                loss = criterion(outputs, targets, turn)
                
                # 计算训练准确率
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 打印进度
                if (i + 1) % 50 == 0:
                    # 计算当前批次的准确率
                    batch_accuracy = (preds == targets).sum().item() / targets.size(0)
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                        f"Step [{i+1}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f}, "
                        f"Accuracy: {batch_accuracy:.4f}")
            
            # 计算平均训练损失和准确率
            train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total

            # 验证阶段
            val_loss, val_accuracy, top3_accuracy = evaluate_discard_model(
                model, val_loader, val_criterion, device)
            
            # 更新学习率
            scheduler.step(val_accuracy)
            
            # 记录训练历史，增加训练准确率
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)  # 添加训练准确率记录
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_top3_accuracy'].append(top3_accuracy)
            
            # 打印本轮结果，增加训练准确率输出
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.4f}, "  # 添加训练准确率输出
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}, "
                f"Top-3 Accuracy: {top3_accuracy:.4f}")
            
            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_accuracy,
                    'loss': val_loss,
                }
                torch.save(checkpoint, best_model_path)
                print(f"保存新的最佳模型，准确率: {val_accuracy:.4f}")
        
        # 绘制训练历史图
        setup_chinese_font()
        plot_training_history(history, results_dir)
        
        # 保存最终模型
        final_model_path = os.path.join(results_dir, "mahjong_discard_final.pth")
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': val_accuracy,
            'loss': val_loss,
        }, final_model_path)
        print(f"保存最终模型")
        
        # 测试最佳模型
        print("\n测试最佳模型...")
        load_and_test_discard_model(best_model_path, val_dataset, results_dir)
        
        print(f"\n微调完成! 最佳验证准确率: {best_accuracy:.4f}")
        print(f"结果保存在: {results_dir}")
        
        return model, results_dir
        
    except KeyboardInterrupt:
        print("\n训练被中断")
        
        # 保存当前模型
        interrupted_model_path = os.path.join(results_dir, "mahjong_discard_interrupted.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, interrupted_model_path)
        print(f"已保存中断时的模型: {interrupted_model_path}")
        
        # 绘制训练历史图
        setup_chinese_font()
        plot_training_history(history, results_dir)
        
        return model, results_dir

def train_full_model(num_epochs=30, lr=1e-4, results_dir=None):
    """
    直接训练完整模型，不使用简化模型预训练
    
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
    results_dir = create_results_dir("full_discard_models")
    print(f"训练结果将保存在: {results_dir}")
    
    # 加载数据集
    full_dataset = MahjongDiscardDataset()  # 使用全部数据
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8)
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"划分数据集完成: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False)
    
    # 保存训练配置
    config = {
        "model_type": "full",
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
    
    # 创建完整模型
    print("初始化完整打牌决策模型...")
    model = MahjongDiscardModel(
        input_size=198,  # 完整输入大小
        output_size=NUM_TILE_TYPES + 3  # 输出：34种牌 + 胡牌 + 暗杠 + 明杠
    ).to(device)
    
    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss()
    criterion = FuzzyLabelLoss()
    val_criterion = nn.CrossEntropyLoss()
    
    # 学习率调整器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
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
    best_model_path = os.path.join(results_dir, "full_mahjong_discard_best.pth")
    
    print(f"开始训练完整模型，共 {num_epochs} 个周期...")
    
    try:
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i, batch in enumerate(train_loader):
                # 获取完整特征和目标
                features = batch['features'].to(device)
                targets = batch['target'].to(device)
                turn = batch['turn'].to(device)
                
                # 获取动作掩码
                if 'action_mask' in batch:
                    action_mask = batch['action_mask'].to(device)
                else:
                    action_mask = torch.ones(features.size(0), NUM_TILE_TYPES + 3, dtype=torch.bool, device=device)
                
                # 前向传播
                outputs = model(features, turn, action_mask)
                loss = criterion(outputs, targets, turn)
                
                # 计算训练准确率
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 打印进度
                if (i + 1) % 50 == 0:
                    batch_accuracy = (preds == targets).sum().item() / targets.size(0)
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Step [{i+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}, "
                          f"Accuracy: {batch_accuracy:.4f}")
            
            # 计算平均训练损失和准确率
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_accuracy = train_correct / train_total
            
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
                    targets = batch['target'].to(device)
                    turn = batch['turn'].to(device)
                    
                    # 获取动作掩码
                    if 'action_mask' in batch:
                        action_mask = batch['action_mask'].to(device)
                    else:
                        action_mask = torch.ones(features.size(0), NUM_TILE_TYPES + 3, dtype=torch.bool, device=device)
                    
                    outputs = model(features, turn, action_mask)
                    loss = val_criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    # 计算准确率
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    
                    # 计算Top-3准确率
                    _, top3_preds = torch.topk(outputs, 3, dim=1)
                    for j in range(len(targets)):
                        if targets[j] in top3_preds[j]:
                            top3_correct += 1
                    
                    total += targets.size(0)
            
            # 计算平均验证损失和准确率
            epoch_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            top3_accuracy = top3_correct / total
            
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
        final_model_path = os.path.join(results_dir, "full_mahjong_discard_final.pth")
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
        load_and_test_discard_model(best_model_path, val_dataset, results_dir)
        
        print(f"\n完整模型训练完成! 最佳验证准确率: {best_accuracy:.4f}")
        print(f"结果保存在: {results_dir}")
        
        return model, results_dir
        
    except KeyboardInterrupt:
        print("\n训练被中断")
        
        # 保存当前模型
        interrupted_model_path = os.path.join(results_dir, "full_mahjong_discard_interrupted.pth")
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

'''
# 训练简化模型
python train_discard_model.py --mode simple --epochs 30 --lr 1e-4

# 微调模型
python train_discard_model.py --mode finetune --base_model path/to/model.pth --epochs 15 --lr 5e-5

# 直接训练完整模型
python train_discard_model.py --mode full --epochs 50 --lr 1e-4
'''
if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='训练麻将打牌决策模型')
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
        print("==== 训练简化模型 ====")
        simple_model, simple_dir = train_simple_model(
            num_epochs=args.epochs,
            lr=args.lr
        )
    elif args.mode == 'finetune':
        print("==== 微调模型 ====")
        if args.base_model is None:
            print("错误: 微调模式需要指定基础模型路径 (--base_model)")
            sys.exit(1)
        
        fine_tuned_model, ft_dir = fine_tune_model(
            base_model_path=args.base_model,
            num_epochs=args.epochs,
            lr=args.lr
        )
    elif args.mode == 'full':
        print("==== 直接训练完整模型 ====")
        full_model, full_dir = train_full_model(
            num_epochs=args.epochs,
            lr=args.lr
        )
    else:
        print(f"错误: 未知的训练模式 {args.mode}")
        sys.exit(1)
    
    print("训练完成!")