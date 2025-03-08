import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.config import BATCH_SIZE
from src.models.model import MahjongTotalModel
from src.utils.constants import *
from src.utils.train_utils import create_results_dir
from src.utils.data_loader import MahjongTotalDataset, split_dataset
from src.utils.train_utils import setup_chinese_font

def setup_training_environment(model_type):
    """设置训练环境和数据"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建结果目录
    results_dir = create_results_dir(f"{model_type}_models")
    print(f"训练结果将保存在: {results_dir}")
    
    # 创建数据集
    full_dataset = MahjongTotalDataset()
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8)
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"划分数据集完成: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
    
    return device, results_dir, train_dataset, val_dataset, train_loader, val_loader, train_size, val_size

def save_training_config(results_dir, config):
    """保存训练配置"""
    with open(os.path.join(results_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)

def plot_training_history(history, results_dir):
    """绘制训练历史图
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_discard_accuracy: [],
        'val_action_accuracy': [],
        'val_chi_accuracy': []
    }
    """
    # 确保matplotlib可以使用中文
    setup_chinese_font()
    
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
    ax2.plot(history['val_discard_accuracy'], label='出牌准确率')
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

def save_model(model, optimizer, epoch, discard_accuracy, action_accuracy, chi_accuracy, val_loss, filepath):
    """保存模型"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'discard_accuracy': discard_accuracy,
        'action_accuracy': action_accuracy,
        'chi_accuracy': chi_accuracy,
        'loss': val_loss,
    }
    
    torch.save(checkpoint, filepath)
    print(f"保存模型: {filepath}")

def train_full_action_model(num_epochs=20, lr=1e-4, results_dir=None):
    """
    训练吃碰杠胡请求模型
    
    参数:
    num_epochs: 训练轮数
    lr: 学习率
    results_dir: 结果保存目录
    
    返回:
    model: 训练好的模型
    history: 训练历史记录
    results_dir: 结果保存目录
    """
    # 设置训练环境
    device, results_dir, train_dataset, val_dataset, train_loader, val_loader, train_size, val_size = setup_training_environment("total")
    
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
    save_training_config(results_dir, config)
    
    # 创建模型
    print("初始化吃碰杠胡决策模型...")
    model = MahjongTotalModel(
        dropout_rate=0.1,
        input_size=198,  # 新的输入尺寸: 14*4 + 16*3 + 30*3 = 198
    ).to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总数 {total_params/1e6:.2f}M, 可训练 {trainable_params/1e6:.2f}M")
    
    # 损失函数和优化器
    discard_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    action_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    chi_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略非吃牌样本
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 训练记录
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_discard_accuracy': [],
        'val_action_accuracy': [],
        'val_chi_accuracy': []
    }
    
    # 最佳模型跟踪
    best_accuracy = 0.0
    best_model_path = os.path.join(results_dir, "mahjong_total_best.pth")
    
    print(f"开始训练，共 {num_epochs} 个周期...")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用同步CUDA操作，便于定位错误
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            # 获取输入和目标
            features = batch['features'].to(device)
            rush_tile_id = batch['rush_tile_id'].to(device)
            turn = batch['turn'].to(device)
            discard_target = batch['discard_target'].to(device)
            discard_mask = batch['discard_mask'].to(device)
            action_targets = batch['action_target'].to(device)
            action_masks = batch['action_mask'].to(device)
            chi_types = batch['chi_type'].to(device)  # 吃牌类型标签
            chi_masks = batch['chi_mask'].to(device)  # 吃牌掩码
            
            # 前向传播
            discard_logits, action_logits, chi_logits = model(features, rush_tile_id, turn, discard_mask, action_masks, chi_masks)
            
            # 计算出牌损失
            discard_loss = discard_criterion(discard_logits, discard_target)

            # 计算动作损失
            action_loss = action_criterion(action_logits, action_targets)
            
            # 仅对需要吃牌的样本计算吃牌损失
            batch_chi_mask = (action_targets == ACTION_CHI)
            chi_loss = 0.0
            
            if batch_chi_mask.sum() > 0:
                # 从chi_logits中选择需要吃的样本
                selected_chi_logits = chi_logits[batch_chi_mask]
                selected_chi_types = chi_types[batch_chi_mask]
                
                # 计算吃牌方式的损失
                chi_loss = chi_criterion(selected_chi_logits, selected_chi_types)
            
            # 总损失
            loss = discard_loss + action_loss + chi_loss
            
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
                        f"Discard Loss: {discard_loss.item():.4f}, "
                        f"Action Loss: {action_loss.item():.4f}, "
                        f"Chi Loss: {chi_loss:.4f}")
        
        # 计算平均训练损失
        train_loss = train_loss / len(train_loader)
        
        # 验证阶段，验证动作准确率和吃牌准确率
        model.eval()
        val_loss = 0.0
        val_discard_correct = 0
        val_discard_total = 0
        val_action_correct = 0
        val_action_total = 0
        val_chi_correct = 0
        val_chi_total = 0
        discard_accuracy = 0
        action_accuracy = 0
        chi_accuracy = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                rush_tile_id = batch['rush_tile_id'].to(device)
                turn = batch['turn'].to(device)
                discard_targets = batch['discard_target'].to(device)
                discard_masks = batch['discard_mask'].to(device)
                action_targets = batch['action_target'].to(device)
                action_masks = batch['action_mask'].to(device)
                chi_types = batch['chi_type'].to(device)
                chi_masks = batch['chi_mask'].to(device)
                
                discard_logits, action_logits, chi_logits = model(features, rush_tile_id, turn, discard_masks, action_masks, chi_masks)
                
                # 计算出牌损失
                discard_loss = discard_criterion(discard_logits, discard_targets)
                # 计算动作损失
                action_loss = action_criterion(action_logits, action_targets)
                # 仅对需要吃牌的样本计算吃牌损失
                batch_chi_mask = (action_targets == ACTION_CHI)
                chi_loss = 0.0
                if batch_chi_mask.sum() > 0:
                    selected_chi_logits = chi_logits[batch_chi_mask]
                    selected_chi_types = chi_types[batch_chi_mask]
                    chi_loss = chi_criterion(selected_chi_logits, selected_chi_types)
                loss = discard_loss + action_loss + chi_loss
                val_loss += loss.item()
                
                # 计算出牌准确率
                _, discard_preds = torch.max(discard_logits, 1)
                discard_correct = (discard_preds == discard_targets).sum().item()
                # discard_targets中-1表示无效牌，不计入准确率
                discard_total = (discard_targets != -1).sum().item()
                val_discard_correct += discard_correct
                val_discard_total += discard_total
                # 计算动作准确率
                _, action_preds = torch.max(action_logits, 1)
                action_correct = (action_preds == action_targets).sum().item()
                # action_targets中-1表示无效动作，不计入准确率
                action_total = (action_targets != -1).sum().item()
                val_action_correct += action_correct
                val_action_total += action_total
                
                # 计算吃牌类型准确率
                if batch_chi_mask.sum() > 0:
                    _, chi_preds = torch.max(selected_chi_logits, 1)
                    chi_correct = (chi_preds == selected_chi_types).sum().item()
                    chi_total = selected_chi_types.size(0)
                    val_chi_correct += chi_correct
                    val_chi_total += chi_total
        
        val_loss = val_loss / len(val_loader)
        # 计算整体准确率
        discard_accuracy = val_discard_correct / val_discard_total if val_discard_total > 0 else 0
        action_accuracy = val_action_correct / val_action_total if val_action_total > 0 else 0
        chi_accuracy = val_chi_correct / val_chi_total if val_chi_total > 0 else 0
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录训练历史
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_discard_accuracy'].append(discard_accuracy)
        history['val_action_accuracy'].append(action_accuracy)
        history['val_chi_accuracy'].append(chi_accuracy)
        
        # 打印本轮结果
        print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Discard Accuracy: {discard_accuracy:.4f}, "
                f"Action Accuracy: {action_accuracy:.4f}, "
                f"Chi Accuracy: {chi_accuracy:.4f}")
        
        # 保存最佳模型 (基于动作准确率)
        if action_accuracy > best_accuracy:
            best_accuracy = action_accuracy
            save_model(model, optimizer, epoch + 1, discard_accuracy, action_accuracy, chi_accuracy, val_loss, best_model_path)
            print(f"保存新的最佳模型，准确率: {action_accuracy:.4f}")
    
        # 绘制训练历史图
        plot_training_history(history, results_dir)
    
    # 保存最终模型
    final_model_path = os.path.join(results_dir, "mahjong_total_final.pth")
    save_model(model, optimizer, num_epochs, discard_accuracy, action_accuracy, chi_accuracy, val_loss, final_model_path)
    
    print(f"\n训练完成! 最佳验证动作准确率: {best_accuracy:.4f}")
    print(f"结果保存在: {results_dir}")
    
    return model, history, results_dir

if __name__ == "__main__":
    train_full_action_model(num_epochs=30)