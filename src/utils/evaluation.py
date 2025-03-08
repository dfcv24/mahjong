import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter
from src.models.model import MahjongDiscardModel, SimpleMahjongDiscardModel
from src.utils.constants import NUM_TILE_TYPES
from src.utils.train_utils import setup_chinese_font
from src.utils.constants import *

def evaluate_discard_model(model, data_loader, criterion, device):
    """
    评估麻将打牌决策模型
    
    参数:
    model: 模型
    data_loader: 数据加载器
    criterion: 损失函数
    device: 计算设备
    
    返回:
    avg_loss: 平均损失
    accuracy: 准确率
    top3_accuracy: Top-3准确率
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    top3_correct = 0
    total = 0
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # 检查是否为简化模型
            if isinstance(model, SimpleMahjongDiscardModel):
                features = batch['features'][:, :14].to(device)  # 只取前14个元素(手牌)
            else:
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
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            
            # 计算Top-3准确率
            _, top3_preds = torch.topk(outputs, 3, dim=1)
            for j in range(len(targets)):
                if targets[j] in top3_preds[j]:
                    top3_correct += 1
            
            total += targets.size(0)
            
            # 保存预测和真实标签用于后续分析
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
    
    # 计算平均指标
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else float('inf')
    accuracy = correct / total if total > 0 else 0
    top3_accuracy = top3_correct / total if total > 0 else 0
    
    return avg_loss, accuracy, top3_accuracy, predictions, true_labels

def load_and_test_discard_model(model_path, test_dataset, results_dir):
    """
    加载模型并在测试数据上进行评估
    
    参数:
    model_path: 模型路径
    test_dataset: 测试数据集
    results_dir: 保存结果的目录
    
    返回:
    model: 加载的模型
    test_results: 测试结果字典
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 确定模型类型
    # 这里根据模型路径或checkpoint中的信息来判断是简化模型还是完整模型
    is_simple_model = "simple" in model_path.lower()
    
    if is_simple_model:
        model = SimpleMahjongDiscardModel(
            input_size=14,
            output_size=NUM_TILE_TYPES + 3
        ).to(device)
    else:
        model = MahjongDiscardModel(
            input_size=240,
            output_size=NUM_TILE_TYPES + 3
        ).to(device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 评估模型
    criterion = nn.CrossEntropyLoss()
    test_loss, accuracy, top3_accuracy, predictions, true_labels = evaluate_discard_model(
        model, test_loader, criterion, device
    )
    
    print(f"测试损失: {test_loss:.4f}")
    print(f"准确率: {accuracy:.4f}")
    print(f"Top-3准确率: {top3_accuracy:.4f}")
    
    # 保存测试结果
    test_results = {
        'loss': test_loss,
        'accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'predictions': predictions,
        'true_labels': true_labels
    }
    
    # 创建结果目录
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 保存测试指标
    metrics_path = os.path.join(results_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"测试损失: {test_loss:.4f}\n")
        f.write(f"准确率: {accuracy:.4f}\n")
        f.write(f"Top-3准确率: {top3_accuracy:.4f}\n")
    
    # 分析和可视化结果
    analyze_predictions(predictions, true_labels, results_dir)
    
    return model, test_results

def analyze_predictions(predictions, true_labels, results_dir):
    """
    分析预测结果并生成可视化报告
    
    参数:
    predictions: 模型预测
    true_labels: 真实标签
    results_dir: 结果保存目录
    """
    setup_chinese_font()
    
    # 确保输入是numpy数组
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # 1. 计算每个类别的准确率
    unique_labels = np.unique(np.concatenate([predictions, true_labels]))
    class_accuracies = {}
    
    for label in unique_labels:
        if label < NUM_TILE_TYPES:  # 只分析牌类，不分析特殊动作
            # 找出真实标签为当前类的样本
            mask = (true_labels == label)
            if np.sum(mask) > 0:
                # 计算这些样本中预测正确的比例
                correct = np.sum(predictions[mask] == label)
                class_accuracies[label] = correct / np.sum(mask)
    
    # 2. 生成混淆矩阵
    # 仅使用实际出现在数据中的类别
    present_classes = sorted(list(set(true_labels) | set(predictions)))
    cm = confusion_matrix(true_labels, predictions, labels=present_classes)
    
    # 3. 可视化混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d', 
                xticklabels=[CHINESE_TILE_NAMES.get(i, f"动作{i-NUM_TILE_TYPES}") for i in present_classes],
                yticklabels=[CHINESE_TILE_NAMES.get(i, f"动作{i-NUM_TILE_TYPES}") for i in present_classes])
    plt.title('预测混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # 4. 绘制类别准确率条形图
    if class_accuracies:
        # 按照牌的类型进行分组和排序
        sorted_labels = sorted(class_accuracies.keys())
        accuracies = [class_accuracies[label] for label in sorted_labels]
        
        plt.figure(figsize=(15, 6))
        bars = plt.bar(range(len(sorted_labels)), accuracies)
        plt.xticks(range(len(sorted_labels)), 
                  [CHINESE_TILE_NAMES.get(i, str(i)) for i in sorted_labels], 
                  rotation=45, ha='right')
        plt.xlabel('牌类')
        plt.ylabel('准确率')
        plt.title('各牌类的预测准确率')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在条形上标记数值
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.savefig(os.path.join(results_dir, 'class_accuracies.png'), dpi=300)
        plt.close()
    
    # 5. 分析预测错误的样本
    error_indices = np.where(predictions != true_labels)[0]
    error_count = len(error_indices)
    
    # 统计最常见的错误类型
    error_types = []
    for i in error_indices:
        error_types.append((true_labels[i], predictions[i]))
    
    error_counter = Counter(error_types)
    most_common_errors = error_counter.most_common(10)
    
    # 保存错误分析
    error_analysis_path = os.path.join(results_dir, "error_analysis.txt")
    with open(error_analysis_path, "w") as f:
        f.write(f"总错误数: {error_count}\n\n")
        f.write("最常见的错误类型 (真实->预测):\n")
        
        for (true_label, pred_label), count in most_common_errors:
            true_name = CHINESE_TILE_NAMES.get(true_label, f"动作{true_label-NUM_TILE_TYPES}")
            pred_name = CHINESE_TILE_NAMES.get(pred_label, f"动作{pred_label-NUM_TILE_TYPES}")
            f.write(f"{true_name} -> {pred_name}: {count} 次\n")
    
    print(f"分析结果已保存到 {results_dir}")

def evaluate_detailed_discard_model(model, val_loader, device, results_dir=None):
    """
    在验证集上详细评估模型，计算每个类别的准确率并生成混淆矩阵
    
    参数:
    model: 要评估的模型
    val_loader: 验证数据加载器
    device: 计算设备
    results_dir: 结果保存目录，如果提供则保存图表
    
    返回:
    dict: 包含每个类别准确率的字典
    """
    model.eval()
    
    # 用于收集每个类别的预测结果
    class_correct = np.zeros(NUM_TILE_TYPES + 1)  # +1是胡牌选项
    class_total = np.zeros(NUM_TILE_TYPES + 1)
    
    # 用于构建混淆矩阵
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for hand, turn, action in val_loader:
            hand = hand.to(device)
            turn = turn.to(device)
            action = action.to(device)
            
            # 前向传播
            logits = model(hand, turn)
            pred = torch.argmax(logits, dim=1)
            
            # 收集预测结果
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(action.cpu().numpy())
            
            # 更新每个类别的统计数据
            for i in range(len(action)):
                label = action[i].item()
                class_total[label] += 1
                if pred[i].item() == label:
                    class_correct[label] += 1
    
    # 计算每个类别的准确率
    class_accuracy = {}
    for i in range(NUM_TILE_TYPES + 1):
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
            class_accuracy[i] = accuracy
    
    # 计算总体准确率
    total_accuracy = sum(class_correct) / sum(class_total)
    
    # 输出结果
    print(f"\n总体准确率: {total_accuracy:.4f}")
    
    # 按牌型范围输出准确率
    categories = [
        ("万子 (0-8)", range(0, 9)),
        ("条子 (9-17)", range(9, 18)),
        ("饼子 (18-26)", range(18, 27)),
        ("风牌 (27-30)", range(27, 31)),
        ("箭牌 (31-33)", range(31, 34)),
        ("胡牌", [NUM_TILE_TYPES])
    ]
    
    print("\n各类别准确率:")
    for name, indices in categories:
        total = sum(class_total[i] for i in indices)
        if total > 0:
            correct = sum(class_correct[i] for i in indices)
            acc = correct / total
            samples = int(total)
            print(f"{name}: {acc:.4f} ({samples} 样本)")
    
    # 输出各牌型详细准确率
    print("\n详细牌型准确率:")
    for i in range(NUM_TILE_TYPES + 1):
        if class_total[i] > 0:
            if i == NUM_TILE_TYPES:
                tile_name = "胡牌"
            else:
                tile_name = f"牌型 {i}"
            
            acc = class_correct[i] / class_total[i]
            samples = int(class_total[i])
            if samples >= 10:  # 只显示样本数足够的类别
                print(f"{tile_name}: {acc:.4f} ({samples} 样本)")
    
    # 分析预测偏差
    if results_dir:
        # 绘制混淆矩阵
        try:
            # 为了可视化，我们限制混淆矩阵的大小
            # 选择样本数最多的类别和胡牌
            top_classes = np.argsort(-class_total)[:20]  # 取样本数最多的20个类别
            if NUM_TILE_TYPES not in top_classes:  # 确保包含胡牌
                top_classes[-1] = NUM_TILE_TYPES
            
            # 筛选这些类别的数据
            selected_indices = [i for i, target in enumerate(all_targets) if target in top_classes]
            filtered_preds = [all_preds[i] for i in selected_indices]
            filtered_targets = [all_targets[i] for i in selected_indices]
            
            # 进一步筛选，只保留预测和真实标签都在top_classes中的数据
            final_preds = []
            final_targets = []
            for i in range(len(filtered_preds)):
                if filtered_preds[i] in top_classes and filtered_targets[i] in top_classes:
                    final_preds.append(filtered_preds[i])
                    final_targets.append(filtered_targets[i])
            
            # 映射到新的连续索引
            class_mapping = {cls: i for i, cls in enumerate(sorted(top_classes))}
            mapped_preds = [class_mapping[p] for p in final_preds]
            mapped_targets = [class_mapping[t] for t in final_targets]
            
            # 计算混淆矩阵
            conf_matrix = confusion_matrix(mapped_targets, mapped_preds)
            
            # 绘制混淆矩阵
            plt.figure(figsize=(12, 10))
            
            # 创建标签
            labels = []
            for cls in sorted(top_classes):
                if cls == NUM_TILE_TYPES:
                    labels.append("胡")
                else:
                    labels.append(str(cls))
            
            # 使用seaborn绘制热图
            ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                          xticklabels=labels, yticklabels=labels)
            plt.title('预测混淆矩阵 (主要类别)')
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300)
            plt.close()
            print(f"混淆矩阵已保存至 {os.path.join(results_dir, 'confusion_matrix.png')}")
            
            # 分析类别分布
            plt.figure(figsize=(12, 6))
            # 只展示有样本的类别
            active_classes = [i for i in range(NUM_TILE_TYPES + 1) if class_total[i] > 0]
            samples = [class_total[i] for i in active_classes]
            
            x_positions = np.arange(len(active_classes))
            plt.bar(x_positions, samples)
            
            # 设置x轴标签
            x_labels = []
            for cls in active_classes:
                if cls == NUM_TILE_TYPES:
                    x_labels.append("胡")
                else:
                    x_labels.append(str(cls))
            
            plt.xticks(x_positions, x_labels, rotation=90)
            plt.title('验证集类别分布')
            plt.xlabel('类别')
            plt.ylabel('样本数')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'class_distribution.png'), dpi=300)
            plt.close()
            print(f"类别分布图已保存至 {os.path.join(results_dir, 'class_distribution.png')}")
            
            # 准确率与样本数的关系
            plt.figure(figsize=(12, 6))
            acc_values = []
            sample_counts = []
            labels = []
            
            for cls in active_classes:
                if class_total[cls] >= 10:  # 只考虑样本充足的类别
                    acc_values.append(class_correct[cls] / class_total[cls])
                    sample_counts.append(class_total[cls])
                    if cls == NUM_TILE_TYPES:
                        labels.append("胡")
                    else:
                        labels.append(str(cls))
            
            # 绘制散点图
            plt.scatter(sample_counts, acc_values, alpha=0.6)
            
            # 添加标签
            for i, label in enumerate(labels):
                plt.annotate(label, (sample_counts[i], acc_values[i]))
                
            plt.title('类别准确率与样本数关系')
            plt.xlabel('样本数')
            plt.ylabel('准确率')
            plt.axhline(y=total_accuracy, color='r', linestyle='-', alpha=0.3, label=f'平均准确率: {total_accuracy:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'accuracy_vs_samples.png'), dpi=300)
            plt.close()
            print(f"准确率与样本数关系图已保存至 {os.path.join(results_dir, 'accuracy_vs_samples.png')}")
            
        except Exception as e:
            print(f"绘制分析图表时出错: {e}")
    
    return class_accuracy, total_accuracy


def evaluate_simple_action_model(model, data_loader, action_criterion, chi_criterion, device):
    model.eval()
    val_loss = 0.0
    action_correct = 0
    action_total = 0
    chi_correct = 0
    chi_total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            try:
                # 获取输入和目标
                features = batch['features'][:,:14].to(device)
                rush_tile = batch['rush_tile'].to(device)
                turn = batch['turn'].to(device)
                action_targets = batch['action'].to(device)
                
                # 确保action_mask是有效的
                if 'action_mask' in batch:
                    action_masks = batch['action_mask'].to(device)
                else:
                    # 如果没有提供action_mask，创建一个默认的
                    action_masks = torch.ones(features.size(0), 5, dtype=torch.bool, device=device)
                
                # 创建吃牌目标 - 确保一致的字段命名
                if 'chi_indices' in batch:
                    chi_targets = batch['chi_indices'].to(device)  # [batch_size, 2]
                elif 'chi_positions' in batch:
                    chi_targets = batch['chi_positions'].to(device)  # [batch_size, 2]
                else:
                    # 创建默认的吃牌目标
                    chi_targets = torch.full((features.size(0), 2), -1, dtype=torch.long, device=device)
                
                # 前向传播
                action_logits, chi_logits = model(features, rush_tile, turn, action_masks)
                
                # 计算动作损失和准确率
                action_loss = action_criterion(action_logits, action_targets)
                _, action_preds = torch.max(action_logits, dim=1)
                action_correct += (action_preds == action_targets).sum().item()
                action_total += action_targets.size(0)
                
                # 计算吃牌损失和准确率
                chi_mask = (action_targets == ACTION_CHI)
                chi_loss = 0.0
                
                if chi_mask.sum() > 0:
                    selected_chi_logits = chi_logits[chi_mask]
                    selected_chi_types = batch['chi_type'][chi_mask].to(device)
                    
                    chi_loss = chi_criterion(selected_chi_logits, selected_chi_types)
                    
                    # 计算吃牌准确率
                    _, chi_preds = torch.max(selected_chi_logits, dim=1)
                    chi_correct += (chi_preds == selected_chi_types).sum().item()
                    chi_total += selected_chi_types.size(0)
                
                # 总损失
                loss = action_loss + chi_loss
                val_loss += loss.item()
                
            except Exception as e:
                print(f"评估时出错: {e}")
                continue
    
    # 计算平均损失和准确率
    avg_val_loss = val_loss / len(data_loader)
    action_accuracy = action_correct / action_total if action_total > 0 else 0
    chi_accuracy = chi_correct / chi_total if chi_total > 0 else 0
    
    return avg_val_loss, action_accuracy, chi_accuracy
# def evaluate_model(model, dataloader, device):
#     """评估模型性能"""
#     model.eval()
#     all_preds = []
#     all_targets = []
#     all_turns = []
    
#     with torch.no_grad():
#         for hands, turns, targets in dataloader:
#             hands = hands.to(device)
#             turns = turns.to(device)
#             targets = targets.to(device)
            
#             logits = model(hands, turns)
#             preds = torch.argmax(logits, dim=1)
            
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(targets.cpu().numpy())
#             all_turns.extend(turns.cpu().numpy())
    
#     # 计算准确率
#     accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    
#     # 按回合统计准确率
#     turn_accuracy = {}
#     for turn, pred, target in zip(all_turns, all_preds, all_targets):
#         turn = int(turn)
#         if turn not in turn_accuracy:
#             turn_accuracy[turn] = {"correct": 0, "total": 0}
        
#         turn_accuracy[turn]["total"] += 1
#         if pred == target:
#             turn_accuracy[turn]["correct"] += 1
    
#     for turn in turn_accuracy:
#         turn_accuracy[turn]["accuracy"] = turn_accuracy[turn]["correct"] / turn_accuracy[turn]["total"]
    
#     return accuracy, turn_accuracy

# def plot_confusion_matrix(all_preds, all_targets, save_path):
#     """绘制混淆矩阵"""
#     cm = confusion_matrix(all_targets, all_preds)
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.savefig(save_path)
#     plt.close()

# def save_evaluation_results(accuracy, turn_accuracy, save_dir):
#     """保存评估结果"""
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 保存总体准确率
#     with open(os.path.join(save_dir, 'accuracy.txt'), 'w') as f:
#         f.write(f"Overall accuracy: {accuracy:.4f}\n")
    
#     # 保存按回合的准确率
#     with open(os.path.join(save_dir, 'turn_accuracy.json'), 'w') as f:
#         json.dump(turn_accuracy, f, indent=4)
    
#     # 绘制按回合的准确率曲线
#     turns = sorted(list(turn_accuracy.keys()))
#     accuracies = [turn_accuracy[turn]["accuracy"] for turn in turns]
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(turns, accuracies, 'o-', linewidth=2)
#     plt.xlabel('Turn')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy by Turn')
#     plt.grid(True)
#     plt.savefig(os.path.join(save_dir, 'turn_accuracy.png'))
#     plt.close()