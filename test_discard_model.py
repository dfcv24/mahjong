import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from datetime import datetime

from src.models.model import MahjongDiscardModel, SimpleMahjongDiscardModel
from src.utils.data_loader import MahjongDiscardDataset
from src.utils.constants import NUM_TILE_TYPES
from src.utils.train_utils import setup_chinese_font
from src.utils.tile_utils import tile_id_to_chinese

def test_model(model_path, dataset, results_dir=None, simple_model=False, batch_size=64, use_first_14=False):
    """
    测试训练好的麻将打牌决策模型
    
    参数:
    model_path: 模型文件路径
    dataset: 测试数据集
    results_dir: 保存结果的目录，如果为None则基于模型路径创建
    simple_model: 是否为简化模型(只使用手牌和回合)
    batch_size: 批次大小
    use_first_14: 是否只使用前14个特征(手牌)
    
    返回:
    model: 加载的模型
    accuracy: 测试准确率
    results_dir: 结果保存目录
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建结果目录
    if results_dir is None:
        model_dir = os.path.dirname(model_path)
        results_dir = os.path.join(model_dir, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"测试结果将保存在: {results_dir}")
    
    # 加载模型
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建适合的模型类型
    if simple_model:
        model = SimpleMahjongDiscardModel(input_size=14, output_size=NUM_TILE_TYPES + 3).to(device)
    else:
        model = MahjongDiscardModel(input_size=240, output_size=NUM_TILE_TYPES + 3).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型加载成功")
    
    # 创建数据加载器
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"测试数据集加载完成，共 {len(dataset)} 个样本")
    
    # 开始测试
    test_records = []
    correct_count = 0
    total_count = 0
    
    # 添加特殊动作的统计
    special_actions = {
        'hu': {'correct': 0, 'total': 0, 'name': '胡牌'},  # 胡牌
        'angang': {'correct': 0, 'total': 0, 'name': '暗杠'},  # 暗杠
        'minggang': {'correct': 0, 'total': 0, 'name': '明杠'}  # 明杠
    }
    
    # 特殊动作的ID（根据你的常量定义）
    HU_ID = NUM_TILE_TYPES  # 胡牌ID通常是34
    ANGANG_ID = NUM_TILE_TYPES + 1  # 暗杠ID通常是35
    MINGGANG_ID = NUM_TILE_TYPES + 2  # 明杠ID通常是36
    
    special_action_ids = {
        HU_ID: 'hu',
        ANGANG_ID: 'angang',
        MINGGANG_ID: 'minggang'
    }
    
    print("开始测试...")
    with torch.no_grad():
        for batch in test_loader:
            # 根据模型类型选择输入特征
            if simple_model or use_first_14:
                features = batch['features'][:, :14].to(device)  # 只使用手牌
            else:
                features = batch['features'].to(device)  # 使用完整特征
                
            targets = batch['target'].to(device)
            turn = batch['turn'].to(device)
            game_ids = batch['game_id']
            action_mask = batch["action_mask"].to(device)
            
            # 模型预测
            outputs = model(features, turn, action_mask)
            _, predictions = torch.max(outputs, 1)
            
            # 获取前3个预测
            _, top3_preds = torch.topk(outputs, 3, dim=1)
            top3_preds = top3_preds.cpu().numpy()
            
            # 计算准确率
            correct = (predictions == targets)
            correct_count += correct.sum().item()
            total_count += targets.size(0)
            
            # 统计特殊动作的准确率
            for i in range(len(targets)):
                target_id = targets[i].item()
                pred_id = predictions[i].item()
                is_correct = correct[i].item()
                
                # 检查目标是否为特殊动作
                if target_id in special_action_ids:
                    action_type = special_action_ids[target_id]
                    special_actions[action_type]['total'] += 1
                    if is_correct:
                        special_actions[action_type]['correct'] += 1
                
                # 获取中文牌名
                target_tile = tile_id_to_chinese(target_id)
                pred_tile = tile_id_to_chinese(pred_id)
                
                # 获取前3预测的中文牌名
                top3_tiles = [tile_id_to_chinese(tid) for tid in top3_preds[i]]
                top3_str = ", ".join(top3_tiles)
                
                # 是否在前3预测中
                in_top3 = target_id in top3_preds[i]
                
                # 收集结果
                record = {
                    "游戏ID": game_ids[i],
                    "回合": turn[i].item(),
                    "是否正确": "✓" if is_correct else "✗",
                    "在前3中": "✓" if in_top3 else "✗",
                    "实际动作": target_tile,
                    "预测动作": pred_tile,
                    "预测前3": top3_str,
                    "实际ID": target_id,
                    "预测ID": pred_id
                }
                test_records.append(record)
    
    # 计算总体准确率
    accuracy = correct_count / total_count
    
    # 计算特殊动作的准确率
    for action_type in special_actions:
        data = special_actions[action_type]
        if data['total'] > 0:
            data['accuracy'] = data['correct'] / data['total']
        else:
            data['accuracy'] = 0.0
    
    # 保存测试结果
    results_df = pd.DataFrame(test_records)
    csv_path = os.path.join(results_dir, "test_results.csv")
    results_df.to_csv(csv_path, index=False, encoding='utf_8_sig')
    
    # 保存测试总结
    summary_path = os.path.join(results_dir, "test_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"测试样本数: {total_count}\n")
        f.write(f"正确预测数: {correct_count}\n")
        f.write(f"总体准确率: {accuracy:.4f}\n")
        
        # 计算前3准确率
        top3_correct = results_df[results_df["在前3中"] == "✓"].shape[0]
        top3_accuracy = top3_correct / total_count
        f.write(f"前3准确率: {top3_accuracy:.4f}\n\n")
        
        # 添加特殊动作准确率
        f.write("特殊动作准确率:\n")
        for action_type, data in special_actions.items():
            f.write(f"{data['name']}: {data['correct']}/{data['total']} = ")
            if data['total'] > 0:
                f.write(f"{data['accuracy']:.4f}\n")
            else:
                f.write("N/A (无样本)\n")
    
    # 绘制类别分布
    setup_chinese_font()
    plt.figure(figsize=(12, 6))
    
    # 只显示前15个最频繁的分类
    top_classes = results_df["实际动作"].value_counts().nlargest(15)
    sns.barplot(x=top_classes.index, y=top_classes.values)
    plt.title("测试集中最常见的15种牌型")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "class_distribution.png"))
    plt.close()
    
    # 绘制特殊动作准确率
    plt.figure(figsize=(10, 6))
    action_names = []
    action_accuracies = []
    
    for action_type, data in special_actions.items():
        if data['total'] > 0:
            action_names.append(data['name'])
            action_accuracies.append(data['accuracy'])
    
    if action_names:  # 确保有特殊动作数据
        bars = plt.bar(action_names, action_accuracies, color=['coral', 'skyblue', 'lightgreen'])
        plt.title("特殊动作准确率")
        plt.ylabel("准确率")
        plt.ylim(0, 1.1)  # 设置Y轴范围
        
        # 在柱状图上显示准确率值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "special_actions_accuracy.png"))
        plt.close()
    
    # 按回合分析准确率
    analyze_by_turn(results_df, results_dir)
    
    # 输出特殊动作统计
    print("\n特殊动作准确率:")
    for action_type, data in special_actions.items():
        print(f"{data['name']}: ", end="")
        if data['total'] > 0:
            print(f"{data['correct']}/{data['total']} = {data['accuracy']:.4f}")
        else:
            print("无样本")
    
    print(f"\n测试完成!")
    print(f"总样本数: {total_count}")
    print(f"正确预测数: {correct_count}")
    print(f"总体准确率: {accuracy:.4f}")
    print(f"前3准确率: {top3_accuracy:.4f}")
    print(f"详细结果已保存至: {results_dir}")
    
    return model, accuracy, results_dir

def analyze_by_turn(results_df, results_dir):
    """分析不同回合下的准确率"""
    print("按回合数分析准确率...")
    
    # 将回合数分组进行统计
    turn_ranges = [
        (0, 4, "回合1-4"),
        (5, 8, "回合5-8"),
        (9, 12, "回合9-12"),
        (13, 16, "回合13-16"),
        (17, 20, "回合17-20"),
        (21, 24, "回合21-24"),
        (25, 28, "回合25-28"),
        (29, 32, "回合29-32"),
        (33, 100, "回合33+")
    ]
    
    # 准备统计结果容器
    turn_stats = []
    
    # 特殊动作ID
    HU_ID = NUM_TILE_TYPES  # 胡牌ID
    ANGANG_ID = NUM_TILE_TYPES + 1  # 暗杠ID
    MINGGANG_ID = NUM_TILE_TYPES + 2  # 明杠ID
    
    special_action_names = {
        HU_ID: "胡牌",
        ANGANG_ID: "暗杠",
        MINGGANG_ID: "明杠"
    }
    
    # 对每个回合范围进行统计
    for start, end, label in turn_ranges:
        # 筛选当前回合范围的数据
        turn_data = results_df[(results_df["回合"] >= start) & (results_df["回合"] <= end)]
        
        # 如果没有此回合范围的数据，跳过
        if len(turn_data) == 0:
            continue
            
        # 计算准确率
        correct = len(turn_data[turn_data["是否正确"] == "✓"])
        top3_correct = len(turn_data[turn_data["在前3中"] == "✓"])
        total = len(turn_data)
        accuracy = correct / total if total > 0 else 0
        top3_accuracy = top3_correct / total if total > 0 else 0
        
        # 计算特殊动作的准确率
        special_stats = {}
        for action_id, action_name in special_action_names.items():
            action_data = turn_data[turn_data["实际ID"] == action_id]
            action_total = len(action_data)
            if action_total > 0:
                action_correct = len(action_data[action_data["是否正确"] == "✓"])
                action_accuracy = action_correct / action_total
                special_stats[action_name] = {
                    "样本数": action_total,
                    "正确数": action_correct,
                    "准确率": action_accuracy
                }
        
        # 记录结果
        stats = {
            "回合范围": label,
            "样本数": total,
            "正确数": correct,
            "前3正确数": top3_correct,
            "准确率": accuracy,
            "前3准确率": top3_accuracy
        }
        
        # 添加特殊动作统计
        for action_name in ["胡牌", "暗杠", "明杠"]:
            if action_name in special_stats:
                stats[f"{action_name}_样本数"] = special_stats[action_name]["样本数"]
                stats[f"{action_name}_正确数"] = special_stats[action_name]["正确数"]
                stats[f"{action_name}_准确率"] = special_stats[action_name]["准确率"]
            else:
                stats[f"{action_name}_样本数"] = 0
                stats[f"{action_name}_正确数"] = 0
                stats[f"{action_name}_准确率"] = 0.0
        
        turn_stats.append(stats)
    
    # 转换为DataFrame
    turn_stats_df = pd.DataFrame(turn_stats)
    
    # 保存到CSV
    csv_output_path = os.path.join(results_dir, "turn_accuracy.csv")
    turn_stats_df.to_csv(csv_output_path, index=False, encoding='utf_8_sig')
    
    # 绘制回合-准确率曲线
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(turn_stats)), turn_stats_df["准确率"].values, 'o-', linewidth=2, label="准确率")
        plt.plot(range(len(turn_stats)), turn_stats_df["前3准确率"].values, 's--', linewidth=2, label="前3准确率")
        
        # 添加特殊动作准确率曲线（如果有足够数据）
        for action_name in ["胡牌", "暗杠", "明杠"]:
            col_name = f"{action_name}_准确率"
            # 检查是否有任何非零值，避免绘制全零的曲线
            if (turn_stats_df[col_name] > 0).any():
                plt.plot(range(len(turn_stats)), turn_stats_df[col_name].values, '--', linewidth=1.5, alpha=0.7, label=f"{action_name}准确率")
        
        plt.xticks(range(len(turn_stats)), turn_stats_df["回合范围"].values, rotation=45)
        plt.title("不同回合阶段的预测准确率")
        plt.xlabel("回合范围")
        plt.ylabel("准确率")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # 在图上标注准确率值
        for i, acc in enumerate(turn_stats_df["准确率"].values):
            plt.annotate(f"{acc:.2f}", 
                         (i, acc), 
                         textcoords="offset points",
                         xytext=(0,10), 
                         ha='center')
            
        # 保存图表
        chart_path = os.path.join(results_dir, "turn_accuracy_chart.png")
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"回合准确率图表已保存至: {chart_path}")
        
        # 另外生成特殊动作按回合统计图表
        plt.figure(figsize=(12, 8))
        bar_width = 0.25
        index = np.arange(len(turn_stats))
        
        # 绘制特殊动作样本数柱状图
        plt.subplot(2, 1, 1)
        for i, action_name in enumerate(["胡牌", "暗杠", "明杠"]):
            sample_counts = [stat[f"{action_name}_样本数"] for stat in turn_stats]
            plt.bar(index + i*bar_width - bar_width, sample_counts, bar_width, label=f"{action_name}")
        
        plt.xticks(index, turn_stats_df["回合范围"].values, rotation=45)
        plt.title("不同回合阶段的特殊动作样本数")
        plt.ylabel("样本数")
        plt.legend()
        
        # 绘制特殊动作准确率柱状图
        plt.subplot(2, 1, 2)
        for i, action_name in enumerate(["胡牌", "暗杠", "明杠"]):
            accuracy_values = turn_stats_df[f"{action_name}_准确率"].values
            # 只显示有样本的回合段
            for j, acc in enumerate(accuracy_values):
                if turn_stats_df[f"{action_name}_样本数"].values[j] > 0:
                    plt.bar(j + i*bar_width - bar_width, acc, bar_width, label=f"{action_name}" if j==0 else "")
        
        plt.xticks(index, turn_stats_df["回合范围"].values, rotation=45)
        plt.title("不同回合阶段的特殊动作准确率")
        plt.ylabel("准确率")
        plt.legend()
        
        plt.tight_layout()
        special_chart_path = os.path.join(results_dir, "special_actions_by_turn.png")
        plt.savefig(special_chart_path, dpi=300)
        plt.close()
        print(f"特殊动作回合统计图表已保存至: {special_chart_path}")
        
    except Exception as e:
        print(f"绘制图表时出错: {e}")

def compare_models(simple_model_path, full_model_path, dataset):
    """比较简化模型和完整模型的性能"""
    print("\n开始比较简化模型和完整模型...")
    
    # 创建结果目录
    results_dir = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 测试简化模型
    print("\n===== 测试简化模型 =====")
    simple_model, simple_acc, simple_dir = test_model(
        simple_model_path, 
        dataset, 
        os.path.join(results_dir, "simple_model"), 
        simple_model=True
    )
    
    # 测试完整模型
    print("\n===== 测试完整模型 =====")
    full_model, full_acc, full_dir = test_model(
        full_model_path, 
        dataset, 
        os.path.join(results_dir, "full_model"), 
        simple_model=False
    )
    
    # 测试完整模型但只用简化输入
    print("\n===== 测试完整模型(使用简化输入) =====")
    full_simple_model, full_simple_acc, full_simple_dir = test_model(
        full_model_path, 
        dataset, 
        os.path.join(results_dir, "full_model_simple_input"), 
        simple_model=False,
        use_first_14=True
    )
    
    # 比较结果
    comparison = {
        "模型类型": ["简化模型", "完整模型", "完整模型(简化输入)"],
        "准确率": [simple_acc, full_acc, full_simple_acc]
    }
    
    df = pd.DataFrame(comparison)
    
    # 保存比较结果
    df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False, encoding='utf_8_sig')
    
    # 绘制对比图
    plt.figure(figsize=(8, 6))
    bars = plt.bar(df["模型类型"], df["准确率"], color=['blue', 'green', 'orange'])
    plt.title("不同模型的准确率对比")
    plt.xlabel("模型类型")
    plt.ylabel("准确率")
    plt.ylim(0, max(df["准确率"]) * 1.2)  # 给顶部留一点空间显示数值
    
    # 在柱状图上显示准确率值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "model_comparison.png"), dpi=300)
    
    print(f"\n模型比较已完成，结果保存在：{results_dir}")
    return df

def inspect_dataset(dataset):
    """
    检查数据集中的目标分布，特别是特殊动作
    
    参数:
    dataset: 要检查的数据集
    """
    print("\n检查数据集中的目标分布...")
    
    # 使用字典统计每种目标的数量
    target_counts = {}
    special_targets = {
        NUM_TILE_TYPES: "胡牌",
        NUM_TILE_TYPES + 1: "暗杠", 
        NUM_TILE_TYPES + 2: "明杠"
    }
    
    # 遍历数据集
    for i in range(len(dataset)):
        sample = dataset[i]
        target = sample['target'].item()
        
        if target in target_counts:
            target_counts[target] += 1
        else:
            target_counts[target] = 1
    
    # 打印特殊动作的统计
    print("\n特殊动作统计:")
    for target_id, name in special_targets.items():
        count = target_counts.get(target_id, 0)
        percentage = count / len(dataset) * 100 if len(dataset) > 0 else 0
        print(f"{name} (ID={target_id}): {count} 样本 ({percentage:.2f}%)")
    
    # 打印常规牌的总数
    regular_count = sum(target_counts.get(i, 0) for i in range(NUM_TILE_TYPES))
    regular_percentage = regular_count / len(dataset) * 100 if len(dataset) > 0 else 0
    print(f"\n普通打牌动作: {regular_count} 样本 ({regular_percentage:.2f}%)")
    
    # 找出最常见的5种牌
    regular_targets = {k: v for k, v in target_counts.items() if k < NUM_TILE_TYPES}
    top_targets = sorted(regular_targets.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n最常见的5种牌:")
    for target_id, count in top_targets:
        tile_name = tile_id_to_chinese(target_id)
        percentage = count / len(dataset) * 100 if len(dataset) > 0 else 0
        print(f"{tile_name} (ID={target_id}): {count} 样本 ({percentage:.2f}%)")
    
    return target_counts
'''
# 检查数据集
python test_discard_model.py --data path/to/your/test_data --inspect
# 测试简化模型
python test_discard_model.py --model path/to/simple_model.pth --simple --data path/to/test_data

# 测试完整模型
python test_discard_model.py --model path/to/full_model.pth --data path/to/test_data

# 比较简化模型和完整模型
python test_discard_model.py --model path/to/simple_model.pth --compare path/to/full_model.pth --data path/to/test_data
'''
# 在测试代码中添加数据集检查
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='测试麻将打牌决策模型')
    parser.add_argument('--model', type=str, required=False, help='要测试的模型文件路径')
    parser.add_argument('--data', type=str, default="/home/luzhiwei/data/a/mahjong_data_test", 
                        help='测试数据路径')
    parser.add_argument('--simple', action='store_true', help='是否为简化模型')
    parser.add_argument('--compare', type=str, help='用于比较的另一个模型路径')
    parser.add_argument('--max_samples', type=int, default=None, help='最大测试样本数')
    parser.add_argument('--inspect', action='store_true', help='只检查数据集，不进行测试')
    
    args = parser.parse_args()
    
    # 加载测试数据集
    test_dataset = MahjongDiscardDataset(data_folder=args.data, max_samples=args.max_samples)
    
    if args.inspect:
        # 只检查数据集
        target_counts = inspect_dataset(test_dataset)
    elif args.compare:
        # 如果提供了两个模型路径，进行比较
        compare_models(args.model, args.compare, test_dataset)
    else:
        # 测试单个模型
        test_model(args.model, test_dataset, simple_model=args.simple)