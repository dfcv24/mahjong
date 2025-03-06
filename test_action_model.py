import os
import torch
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
import matplotlib as plt

from src.models.model import MahjongActionModel
from src.utils.data_loader import MahjongActionDataset
from src.utils.train_utils import setup_chinese_font
from src.utils.tile_utils import format_hand_chinese, tile_id_to_chinese
from src.utils.constants import ACTION_CHI, NUM_TILE_TYPES

# 设置中文字体
setup_chinese_font()

def test_action_request_model(model_path, test_dataset, results_dir=None):
    """使用测试数据集测试吃碰杠胡请求模型，并保存详细结果"""
    # 创建结果目录
    if not results_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/tests/action_model/test_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = MahjongActionModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    
    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 准备结果收集
    results = []
    action_correct = 0
    action_total = 0
    chi_correct = 0
    chi_total = 0
    
    print(f"开始测试模型: {model_path}")
    print(f"测试样本数: {len(test_dataset)}")
    
    # 详细记录测试结果的文件
    detail_file_path = os.path.join(results_dir, "test_details.txt")
    summary_file_path = os.path.join(results_dir, "test_summary.txt")
    
    with open(detail_file_path, 'w', encoding='utf-8') as detail_file:
        detail_file.write(f"麻将吃碰杠胡请求模型测试结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        detail_file.write(f"模型: {model_path}\n")
        detail_file.write(f"测试样本数: {len(test_dataset)}\n")
        detail_file.write("-" * 80 + "\n\n")
        
        for batch_idx, batch in enumerate(test_loader):
            # 移动到设备
            hand = batch["hand"].to(device)
            rush_tile = batch["rush_tile"].to(device)
            turn = batch["turn"].to(device)
            action_mask = batch["action_mask"].to(device)
            action = batch["action"].to(device)
            chi_tiles = batch["chi_tiles"].to(device)
            
            # 前向传播
            with torch.no_grad():
                action_logits, chi_logits, _ = model(hand, rush_tile, turn, action_mask)
                action_probs = torch.softmax(action_logits, dim=1)
                
                # 动作预测
                pred_action = torch.argmax(action_logits, dim=1)
                
                # 计算动作准确率
                action_correct_batch = (pred_action == action).sum().item()
                action_correct += action_correct_batch
                action_total += action.size(0)
                
                # 对吃牌样本计算额外准确率
                is_chi = (action == ACTION_CHI)
                if is_chi.any():
                    chi_samples = chi_tiles[is_chi]
                    sample_hands = hand[is_chi]
                    
                    # 计算真实的吃牌索引
                    chi_indices = []
                    for i in range(len(chi_samples)):
                        current_hand = sample_hands[i]
                        current_chi_tiles = chi_samples[i]
                        
                        indices = []
                        for tile in current_chi_tiles:
                            found = False
                            for j in range(len(current_hand)):
                                if current_hand[j].item() == tile.item() and j not in indices:
                                    indices.append(j)
                                    found = True
                                    break
                            if not found:
                                for j in range(len(current_hand)):
                                    if j not in indices and current_hand[j].item() != NUM_TILE_TYPES:
                                        indices.append(j)
                                        break
                        
                        while len(indices) < 2:
                            for j in range(len(current_hand)):
                                if j not in indices and current_hand[j].item() != NUM_TILE_TYPES:
                                    indices.append(j)
                                    break
                        
                        indices = indices[:2]
                        chi_indices.append(indices)
                    
                    chi_indices = torch.tensor(chi_indices, device=device)
                    chi_preds = chi_logits[is_chi]
                    chi_pred_indices = chi_preds.argmax(dim=-1)
                    
                    # 计算吃牌索引准确率
                    chi_correct_batch = (chi_pred_indices == chi_indices).sum().item()
                    chi_correct += chi_correct_batch
                    chi_total += chi_indices.numel()
                
                # 收集每个样本的详细结果
                for i in range(len(action)):
                    sample_id = batch_idx * test_loader.batch_size + i
                    
                    # 手牌和rush牌转为中文
                    hand_tiles = hand[i].cpu().tolist()
                    hand_chinese = format_hand_chinese(hand_tiles)
                    rush_tile_id = rush_tile[i].item()
                    rush_tile_chinese = tile_id_to_chinese(rush_tile_id)
                    
                    # 实际动作和预测动作
                    actual_action = action[i].item()
                    predicted_action = pred_action[i].item()
                    
                    # 动作名称
                    action_names = ["过/跳过", "吃", "碰", "杠", "胡"]
                    actual_action_name = action_names[actual_action]
                    predicted_action_name = action_names[predicted_action]
                    
                    # 动作掩码
                    mask = action_mask[i].cpu().tolist()
                    available_actions = [action_names[j] for j, available in enumerate(mask) if available]
                    
                    # 是否正确预测
                    is_action_correct = (actual_action == predicted_action)
                    
                    # 如果是吃牌，记录吃牌信息
                    chi_info = ""
                    if actual_action == ACTION_CHI:
                        idx = (is_chi.nonzero(as_tuple=True)[0] == i).nonzero(as_tuple=True)[0]
                        if len(idx) > 0:
                            idx = idx[0].item()
                            actual_chi_tiles = [hand_tiles[chi_indices[idx][0].item()], hand_tiles[chi_indices[idx][1].item()]]
                            actual_chi_chinese = [tile_id_to_chinese(tile) for tile in actual_chi_tiles if tile < NUM_TILE_TYPES]
                            
                            if predicted_action == ACTION_CHI:
                                # 找到预测的吃牌索引
                                pred_idx0 = chi_pred_indices[idx][0].item()
                                pred_idx1 = chi_pred_indices[idx][1].item()
                                
                                if pred_idx0 < len(hand_tiles) and pred_idx1 < len(hand_tiles):
                                    pred_chi_tiles = [hand_tiles[pred_idx0], hand_tiles[pred_idx1]]
                                    pred_chi_chinese = [tile_id_to_chinese(tile) for tile in pred_chi_tiles if tile < NUM_TILE_TYPES]
                                    
                                    chi_correct_flag = "✓" if all((chi_pred_indices[idx] == chi_indices[idx]).tolist()) else "✗"
                                    chi_info = f"实际吃牌组合: {'+'.join(actual_chi_chinese)}+{rush_tile_chinese}, " \
                                              f"预测吃牌组合: {'+'.join(pred_chi_chinese)}+{rush_tile_chinese} {chi_correct_flag}"
                    
                    # 记录结果
                    results.append({
                        "样本ID": sample_id + 1,
                        "手牌": hand_chinese,
                        "Rush牌": rush_tile_chinese,
                        "回合": turn[i].item(),
                        "可用动作": ", ".join(available_actions),
                        "实际动作": actual_action_name,
                        "预测动作": predicted_action_name,
                        "动作准确": "✓" if is_action_correct else "✗",
                        "预测概率": action_probs[i][predicted_action].item(),
                        "吃牌信息": chi_info
                    })
                    
                    # 写入详细记录
                    detail_file.write(f"样本 #{sample_id + 1}:\n")
                    detail_file.write(f"  手牌: {hand_chinese}\n")
                    detail_file.write(f"  Rush牌: {rush_tile_chinese}\n")
                    detail_file.write(f"  回合: {turn[i].item()}\n")
                    detail_file.write(f"  可用动作: {', '.join(available_actions)}\n")
                    detail_file.write(f"  实际动作: {actual_action_name}\n")
                    detail_file.write(f"  预测动作: {predicted_action_name} " \
                                     f"(概率: {action_probs[i][predicted_action].item():.4f})\n")
                    if chi_info:
                        detail_file.write(f"  {chi_info}\n")
                    detail_file.write(f"  结果: {'正确 ✓' if is_action_correct else '错误 ✗'}\n\n")
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"已测试: {batch_idx + 1}/{len(test_loader)} - ")
                print(f"动作准确率: {action_correct/action_total:.4f} - ")
                print(f"吃牌准确率: {chi_correct/chi_total if chi_total > 0 else 'N/A'}")
    
    # 计算最终准确率
    action_accuracy = action_correct / action_total
    chi_accuracy = chi_correct / chi_total if chi_total > 0 else 0
    
    # 保存汇总信息
    with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write(f"麻将吃碰杠胡请求模型测试汇总 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary_file.write(f"模型: {model_path}\n")
        summary_file.write(f"测试样本数: {action_total}\n")
        summary_file.write(f"动作准确率: {action_accuracy:.4f}\n")
        summary_file.write(f"吃牌样本数: {chi_total//2}\n")  # 除以2因为每个吃牌有两个索引
        summary_file.write(f"吃牌准确率: {chi_accuracy:.4f}\n")
    
    # 将结果保存为CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "test_results.csv")
    df.to_csv(csv_path, index=False, encoding='utf_8_sig')  # 使用BOM标记确保Excel正确显示中文
    
    # 分析不同动作类型的准确率
    action_stats = {}
    for action_id, action_name in enumerate(["过/跳过", "吃", "碰", "杠", "胡"]):
        action_df = df[df["实际动作"] == action_name]
        if len(action_df) > 0:
            correct = len(action_df[action_df["动作准确"] == "✓"])
            accuracy = correct / len(action_df)
            action_stats[action_name] = {
                "总数": len(action_df),
                "正确数": correct,
                "准确率": accuracy
            }
    
    # 将动作统计保存到文件
    with open(os.path.join(results_dir, "action_stats.txt"), 'w', encoding='utf-8') as f:
        f.write("动作类型统计结果:\n")
        f.write(f"{'动作类型':<10}{'样本数':<10}{'正确数':<10}{'准确率':<10}\n")
        for action_name, stats in action_stats.items():
            f.write(f"{action_name:<10}{stats['总数']:<10}{stats['正确数']:<10}{stats['准确率']:.4f}\n")
    
    print(f"\n测试完成!")
    print(f"总样本数: {action_total}")
    print(f"动作准确率: {action_accuracy:.4f}")
    print(f"吃牌样本数: {chi_total//2}")
    print(f"吃牌准确率: {chi_accuracy:.4f}")
    print(f"详细结果已保存至: {results_dir}")
    
    return model, action_accuracy, chi_accuracy, results_dir

# 主测试程序
if __name__ == "__main__":
    # 指定模型和测试数据
    results_dir = "models/action_models/"  # 您的训练结果目录
    test_dataset = MahjongActionDataset(data_folder="/home/luzhiwei/data/a/mahjong_data_test")
    test_model_path = os.path.join(results_dir, "training_results_20250302_194835/mahjong_action_request_best.pth")
    
    # 测试模型并获取结果
    model, action_acc, chi_acc, test_results_dir = test_action_request_model(test_model_path, test_dataset)
    
    print(f"\n测试和分析全部完成，请查看 {test_results_dir} 目录下的详细结果")