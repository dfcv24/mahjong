import time
import argparse
from collections import defaultdict
from src.client.client import MahjongClient

def run_auto_games(client, num_games=100, player_name="AI_Player", delay_between_games=2):
    """
    自动运行多场对局并统计胜率
    
    参数:
    client: MahjongClient实例
    num_games: 要运行的游戏场数
    player_name: 客户端玩家名称
    delay_between_games: 游戏间隔时间(秒)
    """
    print(f"开始自动对局测试。计划进行 {num_games} 场游戏。")
    print(f"玩家名称: {player_name}")
    
    # 初始化统计数据
    stats = {
        "games_played": 0,
        "games_completed": 0,
        "wins": 0,
        "total_points": 0,
        "game_results": []
    }
    
    try:
        for game_num in range(1, num_games + 1):
            print(f"\n===== 开始第 {game_num}/{num_games} 场游戏 =====")
            
            # 向服务器发送开始游戏请求
            client.send_play_game(player_name)
            print(f"已发送游戏请求，等待游戏结束...")
            
            # 等待游戏结束
            # 注意：客户端在接收消息时会处理游戏逻辑，这里只需等待
            timeout = 600  # 10分钟超时
            start_time = time.time()
            
            # 清除之前的游戏结果
            client.last_game_result = None
            client.game_completed = False
            client.current_game_score = 0
            
            # 等待游戏完成
            while time.time() - start_time < timeout:
                if client.game_completed:
                    break
                time.sleep(1)
            
            if client.game_completed and client.last_game_result:
                stats["games_completed"] += 1
                game_result = client.last_game_result
                stats["game_results"].append(game_result)
                
                # 获取我方玩家的得分
                my_score = game_result.get("scores", 0)
                stats["total_points"] += my_score
                
                # 判断是否获胜
                if game_result.get("winner") == "玩家0获胜":
                    stats["wins"] += 1
                
                # 打印当前游戏结果
                print(f"游戏 {game_num} 结果:")
                print(f"  胜者: {game_result.get('winner', '无')}")
                print(f"  我方得分: {my_score}")
                
                # # 打印所有玩家得分
                # for p, score in game_result.get("scores", {}).items():
                #     print(f"  {p}: {score} 点")
                
                # 打印当前统计
                completed = stats["games_completed"]
                wins = stats["wins"]
                win_rate = (wins / completed * 100) if completed > 0 else 0
                avg_points = stats["total_points"] / completed if completed > 0 else 0
                
                print(f"\n当前统计 ({completed}/{game_num} 场完成):")
                print(f"  胜场: {wins}, 胜率: {win_rate:.1f}%, 平均得分: {avg_points:.1f}")
            else:
                print(f"游戏 {game_num} 未能正常完成或超时")
            
            stats["games_played"] += 1
            
            # 游戏间延迟
            if game_num < num_games:
                print(f"等待 {delay_between_games} 秒后开始下一场...")
                time.sleep(delay_between_games)
    
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 打印最终统计结果
        completed = stats["games_completed"]
        wins = stats["wins"]
        win_rate = (wins / completed * 100) if completed > 0 else 0
        avg_points = stats["total_points"] / completed if completed > 0 else 0
        
        print("\n===== 最终统计 =====")
        print(f"总场数: {stats['games_played']}, 完成场数: {completed}")
        print(f"胜场: {wins}, 胜率: {win_rate:.1f}%")
        
        # 将统计结果写入文件
        save_stats_to_file(stats, player_name)
    
    return stats

def save_stats_to_file(stats, player_name):
    """将统计结果保存到文件"""
    import datetime
    import json
    import os
    
    # 创建结果目录
    results_dir = "results/game_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"game_stats_{timestamp}.txt"
    filepath = os.path.join(results_dir, filename)
    
    # 保存报告
    with open(filepath, "w", encoding="utf-8") as f:
        completed = stats["games_completed"]
        wins = stats["wins"]
        win_rate = (wins / completed * 100) if completed > 0 else 0
        avg_points = stats["total_points"] / completed if completed > 0 else 0
        
        f.write(f"麻将游戏自动对局统计报告\n")
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"玩家名称: {player_name}\n")
        f.write(f"总场数: {stats['games_played']}, 完成场数: {completed}\n")
        f.write(f"胜场: {wins}, 胜率: {win_rate:.1f}%, 平均得分: {avg_points:.1f}\n\n")
        
        f.write("各场游戏详情:\n")
        for i, result in enumerate(stats["game_results"], 1):
            f.write(f"游戏 {i}:\n")
            f.write(f"  胜者: {result.get('winner', '无')}\n")
            f.write(f"  我方得分: {result.get('scores', 0)}\n")
            f.write("\n")
    
    print(f"\n详细统计报告已保存至: {filepath}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="麻将自动对局测试工具")
    parser.add_argument("--auto", action="store_true", help="开启自动对局模式")
    parser.add_argument("--games", type=int, default=100, help="自动模式下的游戏场数")
    parser.add_argument("--player", type=str, default="AI_Player", help="玩家名称")
    parser.add_argument("--delay", type=float, default=2, help="游戏间延迟时间(秒)")
    
    args = parser.parse_args()
    
    client = MahjongClient()
    
    if client.connect_to_server():
        print("麻将客户端启动")
        
        try:
            client.load_ai_models(model_num=1)
            if args.auto:
                # 自动对局模式
                run_auto_games(client, args.games, args.player, args.delay)
            else:
                # 交互模式
                print("输入 'p' 开始游戏, 'a' 开始自动对局测试, 'q' 退出")
                
                while True:
                    cmd = input("> ")
                    if cmd.lower() == "q":
                        break
                    elif cmd.lower() == "p":
                        player_name = input("输入玩家名称: ")
                        client.send_message({"cmd": "play", "player_name": player_name})
                    elif cmd.lower() == "a":
                        num_games = int(input("输入要进行的游戏场数: "))
                        player_name = input("输入玩家名称 (留空使用默认名称): ")
                        if not player_name.strip():
                            player_name = "AI_Player"
                        run_auto_games(client, num_games, player_name)
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            client.close()
    else:
        print("无法连接到服务器，程序退出")

if __name__ == "__main__":
    main()