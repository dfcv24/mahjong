#!/usr/bin/env python

import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="批量训练PPO麻将AI")
    parser.add_argument("--runs", type=int, default=1, help="训练运行次数")
    parser.add_argument("--base_model", type=str, default=None, help="基础预训练模型路径")
    parser.add_argument("--timesteps", type=int, default=100000, help="每次运行的时间步数")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    args = parser.parse_args()
    
    # 确保模型目录存在
    os.makedirs("results/ppo_training", exist_ok=True)
    
    # 执行多次训练
    for i in range(args.runs):
        print(f"开始第 {i+1}/{args.runs} 次训练...")
        
        # 构建训练命令
        cmd = ["python", "train_ppo.py", "--timesteps", str(args.timesteps), "--lr", str(args.lr)]
        
        # 如果是第一次运行，使用base_model
        if i == 0 and args.base_model:
            cmd.extend(["--pretrained_model", args.base_model])
        # 如果不是第一次运行，使用上一次训练的模型
        elif i > 0:
            # 找到最新的PPO模型
            ppo_dirs = [d for d in os.listdir("results/ppo_training") if d.startswith("ppo_")]
            if ppo_dirs:
                latest_dir = max(ppo_dirs)
                latest_model = os.path.join("results/ppo_training", latest_dir, "mahjong_ppo_model.zip")
                if os.path.exists(latest_model):
                    cmd.extend(["--ppo_model", latest_model])
        
        # 执行训练命令
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
        print(f"完成第 {i+1}/{args.runs} 次训练")

if __name__ == "__main__":
    main()
