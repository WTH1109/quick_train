import os
import re
import pandas as pd
from pathlib import Path


def parse_experiment_name(folder_name):
    """解析实验文件夹名称中的参数"""
    params = {
        "model": "Unknown",
        "lr": None,
        "dropout": None,
        "batch_size": None
    }

    # 使用正则表达式匹配参数
    model_match = re.search(r'M([a-zA-Z0-9]+)_', folder_name)
    if model_match:
        params["model"] = model_match.group(1).lower()

    lr_match = re.search(r'_lr([\de-]+)_', folder_name)
    if lr_match:
        lr_str = lr_match.group(1).replace('e-', 'e-')
        params["lr"] = float(lr_str)

    bs_match = re.search(r'_bc(\d+)', folder_name)
    if bs_match:
        params["batch_size"] = int(bs_match.group(1))

    drop_match = re.search(r'_drop([\d.]+)', folder_name)
    if drop_match:
        params["dropout"] = float(drop_match.group(1)) / 10  # 假设 drop04 表示 0.4

    return params


def parse_log_file(log_path):
    """解析日志文件，提取最后一个best epoch信息"""
    best_epoch = None
    current_epoch = None

    with open(log_path, 'r') as f:
        for line in f:
            # 匹配最佳epoch行
            if line.startswith("Best epoch:"):
                match = re.search(
                    r'Best epoch:(\d+).+?Val Acc=([\d.]+).+?Sen=([\d.]+).+?Spec=([\d.]+).+?F1=([\d.]+).+?AUC=([\d.]+)',
                    line
                )
                if match:
                    best_epoch = {
                        "epoch": int(match.group(1)),
                        "val_acc": float(match.group(2)),
                        "sen": float(match.group(3)),
                        "spec": float(match.group(4)),
                        "f1": float(match.group(5)),
                        "auc": float(match.group(6))
                    }
    return best_epoch


def main(experiment_dir, output_file):
    """主处理函数"""
    results = []

    # 遍历实验目录
    for exp_dir in Path(experiment_dir).iterdir():
        if not exp_dir.is_dir():
            continue

        print(f"Processing: {exp_dir.name}")

        # 解析实验参数
        params = parse_experiment_name(exp_dir.name)

        # 查找日志文件
        log_path = exp_dir / "logstream" / "stream.txt"
        if not log_path.exists():
            print(f"Warning: Log file not found in {exp_dir.name}")
            continue

        # 解析日志
        best_epoch = parse_log_file(log_path)
        if not best_epoch:
            print(f"Warning: No best epoch found in {exp_dir.name}")
            continue

        # 合并结果
        record = {
            **params,
            **best_epoch
        }
        results.append(record)

    # 生成DataFrame
    df = pd.DataFrame(results)

    # 调整列顺序
    columns = ["model", "lr", "dropout", "batch_size", "epoch",
               "val_acc", "sen", "spec", "f1", "auc"]
    df = df[columns]

    # 保存Excel
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # 配置路径
    experiment_root = "../../logs/ResNet_tongue_seg"
    save_dir = "./analyse_result"
    os.makedirs(save_dir, exist_ok=True)
    output_excel = os.path.join(save_dir, "experiment_results.xlsx")

    main(experiment_root, output_excel)
