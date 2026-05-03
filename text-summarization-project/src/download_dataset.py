import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from preprocess import clean_text


def save_split_to_csv(split, save_path: Path):
    """保存指定 split 到 CSV，并保证只包含 article/highlights 两列。"""
    records = []
    for item in split:
        article = clean_text(item.get("article", ""))
        highlights = clean_text(item.get("highlights", ""))
        records.append({"article": article, "highlights": highlights})
    pd.DataFrame(records, columns=["article", "highlights"]).to_csv(save_path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Download and save CNN/DailyMail subset.")
    parser.add_argument("--train_size", type=int, default=2000, help="Number of training samples.")
    parser.add_argument("--val_size", type=int, default=500, help="Number of validation samples.")
    parser.add_argument("--test_size", type=int, default=500, help="Number of test samples.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for CSV files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 自动从 Hugging Face 下载数据集（无需手动下载）
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    train_split = dataset["train"].select(range(min(args.train_size, len(dataset["train"]))))
    val_split = dataset["validation"].select(range(min(args.val_size, len(dataset["validation"]))))
    test_split = dataset["test"].select(range(min(args.test_size, len(dataset["test"]))))

    save_split_to_csv(train_split, output_dir / "train.csv")
    save_split_to_csv(val_split, output_dir / "validation.csv")
    save_split_to_csv(test_split, output_dir / "test.csv")

    print(f"Saved train/validation/test CSV files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
