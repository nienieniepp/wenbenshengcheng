import argparse
from pathlib import Path

import evaluate
import matplotlib.pyplot as plt
import pandas as pd

from inference import summarize_text
from utils import ensure_dir


def compute_rouge(predictions, references):
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline and transformer summaries.")
    parser.add_argument("--test_csv", type=str, default="data/processed/test.csv")
    parser.add_argument("--baseline_csv", type=str, default="results/baseline_predictions.csv")
    parser.add_argument("--output_csv", type=str, default="results/rouge_scores.csv")
    parser.add_argument("--figure_path", type=str, default="results/figures/rouge_comparison.png")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv).fillna("").head(args.max_samples)
    baseline_df = pd.read_csv(args.baseline_csv).fillna("").head(args.max_samples)

    references = test_df["highlights"].tolist()
    baseline_preds = baseline_df["baseline_summary"].tolist()
    transformer_preds = [summarize_text(a) for a in test_df["article"].tolist()]

    baseline_scores = compute_rouge(baseline_preds, references)
    transformer_scores = compute_rouge(transformer_preds, references)

    result_df = pd.DataFrame(
        [
            {"model": "baseline_tfidf", "rouge1": baseline_scores["rouge1"], "rouge2": baseline_scores["rouge2"], "rougeL": baseline_scores["rougeL"]},
            {"model": "t5_transformer", "rouge1": transformer_scores["rouge1"], "rouge2": transformer_scores["rouge2"], "rougeL": transformer_scores["rougeL"]},
        ]
    )

    ensure_dir(str(Path(args.output_csv).parent))
    ensure_dir(str(Path(args.figure_path).parent))
    result_df.to_csv(args.output_csv, index=False, encoding="utf-8")

    print("\n=== ROUGE Evaluation Results ===")
    print(result_df)

    metrics = ["rouge1", "rouge2", "rougeL"]
    plot_df = result_df.set_index("model")[metrics]
    ax = plot_df.plot(kind="bar", figsize=(8, 5))
    ax.set_title("ROUGE Score Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(args.figure_path, dpi=200)
    print(f"Saved figure to: {args.figure_path}")


if __name__ == "__main__":
    main()
