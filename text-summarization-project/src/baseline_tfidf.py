import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import clean_text
from utils import ensure_dir


def split_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text(text)) if s.strip()]


def tfidf_extractive_summary(article: str, top_k: int = 3) -> str:
    sentences = split_sentences(article)
    if len(sentences) <= top_k:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores = tfidf_matrix.sum(axis=1).A1

    top_indices = scores.argsort()[::-1][:top_k]
    top_indices = sorted(top_indices)
    summary = " ".join([sentences[i] for i in top_indices])
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate TF-IDF baseline summaries.")
    parser.add_argument("--test_csv", type=str, default="data/processed/test.csv")
    parser.add_argument("--output_csv", type=str, default="results/baseline_predictions.csv")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    test_path = Path(args.test_csv)
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")

    df = pd.read_csv(test_path)
    if "article" not in df.columns or "highlights" not in df.columns:
        raise ValueError("Input CSV must contain 'article' and 'highlights' columns.")

    predictions = [tfidf_extractive_summary(article, top_k=args.top_k) for article in df["article"].fillna("")]

    output_df = pd.DataFrame(
        {
            "article": df["article"],
            "reference": df["highlights"],
            "baseline_summary": predictions,
        }
    )

    ensure_dir(str(Path(args.output_csv).parent))
    output_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Baseline predictions saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
