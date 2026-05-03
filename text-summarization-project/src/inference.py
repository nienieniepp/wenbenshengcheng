import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from preprocess import prepare_t5_input

DEFAULT_MODEL_DIR = Path("models/t5_summarizer")
DEFAULT_PRETRAINED = "t5-small"


def load_model_and_tokenizer(model_dir: Path = DEFAULT_MODEL_DIR):
    if model_dir.exists() and any(model_dir.iterdir()):
        model_name = str(model_dir)
    else:
        model_name = DEFAULT_PRETRAINED

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device, model_name


def summarize_text(text: str, max_input_words: int = 512, max_output_len: int = 128) -> str:
    tokenizer, model, device, _ = load_model_and_tokenizer()
    prompt = prepare_t5_input(text, max_words=max_input_words)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_output_len,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run summarization inference.")
    parser.add_argument("--text", type=str, required=True, help="Input article text.")
    args = parser.parse_args()

    summary = summarize_text(args.text)
    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":
    main()
