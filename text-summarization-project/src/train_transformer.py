import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from preprocess import clean_text, prepare_t5_input
from utils import ensure_dir


def load_csv_as_dataset(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path).fillna("")
    if "article" not in df.columns or "highlights" not in df.columns:
        raise ValueError(f"CSV missing required columns: {csv_path}")
    return Dataset.from_pandas(df[["article", "highlights"]])


def main():
    parser = argparse.ArgumentParser(description="Fine-tune t5-small for summarization.")
    parser.add_argument("--train_csv", type=str, default="data/processed/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/processed/validation.csv")
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--output_dir", type=str, default="models/t5_summarizer")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--source_max_len", type=int, default=512)
    parser.add_argument("--target_max_len", type=int, default=128)
    args = parser.parse_args()

    # 中文注释：加载分词器与预训练模型，准备进行微调
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 中文注释：读取训练集和验证集 CSV，转换为 Hugging Face Dataset
    train_ds = load_csv_as_dataset(args.train_csv)
    val_ds = load_csv_as_dataset(args.val_csv)

    # 中文注释：定义预处理函数，构造 T5 输入与目标摘要
    def preprocess_batch(batch):
        inputs = [prepare_t5_input(clean_text(a), max_words=args.source_max_len) for a in batch["article"]]
        targets = [clean_text(h) for h in batch["highlights"]]

        model_inputs = tokenizer(
            inputs,
            max_length=args.source_max_len,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=targets,
            max_length=args.target_max_len,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(preprocess_batch, batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(preprocess_batch, batched=True, remove_columns=val_ds.column_names)

    ensure_dir(args.output_dir)

    # 中文注释：配置训练参数，保持轻量级适合课程演示
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        report_to="none",
    )

    # 中文注释：创建 Trainer 并开始训练
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # 中文注释：保存最佳模型与分词器，供推理和 Web Demo 使用
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Model saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
