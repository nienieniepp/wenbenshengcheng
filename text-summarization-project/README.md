# Transformer-based Abstractive Text Summarization System using CNN/DailyMail Dataset

## 1. 项目背景
在数字媒体场景中，新闻内容体量大、更新快，人工阅读与整理成本高。自动文本摘要能够帮助用户快速获取核心信息。本项目面向 COMM7340 课程，构建一个“可训练 + 可评估 + 可演示”的新闻摘要系统。

## 2. 项目目标
- 输入：英文长新闻文章。
- 输出：简洁摘要。
- 比较两类方法：
  1) TF-IDF 抽取式基线；
  2) Transformer（T5）生成式摘要。

## 3. 数据集介绍
- 数据集：Hugging Face `cnn_dailymail`
- 配置：`3.0.0`
- 自动下载方式：
```python
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
```
- 为了课程演示，默认使用子集（可命令行调整）。

## 4. 方法说明
- **基线模型**：TF-IDF 句子打分，选 Top-K 句拼接为摘要。
- **深度学习模型**：`t5-small` 微调，输入前缀使用 `summarize: `。
- **评估指标**：ROUGE-1、ROUGE-2、ROUGE-L。
- **可视化**：输出 ROUGE 对比柱状图。

## 5. 项目结构
见课程要求目录，代码位于 `src/`，Web Demo 位于 `app/`，报告与展示提纲分别位于 `report/` 与 `presentation/`。

## 6. 安装
```bash
pip install -r requirements.txt
```

## 7. 数据下载与子集构建
```bash
python src/download_dataset.py --train_size 2000 --val_size 500 --test_size 500
```

## 8. 运行基线模型
```bash
python src/baseline_tfidf.py
```

## 9. 训练 Transformer 模型
```bash
python src/train_transformer.py
```

## 10. 评估模型
```bash
python src/evaluate.py
```

## 11. 运行 Flask Web Demo
```bash
python app/app.py
```

## 12. 预期结果
- 基线模型可快速生成可读摘要，但语义连贯性有限。
- T5 生成式摘要通常在 ROUGE 指标与可读性上优于基线。
- 输出文件包括：
  - `results/baseline_predictions.csv`
  - `results/rouge_scores.csv`
  - `results/figures/rouge_comparison.png`

## 13. 小组成员贡献表
详见 `report/contribution_table.md`，6 位成员均包含编码任务。

## 14. 课程要求对齐说明
- Python 项目：是。
- Deep Learning + NLP：是（Transformer + 文本摘要）。
- 明确目标：是（新闻自动摘要）。
- 结果展示：是（ROUGE + 可视化 + Web Demo）。
- 报告与展示材料：已提供提纲模板。
- 每位成员代码贡献：已在分工表中明确。
