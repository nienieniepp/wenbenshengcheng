# 4-Page Report Outline (COMM7340)

## 1. Introduction
- Background: information overload in digital news media.
- Why abstractive summarization matters for user efficiency.
- Brief problem statement and scope.

## 2. Objectives
- Build an end-to-end automatic news summarization pipeline.
- Compare a simple extractive baseline vs a Transformer abstractive model.
- Evaluate with ROUGE metrics and provide interpretable findings.

## 3. Proposed Methods
### 3.1 Dataset
- CNN/DailyMail (Hugging Face, version 3.0.0).
- Automatic download and subset strategy for lightweight training.

### 3.2 Preprocessing
- Text cleaning and whitespace normalization.
- Input formatting for T5: `summarize: <article>`.
- Length control for computational feasibility.

### 3.3 Baseline: TF-IDF Extractive Summarization
- Sentence splitting.
- TF-IDF vectorization and sentence scoring.
- Top-K sentence selection.

### 3.4 Transformer Model: T5-small Fine-tuning
- Tokenization settings (source/target max lengths).
- Training setup (epochs, batch size, checkpointing).
- Best model selection and inference pipeline.

## 4. Experiments
- Experimental environment (hardware/software).
- Data split sizes used in this course project.
- Hyperparameters and runtime constraints.
- Evaluation protocol and reproducibility notes.

## 5. Findings and Results
- ROUGE-1 / ROUGE-2 / ROUGE-L comparison table.
- Qualitative examples of generated summaries.
- Visualization (bar chart) and interpretation.

## 6. Discussion
- Strengths and limitations of TF-IDF baseline.
- Strengths and failure cases of T5 model.
- Impact of training size and compute constraints.
- Potential improvements (BART, larger subset, human evaluation).

## 7. Conclusion
- Summary of achievements and key takeaways.
- Applicability in digital media workflows.

## 8. Contributions of Each Group Member
- Member A: data pipeline and preprocessing.
- Member B: baseline model implementation.
- Member C: Transformer training and tuning.
- Member D: evaluation and visualization.
- Member E: Flask demo development.
- Member F: integration, testing, report/slides engineering support.
