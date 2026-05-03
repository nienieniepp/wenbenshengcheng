from flask import Flask, render_template, request

from src.inference import load_model_and_tokenizer, summarize_text

app = Flask(__name__)

# 提前加载模型，避免每次请求重复初始化
TOKENIZER, MODEL, DEVICE, MODEL_NAME = load_model_and_tokenizer()


@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    article = ""
    error = ""
    original_wc = 0
    summary_wc = 0

    if request.method == "POST":
        try:
            article = request.form.get("article", "").strip()
            if not article:
                raise ValueError("请输入新闻正文后再生成摘要。")

            summary = summarize_text(article)
            original_wc = len(article.split())
            summary_wc = len(summary.split())
        except Exception as exc:
            error = f"生成摘要时发生错误：{exc}"

    return render_template(
        "index.html",
        summary=summary,
        article=article,
        error=error,
        model_name=MODEL_NAME,
        original_wc=original_wc,
        summary_wc=summary_wc,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
