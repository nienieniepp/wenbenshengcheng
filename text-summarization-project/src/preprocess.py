import re
from typing import Optional


def clean_text(text: str) -> str:
    """清洗文本：移除多余空白并保留基本可读性。"""
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(text: str, max_words: int = 512) -> str:
    """按词数安全截断，避免输入过长导致训练或推理显存压力过大。"""
    cleaned = clean_text(text)
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words])


def prepare_t5_input(article: str, max_words: Optional[int] = 512) -> str:
    """为 T5 构造输入格式：'summarize: ' + article。"""
    if max_words is None:
        article_processed = clean_text(article)
    else:
        article_processed = truncate_text(article, max_words=max_words)
    return f"summarize: {article_processed}"
