from pathlib import Path


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def word_count(text: str) -> int:
    return len(text.split()) if isinstance(text, str) else 0
