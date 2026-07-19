import re
from collections import Counter


WORD_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)


def analyze_text(text: str) -> dict:
    words = WORD_PATTERN.findall(text.lower())
    top_words = [
        {"word": word, "count": count}
        for word, count in Counter(words).most_common(10)
    ]

    return {
        "word_count": len(words),
        "character_count": len(text),
        "line_count": text.count("\n") + 1 if text else 0,
        "top_words": top_words,
    }
