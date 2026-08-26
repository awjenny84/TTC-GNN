# -*- coding: utf-8 -*-
"""Extract style-prototype features from every ``text_raw`` in aigc_new.

Dependencies
------------
    pip install jieba snownlp

The output keeps all original columns and appends both the paper variable
names and aliases expected by the training scripts in this repository.
"""

import argparse
import csv
import math
import os
import re
from collections import Counter

try:
    import jieba
    from snownlp import SnowNLP
except ImportError as exc:
    raise SystemExit("Missing dependency. Run: pip install jieba snownlp") from exc


TEXT_COLUMN = "text_raw"
NUMERIC_RE = re.compile(r"^[\d.]+$")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
WEIBO_EMOJI_RE = re.compile(r"\[[^\[\]\s]{1,12}\]")
UNICODE_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # flags
    "\U0001F300-\U0001FAFF"  # pictographs and emoji
    "\u2600-\u27BF"          # miscellaneous symbols/dingbats
    "]",
    flags=re.UNICODE,
)
QUESTION_RE = re.compile(
    r"[?？]|(?:^|[，。！？；\s])(吗|么|呢|为何|为什么|怎么|怎样|如何|谁|什么|哪(?:个|些|里)?|几|多少)(?:[，。！？；\s]|$)"
)

PAPER_COLUMNS = [
    "length",
    "emoji_count",
    "is_qa",
    "sentiment_score",
    "emotional_polarity",
    "TTR",
    "RTTR",
    "MTLD",
    "MSTTR",
    "commonly_used_words_ratio",
    "stop_words_ratio",
]
COMPATIBILITY_COLUMNS = ["sentiment", "ttr", "rttr", "mtld", "msttr", "common_ratio", "stop_ratio"]


def find_aigc_csv(directory):
    matches = [
        name for name in os.listdir(directory)
        if name.lower().endswith(".csv") and "aigc_new" in name.lower()
    ]
    if not matches:
        raise FileNotFoundError("No CSV containing 'aigc_new' was found.")
    return os.path.join(directory, sorted(matches)[0])


def read_word_set(path):
    if not path or not os.path.exists(path):
        return set()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with open(path, "r", encoding=encoding) as file_obj:
                return {line.strip() for line in file_obj if line.strip()}
        except UnicodeDecodeError:
            continue
    raise UnicodeError("Cannot decode word-list file: {}".format(path))


def tokenize(text):
    """Jieba segmentation with URLs, whitespace and punctuation removed."""
    clean = URL_RE.sub(" ", text)
    tokens = []
    for token in jieba.lcut(clean):
        token = token.strip().lower()
        if not token or NUMERIC_RE.fullmatch(token):
            continue
        if not any("\u4e00" <= char <= "\u9fff" or char.isalpha() for char in token):
            continue
        tokens.append(token)
    return tokens


def ttr(tokens):
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def rttr(tokens):
    return len(set(tokens)) / math.sqrt(len(tokens)) if tokens else 0.0


def msttr(tokens, segment_length=50):
    if not tokens:
        return 0.0
    if len(tokens) < segment_length:
        return ttr(tokens)
    segments = [tokens[i:i + segment_length] for i in range(0, len(tokens), segment_length)]
    # Standard MSTTR discards an incomplete final segment.
    segments = [segment for segment in segments if len(segment) == segment_length]
    return sum(ttr(segment) for segment in segments) / len(segments)


def _mtld_direction(tokens, threshold=0.72):
    if not tokens:
        return 0.0
    factors = 0.0
    types = set()
    start = 0
    for index, token in enumerate(tokens):
        types.add(token)
        length = index - start + 1
        current_ttr = len(types) / length
        if current_ttr <= threshold:
            factors += 1.0
            types.clear()
            start = index + 1
    remainder = len(tokens) - start
    if remainder:
        remainder_ttr = len(types) / remainder
        factors += (1.0 - remainder_ttr) / (1.0 - threshold)
    return len(tokens) / factors if factors > 0 else float(len(tokens))


def mtld(tokens, threshold=0.72):
    if not tokens:
        return 0.0
    return (_mtld_direction(tokens, threshold) + _mtld_direction(list(reversed(tokens)), threshold)) / 2.0


def polarity_code(score):
    """Paper boundaries: negative < .4; neutral [.4, .6); positive >= .6."""
    if score < 0.4:
        return -1
    if score < 0.6:
        return 0
    return 1


def build_common_words(csv_path, stopwords, top_n):
    counts = Counter()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if TEXT_COLUMN not in (reader.fieldnames or []):
            raise ValueError("CSV is missing required column: {}".format(TEXT_COLUMN))
        for row in reader:
            counts.update(token for token in tokenize(row.get(TEXT_COLUMN, "")) if token not in stopwords)
    return {word for word, _ in counts.most_common(top_n)}


def extract_features(text, stopwords, common_words, segment_length):
    text = "" if text is None else str(text)
    tokens = tokenize(text)
    token_count = len(tokens)
    try:
        score = float(SnowNLP(text).sentiments) if text.strip() else 0.5
    except Exception:
        score = 0.5
    score = min(1.0, max(0.0, score))
    common_ratio = sum(token in common_words for token in tokens) / token_count if token_count else 0.0
    stop_ratio = sum(token in stopwords for token in tokens) / token_count if token_count else 0.0
    ttr_value = ttr(tokens)
    rttr_value = rttr(tokens)
    mtld_value = mtld(tokens)
    msttr_value = msttr(tokens, segment_length)
    result = {
        "length": len(text),
        "emoji_count": len(WEIBO_EMOJI_RE.findall(text)) + len(UNICODE_EMOJI_RE.findall(text)),
        "is_qa": int(bool(QUESTION_RE.search(text))),
        "sentiment_score": score,
        "emotional_polarity": polarity_code(score),
        "TTR": ttr_value,
        "RTTR": rttr_value,
        "MTLD": mtld_value,
        "MSTTR": msttr_value,
        "commonly_used_words_ratio": common_ratio,
        "stop_words_ratio": stop_ratio,
        # Names consumed by the repository's existing training scripts.
        "sentiment": polarity_code(score),
        "ttr": ttr_value,
        "rttr": rttr_value,
        "mtld": mtld_value,
        "msttr": msttr_value,
        "common_ratio": common_ratio,
        "stop_ratio": stop_ratio,
    }
    for key, value in result.items():
        if isinstance(value, float):
            result[key] = "{:.8f}".format(value)
    return result


def process_csv(input_path, output_path, stopwords, common_words, segment_length, progress_every):
    with open(input_path, "r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        original_columns = reader.fieldnames or []
        if TEXT_COLUMN not in original_columns:
            raise ValueError("CSV is missing required column: {}".format(TEXT_COLUMN))
        appended = [name for name in PAPER_COLUMNS + COMPATIBILITY_COLUMNS if name not in original_columns]
        with open(output_path, "w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=original_columns + appended)
            writer.writeheader()
            for number, row in enumerate(reader, start=1):
                row.update(extract_features(row.get(TEXT_COLUMN, ""), stopwords, common_words, segment_length))
                writer.writerow(row)
                if progress_every > 0 and number % progress_every == 0:
                    print("Processed {:,} rows...".format(number), flush=True)
    return number if 'number' in locals() else 0


def parse_args():
    parser = argparse.ArgumentParser(description="Extract paper-style features from aigc_new text_raw.")
    parser.add_argument("--input", default="", help="Input CSV; default: auto-detect aigc_new*.CSV.")
    parser.add_argument("--output", default="aigc_new_with_style_features.csv")
    parser.add_argument("--stopwords", default="stopwords.txt")
    parser.add_argument("--common-words", default="", help="Optional one-word-per-line common-word list.")
    parser.add_argument("--common-top-n", type=int, default=1000,
                        help="If no common-word list is given, derive this many frequent corpus words.")
    parser.add_argument("--msttr-segment-length", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(args.input) if args.input else find_aigc_csv(base_dir)
    output_path = os.path.abspath(args.output)
    stopword_path = args.stopwords if os.path.isabs(args.stopwords) else os.path.join(base_dir, args.stopwords)
    stopwords = read_word_set(stopword_path)

    if args.common_words:
        common_path = args.common_words if os.path.isabs(args.common_words) else os.path.join(base_dir, args.common_words)
        common_words = read_word_set(common_path)
        print("Loaded {} common words from {}.".format(len(common_words), common_path))
    else:
        print("Building top-{} common-word list from the corpus...".format(args.common_top_n))
        common_words = build_common_words(input_path, stopwords, args.common_top_n)
        print("Built {} common words.".format(len(common_words)))

    count = process_csv(
        input_path, output_path, stopwords, common_words,
        args.msttr_segment_length, args.progress_every,
    )
    print("Done: wrote {:,} rows to {}".format(count, output_path))


if __name__ == "__main__":
    main()
