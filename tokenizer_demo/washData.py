import pandas as pd
import re
import glob
from tqdm import tqdm


def clean_text(text):

    if not isinstance(text, str):
        return None

    # 去换行
    text = text.replace("\n", " ")

    # 删除URL
    text = re.sub(r"http\S+", "", text)

    # 删除新闻来源 (CNN) (BBC)
    text = re.sub(r"\([A-Z]{2,}\)", "", text)

    # 删除破折号
    text = text.replace("--", " ")

    # 删除多余空格
    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    # 删除过短文本
    if len(text) < 50:
        return None

    # unicode清理
    text = text.encode("utf-8", "ignore").decode()

    return text



files = sorted(glob.glob("train-*.parquet"))

print("Found files:", files)

out = open("tokenizer/corpus.txt", "w", encoding="utf-8")

for f in files:

    print("Processing:", f)

    df = pd.read_parquet(f)

    for text in tqdm(df["text"]):

        clean = clean_text(text)

        if clean:
            out.write(clean + "\n")

out.close()

print("Cleaning finished!")