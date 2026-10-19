"""Build the G2P training data from two sources, one output format each.

- phonsobon/khmer-dictionary (g2p config): NCKL Khmer-script respelling, task `respell`
    ក្របី -> ក្រ-បី
- seanghay/google-khmer-lexicon: Google phoneme transcription, task `ipa`
    ក្របី -> k r ɑ . ɓ ə j

Splits are assigned per word, so a word is in the same split for both tasks.
NCKL words keep the split from the Hub; other words are hashed into 90/5/5.

Writes g2p/data/{train,validation,test}.tsv with columns: task, word, target
(target tokens separated by spaces).
"""
import hashlib
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT = Path(__file__).resolve().parent / "data"
KHMER_WORD = r"[ក-៿‌‍]+"


def hash_split(word: str) -> str:
    bucket = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def main() -> None:
    nckl = load_dataset("phonsobon/khmer-dictionary", "g2p")
    rows, word_split = [], {}
    for split, ds in nckl.items():
        for r in ds:
            word_split[r["word"]] = split
            rows.append(("respell", r["word"], " ".join(r["pronunciation"]), split))

    google = load_dataset("seanghay/google-khmer-lexicon", split="train").to_pandas()
    google = google.dropna()
    google["word"] = google["word"].str.strip()
    google["pro"] = google["pro"].str.split().str.join(" ")
    google = google[google["word"].str.fullmatch(KHMER_WORD)].drop_duplicates(["word", "pro"])
    for w, p in zip(google["word"], google["pro"]):
        rows.append(("ipa", w, p, word_split.get(w) or hash_split(w)))

    df = pd.DataFrame(rows, columns=["task", "word", "target", "split"])
    OUT.mkdir(exist_ok=True)
    for split, part in df.groupby("split"):
        part.drop(columns="split").to_csv(OUT / f"{split}.tsv", sep="\t", index=False)
    print(df.groupby(["split", "task"]).size().unstack())
    print("unique words:", df["word"].nunique())


if __name__ == "__main__":
    main()
