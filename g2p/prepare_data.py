"""Build the G2P training data from phonsobon/khmer-G2P (cleaned NCKL pairs):
NCKL Khmer-script respelling, task `respell`
    ក្របី -> ក្រ-បី

Splits are kept from the Hub.

Writes g2p/data/{train,validation,test}.tsv with columns: task, word, target
(target tokens separated by spaces).
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT = Path(__file__).resolve().parent / "data"


def main() -> None:
    nckl = load_dataset("phonsobon/khmer-G2P")
    rows = []
    for split, ds in nckl.items():
        for r in ds:
            rows.append(("respell", r["word"], " ".join(r["pronunciation"]), split))

    df = pd.DataFrame(rows, columns=["task", "word", "target", "split"])
    OUT.mkdir(exist_ok=True)
    for split, part in df.groupby("split"):
        part.drop(columns="split").to_csv(OUT / f"{split}.tsv", sep="\t", index=False)
    print(df.groupby(["split", "task"]).size().unstack())
    print("unique words:", df["word"].nunique())


if __name__ == "__main__":
    main()
