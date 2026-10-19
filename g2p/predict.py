"""Use the trained Khmer G2P model.

    from predict import KhmerG2P
    g2p = KhmerG2P()
    g2p("ក្របី")                    # 'ក្រ-បី'           (NCKL Khmer-script respelling)
    g2p(["សម្ដេច", "កម្ពុជា"])       # list in, list out

CLI: python g2p/predict.py ក្របី កម្ពុជា
"""
import argparse
import re
from pathlib import Path

import torch

from model import TASKS, G2PModel, Vocab, format_output, pad_batch, source_tokens

DEFAULT_CKPT = Path(__file__).resolve().parent / "checkpoints/best.pt"
NON_KHMER = re.compile(r"[^ក-៿‌‍]+")


class KhmerG2P:
    def __init__(self, checkpoint: str | Path = DEFAULT_CKPT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint, map_location=self.device)
        self.src_vocab, self.tgt_vocab = Vocab([]), Vocab([])
        self.src_vocab.itos, self.tgt_vocab.itos = state["src_vocab"], state["tgt_vocab"]
        self.src_vocab.stoi = {t: i for i, t in enumerate(state["src_vocab"])}
        self.tgt_vocab.stoi = {t: i for i, t in enumerate(state["tgt_vocab"])}
        self.model = G2PModel(**state["config"]).to(self.device).eval()
        self.model.load_state_dict(state["state_dict"])

    def __call__(self, words: str | list[str], task: str = "respell") -> str | list[str]:
        single = isinstance(words, str)
        batch = [NON_KHMER.sub("", w) for w in ([words] if single else words)]
        src = pad_batch([self.src_vocab.encode(source_tokens(task, w)) for w in batch]).to(self.device)
        ids = self.model.greedy(src)
        out = [format_output(task, self.tgt_vocab.decode(s)) if w else "" for w, s in zip(batch, ids)]
        return out[0] if single else out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="+")
    ap.add_argument("--task", choices=TASKS, default="respell")
    args = ap.parse_args()
    for w, p in zip(args.words, KhmerG2P()(args.words, task=args.task)):
        print(f"{w}\t{p}")
