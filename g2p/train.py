"""Train the Khmer G2P model on g2p/data/*.tsv (run prepare_data.py first).

Usage: python g2p/train.py [--epochs 80] [--batch 256]
Saves the best checkpoint (lowest validation word error rate) to g2p/checkpoints/best.pt.
"""
import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from model import BOS, EOS, PAD, TASKS, G2PModel, Vocab, pad_batch, source_tokens

HERE = Path(__file__).resolve().parent


def use_bf16(device: str) -> bool:
    """bfloat16 autocast only on GPUs with native support (not e.g. Colab's T4)."""
    return device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)


def load(split: str) -> pd.DataFrame:
    return pd.read_csv(HERE / f"data/{split}.tsv", sep="\t", dtype=str, keep_default_na=False)


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def predict(model, src_vocab, tgt_vocab, pairs, device, batch=512) -> list[list[str]]:
    """pairs: list of (task, word) -> predicted token lists."""
    model.eval()
    out = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        src = pad_batch([src_vocab.encode(source_tokens(t, w)) for t, w in chunk]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16(device)):
            ids = model.greedy(src)
        out += [tgt_vocab.decode(s) for s in ids]
    return out


def evaluate(model, src_vocab, tgt_vocab, df, device) -> dict:
    """Word error rate and phoneme error rate per task. A word counts as correct
    if the prediction matches any of its reference readings."""
    refs = defaultdict(list)
    for t, w, y in zip(df.task, df.word, df.target):
        refs[(t, w)].append(y.split(" "))
    keys = list(refs)
    preds = predict(model, src_vocab, tgt_vocab, keys, device)
    stats = defaultdict(lambda: [0, 0, 0, 0])  # words, wrong words, edits, ref tokens
    for (t, w), p in zip(keys, preds):
        best = min(refs[(t, w)], key=lambda r: edit_distance(p, r))
        e = edit_distance(p, best)
        s = stats[t]
        s[0] += 1; s[1] += e > 0; s[2] += e; s[3] += len(best)
    return {t: {"words": s[0], "wer": s[1] / s[0], "per": s[2] / s[3]} for t, s in stats.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=4, help="evaluations without improvement")
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load("train"), load("validation")

    src_vocab = Vocab([f"<{t}>" for t in TASKS] + [c for w in train.word for c in w])
    tgt_vocab = Vocab([tok for y in train.target for tok in y.split(" ")])
    data = [(src_vocab.encode(source_tokens(t, w)), tgt_vocab.encode(y.split(" ")))
            for t, w, y in zip(train.task, train.word, train.target)]
    print(f"train={len(data)} val={len(val)} src_vocab={len(src_vocab)} tgt_vocab={len(tgt_vocab)} device={device}")

    model = G2PModel(len(src_vocab), len(tgt_vocab)).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    steps = args.epochs * (len(data) // args.batch + 1)
    warmup = 2000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, max(0.05, (steps - s) / (steps - warmup))))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    ckpt = HERE / "checkpoints/best.pt"
    ckpt.parent.mkdir(exist_ok=True)
    best, bad = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(data)
        t0, total = time.time(), 0.0
        for i in range(0, len(data), args.batch):
            chunk = data[i:i + args.batch]
            src = pad_batch([s for s, _ in chunk]).to(device)
            tgt = pad_batch([[BOS] + y + [EOS] for _, y in chunk]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16(device)):
                logits = model(src, tgt[:, :-1])
                loss = loss_fn(logits.float().reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total += loss.item() * len(chunk)
        msg = f"epoch {epoch:3d} loss {total / len(data):.4f} ({time.time() - t0:.0f}s)"

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m = evaluate(model, src_vocab, tgt_vocab, val, device)
            score = sum(v["wer"] for v in m.values()) / len(m)
            msg += "  val " + "  ".join(f"{t}: WER {v['wer']:.2%} PER {v['per']:.2%}" for t, v in m.items())
            if score < best:
                best, bad = score, 0
                torch.save({"config": model.config, "state_dict": model.state_dict(),
                            "src_vocab": src_vocab.itos, "tgt_vocab": tgt_vocab.itos}, ckpt)
                msg += "  *saved"
            else:
                bad += 1
        print(msg, flush=True)
        if bad >= args.patience:
            print("early stop")
            break

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    test = evaluate(model, src_vocab, tgt_vocab, load("test"), device)
    for t, v in test.items():
        print(f"TEST {t}: words={v['words']} WER {v['wer']:.2%} PER {v['per']:.2%}")


if __name__ == "__main__":
    main()
