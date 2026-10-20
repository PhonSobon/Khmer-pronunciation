"""Plot training curves from g2p/checkpoints/history.json (written by train.py).

Usage: python g2p/plot.py [--history PATH] [--out PATH]
train.py also calls plot_history() after every epoch, so the PNG stays current.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).resolve().parent
TRAIN, VAL = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel("epoch", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def _line(ax, x, y, color, label, marker=None) -> None:
    ax.plot(x, y, color=color, linewidth=2, label=label, marker=marker, markersize=6,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"{y[-1]:.3f}" if max(y) > 1.5 else f"{y[-1]:.1%}", (x[-1], y[-1]),
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)


def plot_history(history: list[dict], out: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    _line(ax, epochs, [h["train_loss"] for h in history], TRAIN, "train")
    _line(ax, epochs, [h["val_loss"] for h in history], VAL, "validation")
    _style(ax, "Loss (cross-entropy, label-smoothed)", "loss")
    ax.legend(frameon=False, labelcolor=INK)

    ax = axes[1]
    _line(ax, epochs, [h["train_acc"] for h in history], TRAIN, "train")
    _line(ax, epochs, [h["val_acc"] for h in history], VAL, "validation")
    _style(ax, "Phoneme accuracy (teacher-forced)", "accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, labelcolor=INK)

    ax = axes[2]
    evals = [h for h in history if "val_word_acc" in h]
    if evals:
        _line(ax, [h["epoch"] for h in evals], [h["val_word_acc"] for h in evals], VAL,
              "validation", marker="o")
    _style(ax, "Word accuracy (greedy decode, exact match)", "accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(axes[0].get_xlim())

    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, default=HERE / "checkpoints/history.json")
    ap.add_argument("--out", type=Path, default=HERE / "checkpoints/training_curves.png")
    args = ap.parse_args()
    plot_history(json.loads(args.history.read_text()), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
