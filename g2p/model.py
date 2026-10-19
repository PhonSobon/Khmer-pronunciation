"""Character-level Transformer for Khmer grapheme-to-phoneme conversion.

The source is `<task>` + the word's characters; the target is a token sequence:
Khmer characters with `-` between syllables for task `respell`.
"""
import math

import torch
from torch import nn

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]
TASKS = ["respell"]


class Vocab:
    def __init__(self, tokens: list[str]):
        self.itos = SPECIALS + sorted(set(tokens) - set(SPECIALS))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, UNK) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        out = []
        for i in ids:
            if i == EOS:
                break
            if i > UNK:
                out.append(self.itos[i])
        return out


def source_tokens(task: str, word: str) -> list[str]:
    return [f"<{task}>"] + list(word)


def format_output(task: str, tokens: list[str]) -> str:
    return "".join(tokens) if task == "respell" else " ".join(tokens)


class G2PModel(nn.Module):
    def __init__(self, n_src: int, n_tgt: int, d_model: int = 256, heads: int = 4,
                 layers: int = 4, ff: int = 1024, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.config = dict(n_src=n_src, n_tgt=n_tgt, d_model=d_model, heads=heads,
                           layers=layers, ff=ff, dropout=dropout, max_len=max_len)
        self.scale = math.sqrt(d_model)
        self.src_emb = nn.Embedding(n_src, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(n_tgt, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        self.transformer = nn.Transformer(
            d_model, heads, layers, layers, ff, dropout, batch_first=True, norm_first=True)
        self.out = nn.Linear(d_model, n_tgt)

    def _embed(self, emb: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        return emb(x) * self.scale + self.pos(pos)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        return self.transformer.encoder(self._embed(self.src_emb, src),
                                        src_key_padding_mask=src == PAD)

    def decode_step(self, memory: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        mask = self.transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        h = self.transformer.decoder(self._embed(self.tgt_emb, tgt), memory, tgt_mask=mask,
                                     tgt_is_causal=True, tgt_key_padding_mask=tgt == PAD,
                                     memory_key_padding_mask=src == PAD)
        return self.out(h)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.decode_step(self.encode(src), src, tgt_in)

    @torch.no_grad()
    def greedy(self, src: torch.Tensor, max_len: int = 100) -> list[list[int]]:
        memory = self.encode(src)
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            nxt = self.decode_step(memory, src, ys)[:, -1].argmax(-1)
            nxt = nxt.masked_fill(done, PAD)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        return ys[:, 1:].tolist()


def pad_batch(seqs: list[list[int]]) -> torch.Tensor:
    width = max(map(len, seqs))
    return torch.tensor([s + [PAD] * (width - len(s)) for s in seqs], dtype=torch.long)
