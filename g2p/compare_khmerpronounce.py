"""Compare our model with the `khmerpronounce` package on the respell test set.

khmerpronounce is tested two ways:
- `pronounce()`: lexicon lookup first. Its lexicon is built from the same NCKL
  dictionary, so it has probably seen these test words (an unfair advantage).
- `model.phoneticize()`: its Phonetisaurus FST model alone, closer to a fair test.

Needs Python UTF-8 mode on Windows: set PYTHONUTF8=1.
"""
from collections import defaultdict

from khmerpronounce import model as kp_fst
from khmerpronounce import pronounce as kp_pronounce

from predict import KhmerG2P
from train import edit_distance, load


def score(name: str, refs: dict, preds: dict) -> None:
    wrong = edits = total = 0
    for w, rs in refs.items():
        p = list(preds[w])
        best = min(rs, key=lambda r: edit_distance(p, list(r)))
        e = edit_distance(p, list(best))
        wrong += e > 0; edits += e; total += len(best)
    print(f"{name:<34} WER {wrong / len(refs):7.2%}   character error {edits / total:6.2%}")


def main() -> None:
    test = load("test")
    test = test[test.task == "respell"]
    refs = defaultdict(list)
    for w, y in zip(test.word, test.target):
        refs[w].append(y.replace(" ", ""))
    words = list(refs)
    print(f"respell test words: {len(words)}\n")

    ours = KhmerG2P()(words)
    score("this model", refs, dict(zip(words, ours)))
    score("khmerpronounce FST only", refs, {w: "-".join(kp_fst.phoneticize(w)) for w in words})
    score("khmerpronounce pronounce() (lexicon)", refs, {w: "-".join(kp_pronounce(w)) for w in words})

    print("\nwhere this model is wrong (first 15):")
    errors = [(w, p) for w, p in zip(words, ours) if p not in refs[w]]
    for w, p in errors[:15]:
        print(f"  {w:<16} predicted {p:<22} reference {' | '.join(refs[w])}")


if __name__ == "__main__":
    main()
