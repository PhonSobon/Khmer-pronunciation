# Khmer Pronunciation (G2P)

A small Transformer (7.5M parameters) that converts Khmer words to their pronunciation, in two formats:

| task | example |
|---|---|
| `respell` – NCKL Khmer-script respelling | ក្របី → ក្រ-បី |
| `ipa` – phonemes | ក្របី → k r ɑ . ɓ ə j |

Training data is downloaded from the Hugging Face Hub:
[phonsobon/khmer-dictionary](https://huggingface.co/datasets/phonsobon/khmer-dictionary) (`g2p` config) and
[seanghay/google-khmer-lexicon](https://huggingface.co/datasets/seanghay/google-khmer-lexicon).

## Train on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PhonSobon/Khmer-pronunciation/blob/main/colab_train.ipynb)

Choose **Runtime → Change runtime type → GPU**, then run the cells in order.

## Train locally

```bash
pip install -r requirements.txt
python g2p/prepare_data.py               # writes g2p/data/{train,validation,test}.tsv
python g2p/train.py --epochs 100         # best model -> g2p/checkpoints/best.pt
```

Options: `--batch 256`, `--lr 5e-4`, `--eval-every 5`, `--patience 4` (stop after this many
validation checks without improvement).

## Use

```bash
python g2p/predict.py ក្របី កម្ពុជា
python g2p/predict.py ក្របី កម្ពុជា --task ipa
```

```python
from predict import KhmerG2P   # run from inside g2p/
g2p = KhmerG2P()
g2p("ក្របី")               # 'ក្រ-បី'
g2p("ក្របី", task="ipa")   # 'k r ɑ . ɓ ə j'
```
