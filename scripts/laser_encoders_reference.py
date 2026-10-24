"""Encodes the parity sentences with the original laser_encoders, for scripts/check_laser3.py.

Runs in a separate environment with laser_encoders 0.0.2 and fairseq 0.12.2 (Python 3.10):

    python scripts/laser_encoders_reference.py --lang mos_Latn --model-dir DIR --sentences S.json --out ref.npz
"""

import argparse
import json

import numpy as np
from laser_encoders import initialize_encoder, initialize_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--lang", required=True)
parser.add_argument("--model-dir", required=True)
parser.add_argument("--sentences", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

sentences = json.load(open(args.sentences, encoding="utf-8"))
tokenizer = initialize_tokenizer(lang=args.lang, model_dir=args.model_dir, laser="laser3")
encoder = initialize_encoder(lang=args.lang, model_dir=args.model_dir, laser="laser3")
tokens = [tokenizer.tokenize(s) for s in sentences]
# one sentence per batch, the default: larger batches fail in fairseq's BetterTransformer path
np.savez(args.out, tokens=np.array(tokens, dtype=object), embeddings=encoder.encode_sentences(tokens))
