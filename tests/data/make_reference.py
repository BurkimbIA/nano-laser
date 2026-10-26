"""Writes reference.npz with the original laser_encoders, in an environment where it is installed:

    pip install laser_encoders==0.0.2 fairseq==0.12.2 torch==2.1.2
    python tests/data/make_reference.py

LASER3 is encoded one sentence at a time, laser_encoders' default: with larger batches, fairseq's
BetterTransformer path fails on left-padded batches ("Mask Type should be defined").
"""

import json
from pathlib import Path

import numpy as np
from laser_encoders import initialize_encoder, initialize_tokenizer

DATA = Path(__file__).parent
sentences = json.loads((DATA / "sentences.json").read_text(encoding="utf-8"))
reference = {}
for lang in ["english", "mos_Latn", "dyu_Latn", "fuv_Latn"]:
    tokenizer = initialize_tokenizer(lang=lang)
    encoder = initialize_encoder(lang=lang)
    encoder.use_cuda = False
    encoder.encoder.cpu()
    reference[lang] = encoder.encode_sentences([tokenizer.tokenize(s) for s in sentences])
np.savez_compressed(DATA / "reference.npz", **reference)
