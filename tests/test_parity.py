"""The real LASER models against embeddings computed by laser_encoders 0.0.2 (fairseq 0.12.2, torch 2.1.2, cpu).

Each case is skipped unless its files are already in LASER_MODEL_DIR (default ~/.cache/laser_encoders).
With LASER_PARITY_REQUIRED=1 (as in CI), missing models are downloaded instead, so a failed download fails
the test. Regenerate the reference with tests/data/make_reference.py.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from nano_laser import initialize_encoder, initialize_tokenizer
from nano_laser.language_list import SPM_LANGUAGE

DATA = Path(__file__).parent / "data"
MODEL_DIR = Path(os.environ.get("LASER_MODEL_DIR", "~/.cache/laser_encoders")).expanduser()
LANGS = {
    "english": "laser2",
    "mos_Latn": "laser3-mos_Latn.v1",
    "dyu_Latn": "laser3-dyu_Latn.v1",
    "fuv_Latn": "laser3-fuv_Latn.v1",  # the only one here with its own spm
}


def model_files(lang):
    name = LANGS[lang]
    vocab = name if lang in SPM_LANGUAGE or name == "laser2" else "laser2"
    return [f"{name}.pt", f"{vocab}.spm", f"{vocab}.cvocab"]


@pytest.fixture(scope="module")
def reference():
    return json.loads((DATA / "sentences.json").read_text(encoding="utf-8")), np.load(DATA / "reference.npz")


@pytest.mark.parametrize("lang", LANGS)
def test_matches_laser_encoders(lang, reference):
    missing = [f for f in model_files(lang) if not (MODEL_DIR / f).exists()]
    if missing and not os.environ.get("LASER_PARITY_REQUIRED"):
        pytest.skip(f"not in {MODEL_DIR}: {', '.join(missing)} (LASER_PARITY_REQUIRED=1 downloads them)")
    sentences, expected = reference
    tokens = [initialize_tokenizer(lang=lang, model_dir=str(MODEL_DIR)).tokenize(s) for s in sentences]
    for max_sentences in [1, 4]:
        encoder = initialize_encoder(
            lang=lang, model_dir=str(MODEL_DIR), cpu=True, max_sentences=max_sentences
        )
        np.testing.assert_allclose(encoder.encode_sentences(tokens), expected[lang], rtol=1e-4, atol=1e-4)
