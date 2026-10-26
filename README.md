# nano-laser

The [LASER](https://github.com/facebookresearch/LASER) sentence encoders (LASER2 and the LASER3 language
models) in plain torch. It is a drop-in replacement for [`laser_encoders`](https://pypi.org/project/laser-encoders/),
without fairseq: the same API, the same checkpoints, the same embeddings.

## Install

```bash
pip install nano-laser
# or, in a uv project
uv add nano-laser
```

It needs Python ≥ 3.10 and torch ≥ 2.5. To get unreleased changes, install from GitHub instead:
`pip install git+https://github.com/BurkimbIA/nano-laser.git` (append `@<commit or tag>` to pin one).

## Usage

Replace the import, nothing else:

```python
from nano_laser import LaserEncoderPipeline  # was: from laser_encoders import LaserEncoderPipeline

encoder = LaserEncoderPipeline(lang="mos_Latn")
embeddings = encoder.encode_sentences(["Laafi bala?"], normalize_embeddings=True)  # (1, 1024)
```

`initialize_tokenizer`, `initialize_encoder`, `SentenceEncoder`, `LaserTokenizer` and the module layout
(`nano_laser.models`, `nano_laser.laser_tokenizer`, `nano_laser.download_models`, `nano_laser.language_list`)
mirror `laser_encoders`. The models are downloaded from the same place into the same cache,
`~/.cache/laser_encoders`, so models already downloaded by `laser_encoders` are reused.

Like `laser_encoders`, the encoder takes one sentence per batch by default. Batch on GPU with the
`SentenceEncoder` options, which `initialize_encoder` and `LaserEncoderPipeline` now pass on:

```python
encoder = LaserEncoderPipeline(lang="dyu_Latn", max_sentences=64)  # or max_tokens=..., cpu=True, fp16=True
```

## Parity

Against `laser_encoders` 0.0.2 (fairseq 0.12.2, torch 2.1.2), on 85 sentences of 0 to 2,000 tokens:

| model | max abs. difference (cpu) | min cosine (cpu and gpu, batched or not) |
| --- | --- | --- |
| LASER2 (LSTM) | 1.3e-6 | 0.9999996 |
| LASER3 mos_Latn, dyu_Latn, fuv_Latn (transformer) | 1.4e-5 | 0.9999998 |

Every LASER3 model is also checked, on a Colab T4 with 18 sentences, one at a time and in batches of 8
([`notebooks/check_laser3.ipynb`](notebooks/check_laser3.ipynb)): all 147 give the same tokens and embeddings,
with a min cosine of 0.9999997 and a max abs. difference of 2.0e-5. They come in three kinds:

| LASER3 models | languages |
| --- | --- |
| 12-layer transformer, `maxpool` pooling | 135 |
| LSTM, like LASER2 | 10: `als_Latn`, `cym_Latn`, `hye_Armn`, `kat_Geor`, `khk_Cyrl`, `khm_Khmr`, `ltz_Latn`, `mlt_Latn`, `quy_Latn`, `ydd_Hebr` |
| 12-layer transformer, `cls` pooling on a leading `<s>` | 2: `lij_Latn`, `wol_Latn` |

The results per language are in [`scripts/results/laser3_parity.jsonl`](scripts/results/laser3_parity.jsonl).

`tests/test_parity.py` checks this against `tests/data/reference.npz` whenever the models are in the cache
(or in `LASER_MODEL_DIR`); it is skipped otherwise, so CI downloads nothing.

## Differences from laser_encoders

- No fairseq: the transformer encoder and fairseq's `Dictionary` are re-implemented in `nano_laser/models.py`.
- Checkpoints load with `torch.load(weights_only=True)`: they only hold tensors and an `argparse.Namespace`.
- LASER3 works with `max_sentences > 1`. In `laser_encoders` on torch ≥ 1.12, fairseq's BetterTransformer path fails on
  left-padded batches (`RuntimeError: Mask Type should be defined`).
- With `sentemb_criterion="cls"`, each sentence of a batch gets its own `<s>` state whatever the padding side.
  `laser_encoders` takes the `<s>` states in position order, which is only right when they share a position:
  true for the two `cls` models (`lij_Latn`, `wol_Latn`), which pad on the right.
- Importing the package does not call `logging.basicConfig`, and a failed download raises instead of saving the
  error page as the model.

## Development

```bash
just install  # uv sync and the pre-commit hooks
just test
just lint
```

To regenerate the parity reference, see `tests/data/make_reference.py`.

## License

[BSD 3-Clause](LICENSE), Copyright (c) 2026, BurkimbIA. The code adapted from LASER keeps its copyright,
Copyright (c) Facebook, Inc. and its affiliates, also under the BSD license; each such file says so in its header.
The LASER model weights are downloaded from Facebook and are not covered by this license.
