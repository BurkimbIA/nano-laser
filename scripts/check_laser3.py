"""Compares nano_laser with laser_encoders on every LASER3 model, one model at a time.

Each checkpoint (600 MB) is downloaded, encoded by both packages and deleted, so the disk holds one at a time.
Results are appended to a JSONL file, one line per language, and languages already in it are skipped:
an interrupted run resumes where it stopped. See notebooks/check_laser3.ipynb to run it on Colab.

    python scripts/check_laser3.py --ref-python /path/to/laser_encoders/env/bin/python --results results.jsonl
"""

import argparse
import json
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from nano_laser import initialize_encoder, initialize_tokenizer
from nano_laser.download_models import LaserModelDownloader
from nano_laser.language_list import LASER3_LANGUAGES_LIST
from nano_laser.models import load_checkpoint

ROOT = Path(__file__).resolve().parent.parent
CFG_FIELDS = [
    "encoder_layers",
    "encoder_embed_dim",
    "encoder_attention_heads",
    "encoder_normalize_before",
    "activation_fn",
    "sentemb_criterion",
    "prepend_bos",
    "left_pad_source",
]


def cosine(a, b):
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def check(lang, model_dir, sentences_path, ref_python, cpu):
    result = {"lang": lang}
    downloader = LaserModelDownloader(model_dir)
    downloader.download_laser3(lang, spm=True)
    checkpoint = load_checkpoint(model_dir / f"laser3-{lang}.v1.pt")
    if "params" in checkpoint:
        result["arch"] = "lstm"
    else:
        cfg = checkpoint["cfg"]["model"]
        result["arch"] = "transformer"
        result.update({name: getattr(cfg, name, None) for name in CFG_FIELDS})
    del checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "ref.npz"
        cmd = [ref_python, str(ROOT / "scripts" / "laser_encoders_reference.py"), "--lang", lang]
        cmd += ["--model-dir", str(model_dir), "--sentences", str(sentences_path), "--out", str(ref_path)]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(f"laser_encoders failed:\n{run.stderr[-2000:]}")
        ref = np.load(ref_path, allow_pickle=True)
        ref_tokens, expected = list(ref["tokens"]), ref["embeddings"]

    sentences = json.loads(Path(sentences_path).read_text(encoding="utf-8"))
    tokens = [
        initialize_tokenizer(lang=lang, model_dir=str(model_dir), laser="laser3").tokenize(s)
        for s in sentences
    ]
    result["tokens_match"] = tokens == ref_tokens
    for max_sentences in [1, 8]:
        encoder = initialize_encoder(
            lang=lang, model_dir=str(model_dir), laser="laser3", cpu=cpu, max_sentences=max_sentences
        )
        embeddings = encoder.encode_sentences(ref_tokens)
        diff = np.abs(embeddings - expected)
        result[f"max_abs_diff_bs{max_sentences}"] = float(diff.max())
        result[f"max_rel_diff_bs{max_sentences}"] = float((diff.max(1) / np.abs(expected).max(1)).max())
        result[f"min_cosine_bs{max_sentences}"] = float(cosine(embeddings, expected).min())
        del encoder
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ref-python", required=True, help="python of the env with laser_encoders and fairseq"
    )
    parser.add_argument("--results", default="laser3_parity.jsonl")
    parser.add_argument("--model-dir", default=None, help="default: a temporary directory")
    parser.add_argument("--sentences", default=str(ROOT / "tests" / "data" / "sentences.json"))
    parser.add_argument("--langs", nargs="*", help="default: every LASER3 language")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    args = parser.parse_args()

    results_path = Path(args.results)
    done = set()
    if results_path.exists():
        done = {json.loads(line)["lang"] for line in results_path.read_text().splitlines() if line.strip()}
    langs = [lang for lang in (args.langs or sorted(set(LASER3_LANGUAGES_LIST))) if lang not in done]
    print(f"{len(done)} languages already checked, {len(langs)} to go", flush=True)

    model_dir = Path(args.model_dir or tempfile.mkdtemp(prefix="laser-models-"))
    model_dir.mkdir(parents=True, exist_ok=True)
    for i, lang in enumerate(langs, 1):
        start = time.time()
        try:
            result = check(lang, model_dir, args.sentences, args.ref_python, args.cpu)
            result["ok"] = (
                result["tokens_match"]
                and min(result["min_cosine_bs1"], result["min_cosine_bs8"]) >= args.min_cosine
            )
        except Exception as error:  # recorded, so that one bad checkpoint does not stop the run
            result = {"lang": lang, "ok": False, "error": f"{type(error).__name__}: {error}"}
            traceback.print_exc()
        finally:
            (model_dir / f"laser3-{lang}.v1.pt").unlink(missing_ok=True)  # the spm and cvocab files are small
        result["seconds"] = round(time.time() - start, 1)
        with results_path.open("a") as f:
            f.write(json.dumps(result) + "\n")
        status = "ok" if result["ok"] else "FAILED"
        detail = (
            result.get("error") or f"min cosine {min(result['min_cosine_bs1'], result['min_cosine_bs8']):.7f}"
        )
        print(f"[{i}/{len(langs)}] {lang}: {status}, {detail} ({result['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
