# Adapted from laser_encoders.laser_tokenizer (LASER, BSD license, Copyright (c) Facebook, Inc. and its affiliates).
# The original is already free of fairseq; it is kept as is, so that tokens stay identical.

import gzip
import logging
import os
import re
from pathlib import Path
from typing import IO

import sentencepiece as spm
from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from unicategories import categories

from nano_laser.download_models import LaserModelDownloader
from nano_laser.language_list import LASER2_LANGUAGE, LASER3_LANGUAGE, SPM_LANGUAGE

SPACE_NORMALIZER = re.compile(r"\s+")
NON_PRINT_CHARS = set(categories["C"].characters())

logger = logging.getLogger(__name__)


class LaserTokenizer:
    def __init__(
        self,
        spm_model: Path,
        lang: str = "en",
        lower_case: bool = True,
        descape: bool = False,
        verbose: bool = False,
        over_write: bool = False,
        normalize_punct: bool = True,
    ):
        self.spm_model = Path(spm_model)
        self.lang = lang
        self.lower_case = lower_case
        self.descape = descape
        self.verbose = verbose
        self.over_write = over_write
        self.normalize_punct = normalize_punct

        assert self.spm_model.exists(), f"spm model file: {spm_model} does not exist"
        self.moses_punct_normalizer = MosesPunctNormalizer(self.lang, perl_parity=True)
        # add parity with MOSES release-4.0
        self.moses_punct_normalizer.substitutions[21] = ("‘", r'"')
        self.moses_punct_normalizer.substitutions[22] = ("‚", r'"')
        self.moses_detokenizer = MosesDetokenizer()
        self.spm_encoder = spm.SentencePieceProcessor(model_file=str(self.spm_model))

    def open(self, file: Path, mode: str, encoding="utf-8") -> IO:
        return (
            gzip.open(file, mode, encoding=encoding)
            if file.name.endswith(".gz")
            else open(file, mode, encoding=encoding)
        )

    def log(self, message: str) -> None:
        if self.verbose:
            logger.info(message)

    def tokenize(self, text: str) -> str:
        # Preprocessing
        sentence_text = "".join([c if c not in NON_PRINT_CHARS else " " for c in text])
        if self.normalize_punct:
            sentence_text = self.moses_punct_normalizer.normalize(sentence_text)
        if self.descape:
            sentence_text = self.moses_detokenizer.unescape_xml(text=sentence_text)
        if self.lower_case:
            sentence_text = sentence_text.lower()

        # SentencePiece encoding
        return " ".join(self.spm_encoder.encode(sentence_text, out_type=str))

    def tokenize_file(self, inp_fname: Path, out_fname: Path) -> None:
        if not self.over_write and out_fname.exists():
            self.log(f"tokenized file {out_fname.name} already exists")
            return
        self.log(
            f"tokenizing {inp_fname.name}"
            + f"{' (de-escaped)' if self.descape else ''}"
            + f"{' (lower-cased)' if self.lower_case else ' (cased)'} "
            + f"(punctuation-normalization lang: {self.lang})"
        )
        with self.open(inp_fname, "rt") as file_in, open(out_fname, "w") as file_out:
            for line in file_in:
                file_out.write(self.tokenize(line.strip()) + "\n")

    def __call__(self, text_or_batch):
        if isinstance(text_or_batch, str):
            return self.tokenize(text_or_batch)
        return self.tokenize_batch(text_or_batch)

    def tokenize_batch(self, batch: list[str]) -> list[str]:
        return [self.tokenize(text) for text in batch]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.spm_encoder.DecodeIds(i) for i in ids]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        ids = []
        for token in tokens:
            for piece in SPACE_NORMALIZER.sub(" ", token).strip().split():
                token_id = self.spm_encoder.PieceToId(piece)
                if token_id == 0:  # Handle out-of-vocabulary tokens
                    token_id = self.spm_encoder.PieceToId("<unk>")
                ids.append(token_id)
        return ids


def initialize_tokenizer(lang: str | None = None, model_dir: str | None = None, laser: str | None = None):
    downloader = LaserModelDownloader(model_dir)
    if laser is not None:
        if laser == "laser3":
            lang = downloader.get_language_code(LASER3_LANGUAGE, lang)
            filename = f"laser3-{lang}.v1.spm" if lang in SPM_LANGUAGE else "laser2.spm"
        elif laser == "laser2":
            filename = "laser2.spm"
        else:
            raise ValueError(f"Unsupported laser model: {laser}. Choose either laser2 or laser3.")
    elif lang in LASER3_LANGUAGE:
        lang = downloader.get_language_code(LASER3_LANGUAGE, lang)
        filename = f"laser3-{lang}.v1.spm" if lang in SPM_LANGUAGE else "laser2.spm"
    elif lang in LASER2_LANGUAGE:
        filename = "laser2.spm"
    else:
        raise ValueError(f"Unsupported language name: {lang}. Please specify a supported language name.")

    downloader.download(filename)
    return LaserTokenizer(spm_model=Path(os.path.join(downloader.model_dir, filename)))
