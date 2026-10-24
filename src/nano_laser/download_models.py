# Adapted from laser_encoders.download_models (LASER, BSD license, Copyright (c) Facebook, Inc. and its affiliates).
# Same download locations and cache directory, with urllib instead of requests and tqdm.

import logging
import os
import tempfile
import urllib.request
from pathlib import Path

from nano_laser.language_list import LASER3_LANGUAGE, SPM_LANGUAGE

logger = logging.getLogger(__name__)


class LaserModelDownloader:
    def __init__(self, model_dir: str | None = None):
        if model_dir is None:
            model_dir = os.path.expanduser("~/.cache/laser_encoders")  # shared with laser_encoders
            os.makedirs(model_dir, exist_ok=True)

        self.model_dir = Path(model_dir)
        self.base_url = "https://dl.fbaipublicfiles.com/nllb/laser"

    def download(self, filename: str):
        local_file_path = self.model_dir / filename
        if local_file_path.exists():
            logger.info(f" - {filename} already downloaded")
            return

        logger.info(f" - Downloading {filename}")
        # download next to the target, then rename, so an interrupted download never looks complete
        fd, temp_file_path = tempfile.mkstemp(dir=self.model_dir, prefix=f".{filename}.")
        try:
            with os.fdopen(fd, "wb") as tf, urllib.request.urlopen(f"{self.base_url}/{filename}") as response:
                while chunk := response.read(1 << 20):
                    tf.write(chunk)
            os.replace(temp_file_path, local_file_path)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def get_language_code(self, language_list: dict, lang: str) -> str:
        try:
            lang_3_4 = language_list[lang]
        except KeyError:
            raise ValueError(
                f"language name: {lang} not found in language list. Specify a supported language name"
            ) from None
        if isinstance(lang_3_4, list):
            options = ", ".join(f"'{opt}'" for opt in lang_3_4)
            raise ValueError(
                f"Language '{lang}' has multiple options: {options}. Please specify using the 'lang' argument."
            )
        return lang_3_4

    def download_laser2(self):
        self.download("laser2.pt")
        self.download("laser2.spm")
        self.download("laser2.cvocab")

    def download_laser3(self, lang: str, spm: bool = False):
        lang = self.get_language_code(LASER3_LANGUAGE, lang)
        self.download(f"laser3-{lang}.v1.pt")
        if spm:
            if lang in SPM_LANGUAGE:
                self.download(f"laser3-{lang}.v1.spm")
                self.download(f"laser3-{lang}.v1.cvocab")
            else:
                self.download("laser2.spm")
                self.download("laser2.cvocab")
