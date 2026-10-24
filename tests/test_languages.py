import pytest

from nano_laser import LaserEncoderPipeline, initialize_encoder, initialize_tokenizer
from nano_laser.download_models import LaserModelDownloader
from nano_laser.language_list import LASER3_LANGUAGE


def test_language_names_resolve_to_codes(tmp_path):
    downloader = LaserModelDownloader(tmp_path)
    assert downloader.get_language_code(LASER3_LANGUAGE, "mos_Latn") == "mos_Latn"
    assert downloader.get_language_code(LASER3_LANGUAGE, "dyu") == "dyu_Latn"


def test_ambiguous_language_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="multiple options"):
        LaserModelDownloader(tmp_path).get_language_code(LASER3_LANGUAGE, "kas")


@pytest.mark.parametrize("factory", [initialize_encoder, initialize_tokenizer])
def test_unknown_language_is_rejected(tmp_path, factory):
    with pytest.raises(ValueError, match="Unsupported language name"):
        factory(lang="klingon", model_dir=str(tmp_path))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({}, "Either 'laser' or 'lang'"), ({"laser": "laser3"}, "'lang' parameter is required")],
)
def test_pipeline_arguments(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        LaserEncoderPipeline(model_dir=str(tmp_path), **kwargs)
