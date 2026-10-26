"""LaserModelDownloader against a fake urlopen and a silent local server: no network."""

import io
import socket

import pytest

from nano_laser import download_models
from nano_laser.download_models import LaserModelDownloader


def test_creates_a_missing_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        download_models.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"weights")
    )
    LaserModelDownloader(tmp_path / "new" / "dir").download("laser2.cvocab")
    assert (tmp_path / "new" / "dir" / "laser2.cvocab").read_bytes() == b"weights"


def test_a_silent_server_times_out(tmp_path):
    with socket.create_server(("127.0.0.1", 0)) as server:  # never accepts, so never answers
        downloader = LaserModelDownloader(tmp_path)
        downloader.base_url = f"http://127.0.0.1:{server.getsockname()[1]}"
        downloader.timeout = 0.5
        with pytest.raises(OSError):
            downloader.download("laser2.cvocab")
    assert list(tmp_path.iterdir()) == []  # no partial file left behind
