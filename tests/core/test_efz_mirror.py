import os
import tempfile
import pytest
import nussl
from nussl.core import constants, efz_utils


def test_efz_mirror_fallback(monkeypatch):
    """With the EFZ unreachable, a benchmark file still comes from the mirror (no hash check)."""
    monkeypatch.setattr(efz_utils, '_download_all_metadata',
                        lambda url: (_ for _ in ()).throw(efz_utils.NoConnectivityError('offline')))
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            path = efz_utils.download_benchmark_file('babywsj_oW0F0H9.zip', tmp_dir, verbose=False)
        except efz_utils.NoConnectivityError:
            pytest.skip('no network to the mirror')
        assert os.path.getsize(path) > 1_000_000
        # second call: served from the local copy, no download
        monkeypatch.setattr(constants, 'NUSSL_EFZ_MIRRORS', [])
        assert efz_utils.download_benchmark_file('babywsj_oW0F0H9.zip', tmp_dir, verbose=False) == path


def test_efz_mirror_gives_up(monkeypatch):
    monkeypatch.setattr(efz_utils, '_download_all_metadata',
                        lambda url: (_ for _ in ()).throw(efz_utils.NoConnectivityError('offline')))
    monkeypatch.setattr(constants, 'NUSSL_EFZ_MIRRORS', [])
    with tempfile.TemporaryDirectory() as tmp_dir:
        pytest.raises(efz_utils.NoConnectivityError, efz_utils.download_audio_file,
                      'does_not_exist.wav', tmp_dir, verbose=False)
