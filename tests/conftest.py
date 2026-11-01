import shutil
if shutil.which('ffmpeg') is None:  # ffmpeg for musdb / stempeg on machines without it on PATH
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except Exception:
        pass
import pytest
from nussl import efz_utils
import tempfile
import os
import musdb
import zipfile
import numpy as _np
if not hasattr(_np, 'Inf'):  # scaper still uses np.Inf, removed in NumPy 2
    _np.Inf = _np.inf
import scaper
import random
import glob
import nussl
from nussl.datasets import transforms
from nussl import datasets
import numpy as np
import torch
import json
import sys
import shutil
os.environ.setdefault('MPLBACKEND', 'Agg')  # no Tk on headless / Windows CI

# Windows can't reopen a NamedTemporaryFile while it's open unless delete_on_close=False (3.12+);
# the tests write to tmp.name inside the `with`, which only worked on Linux / macOS.
if os.name == 'nt' and sys.version_info >= (3, 12):
    _ntf = tempfile.NamedTemporaryFile
    def _ntf_windows(*args, **kwargs):
        if kwargs.get('delete', True):
            kwargs.setdefault('delete_on_close', False)
        return _ntf(*args, **kwargs)
    tempfile.NamedTemporaryFile = _ntf_windows


def _unzip(path_to_zip, target_path):
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(target_path)


fix_dir = os.path.expanduser('~/.nussl/tests/')
os.makedirs(fix_dir, exist_ok=True)
# NUSSL_OVERWRITE_REGRESSION=1 rewrites the stored scores (only after checking a change is expected)
OVERWRITE_REGRESSION_DATA = os.environ.get('NUSSL_OVERWRITE_REGRESSION') == '1'


@pytest.fixture(scope="module")
def benchmark_audio(toy_datasets):
    audio_files = {}
    keys = ['K0140.wav', 'K0149.wav', 'dev1_female3_inst_mix.wav']
    with tempfile.TemporaryDirectory() as tmp_dir:
        _dir = tmp_dir if fix_dir is None else fix_dir
        for k in keys:
            try:
                audio_files[k] = efz_utils.download_audio_file(k, _dir)
            except efz_utils.NoConnectivityError:
                audio_files[k] = _standin_audio(k, _dir, toy_datasets['babywsj_oW0F0H9.zip'])
        yield audio_files


STANDINS = set()

def _standin_audio(name, folder, wsj_dir):
    """The EFZ server is retired and its audio files were never archived, so build stand-ins
    with the same names from the (archived) babywsj speech: K0140 / K0149 mono at 44.1 kHz,
    dev1_female3_inst_mix a 3-speaker stereo instantaneous mix (like the SiSEC file), all 44.1 kHz.
    Regression data for tests that use them is kept apart (regression_standin/)."""
    import librosa, soundfile as sf
    path = os.path.join(folder, 'standin_v2', name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    STANDINS.add(name)
    if os.path.exists(path):
        return path
    wavs = sorted(glob.glob(os.path.join(wsj_dir, '**', '*.wav'), recursive=True))
    load = lambda i, sr: librosa.load(wavs[i % len(wavs)], sr=sr, mono=True)[0]
    if name == 'dev1_female3_inst_mix.wav':
        srcs = [load(i, 44100) for i in (0, 7, 13)]
        n = min(len(x) for x in srcs)
        gains = [(0.9, 0.3), (0.55, 0.55), (0.25, 0.95)]
        mix = np.stack([sum(g[c] * x[:n] for g, x in zip(gains, srcs)) for c in (0, 1)], 1)
        mix = mix[:len(mix) // 4410 * 4410]
        sf.write(path, 0.7 * mix / np.abs(mix).max(), 44100, subtype='PCM_16')
    else:
        k = 0 if name == 'K0140.wav' else 2
        y = np.concatenate([load(k, 44100), load(k + 1, 44100)])
        y = y[:len(y) // 4410 * 4410]  # whole tenths of a second, like the originals
        sf.write(path, 0.8 * y / np.abs(y).max(), 44100, subtype='PCM_16')
    return path


@pytest.fixture(scope="module")
def musdb_tracks():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _dir = tmp_dir if fix_dir is None else fix_dir
        db = musdb.DB(root=_dir, download=True)
        yield db


@pytest.fixture(scope="module")
def toy_datasets():
    dataset_locations = {}
    keys = ['babywsj_oW0F0H9.zip']
    with tempfile.TemporaryDirectory() as tmp_dir:
        _dir = tmp_dir if fix_dir is None else fix_dir
        for k in keys:
            target_folder = os.path.join(_dir, os.path.splitext(k)[0])
            data = efz_utils.download_benchmark_file(k, _dir)
            _unzip(data, target_folder)
            dataset_locations[k] = target_folder
        yield dataset_locations


@pytest.fixture(scope="module")
def mix_source_folder(toy_datasets):
    wsj_sources = toy_datasets['babywsj_oW0F0H9.zip']
    audio_files = glob.glob(
        f"{wsj_sources}/**/*.wav", recursive=True)
    n_sources = 2
    n_mixtures = 10

    with tempfile.TemporaryDirectory() as tmp_dir:
        _dir = tmp_dir if fix_dir is None else fix_dir
        _dir = os.path.join(_dir, 'mix_source_folder')
        for i in range(n_mixtures):
            sources = []
            for n in range(n_sources):
                path = random.choice(audio_files)
                source = nussl.AudioSignal(path)
                sources.append(source)

            min_length = min([s.signal_length for s in sources])

            for n in range(n_sources):
                output_path = os.path.join(_dir, f's{n}', f'{i}.wav')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                sources[n].truncate_samples(min_length)
                sources[n].write_audio_to_file(output_path)

            mix = sum(sources)
            output_path = os.path.join(_dir, 'mix', f'{i}.wav')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            mix.write_audio_to_file(output_path)
        yield _dir


@pytest.fixture(scope="module")
def scaper_folder(toy_datasets):
    if shutil.which('sox') is None:  # scaper renders through the SoX binary
        pytest.skip('SoX is not installed (needed by scaper to build the test mixtures)')
    wsj_sources = toy_datasets['babywsj_oW0F0H9.zip']
    fg_path = os.path.join(
        wsj_sources, 'babywsj', 'dev')

    n_sources = 2
    n_mixtures = 10
    duration = 3
    ref_db = -40

    with tempfile.TemporaryDirectory() as tmp_dir:
        _dir = tmp_dir if fix_dir is None else fix_dir
        _dir = os.path.join(_dir, 'scaper')
        os.makedirs(_dir, exist_ok=True)
        for i in range(n_mixtures):
            sc = scaper.Scaper(
                duration, fg_path, fg_path, random_state=i)
            sc.ref_db = ref_db
            sc.sr = 16000

            for j in range(n_sources):
                sc.add_event(
                    label=('choose', []),
                    source_file=('choose', []),
                    source_time=('const', 0),
                    event_time=('const', 0),
                    event_duration=('const', duration),
                    snr=('const', 0),
                    pitch_shift=None,
                    time_stretch=None
                )

            audio_path = os.path.join(_dir, f'{i}.wav')
            jams_path = os.path.join(_dir, f'{i}.jams')
            sc.generate(audio_path, jams_path, save_isolated_events=True)

        yield _dir


@pytest.fixture(scope="module")
def mix_and_sources(scaper_folder):
    dataset = datasets.Scaper(scaper_folder)
    item = dataset[0]
    return item['mix'], item['sources']


@pytest.fixture(scope="module")
def music_mix_and_sources(musdb_tracks):
    dataset = datasets.MUSDB18(
        folder=musdb_tracks.root, download=False,
        transform=transforms.SumSources(
            [['drums', 'bass', 'other']]))
    item = dataset[0]
    return item['mix'], item['sources']


@pytest.fixture(scope="module")
def drum_and_vocals(musdb_tracks):
    dataset = datasets.MUSDB18(
        folder=musdb_tracks.root, download=False)
    item = dataset[0]
    return item['sources']['drums'], item['sources']['vocals']

@pytest.fixture(scope="module")
def random_noise():
    def _random_noise(duration, ch, kind):
        if kind == 'ones':
            x = np.ones((int(ch), int(duration * 44100)))
        elif kind == 'random':
            x = np.random.randn(int(ch), int(duration * 44100))
        signal = nussl.AudioSignal(
            audio_data_array=x, 
            sample_rate=44100
        )
        signal.peak_normalize()
        return signal
    return _random_noise

@pytest.fixture(scope="module")
def bad_scaper_folder(toy_datasets):
    if shutil.which('sox') is None:  # scaper renders through the SoX binary
        pytest.skip('SoX is not installed (needed by scaper to build the test mixtures)')
    wsj_sources = toy_datasets['babywsj_oW0F0H9.zip']
    fg_path = os.path.join(
        wsj_sources, 'babywsj', 'dev')

    n_sources = 2
    n_mixtures = 10
    duration = 3
    ref_db = -40

    with tempfile.TemporaryDirectory() as tmp_dir:
        _dir = tmp_dir if fix_dir is None else fix_dir
        _dir = os.path.join(_dir, 'bad_scaper')
        os.makedirs(_dir, exist_ok=True)
        for i in range(n_mixtures):
            sc = scaper.Scaper(
                duration, fg_path, fg_path, random_state=i)
            sc.ref_db = ref_db
            sc.sr = 16000

            for j in range(n_sources):
                sc.add_event(
                    label=('choose', []),
                    source_file=('choose', []),
                    source_time=('const', 0),
                    event_time=('const', 0),
                    event_duration=('const', duration),
                    snr=('const', 0),
                    pitch_shift=None,
                    time_stretch=None
                )

            audio_path = os.path.join(_dir, f'{i}.wav')
            jams_path = os.path.join(_dir, f'{i}.jams')
            sc.generate(audio_path, jams_path, save_isolated_events=False)

        yield _dir


@pytest.fixture(scope="module")
def one_item(scaper_folder):
    stft_params = nussl.STFTParams(
        window_length=512,
        hop_length=128
    )
    tfms = transforms.Compose([
        transforms.PhaseSensitiveSpectrumApproximation(),
        transforms.GetAudio(),
        transforms.ToSeparationModel()
    ])
    dataset = nussl.datasets.Scaper(
        scaper_folder, transform=tfms, stft_params=stft_params)
    i = np.random.randint(len(dataset))
    data = dataset[i]
    for k in data:
        # fake a batch dimension
        if torch.is_tensor(data[k]):
            data[k] = data[k].unsqueeze(0)
    yield data


@pytest.fixture
def check_against_regression_data(request):
    standin = 'benchmark_audio' in request.fixturenames and STANDINS
    def check(scores, path, atol=1e-4):
        if standin:  # scores on stand-in audio are not comparable to the originals
            path = path.replace('regression', 'regression_standin', 1)
            os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(scores, f, indent=4)
        if OVERWRITE_REGRESSION_DATA:
            with open(path, 'w') as f:
                json.dump(scores, f, indent=4)
        else:
            with open(path, 'r') as f:
                reg_scores = json.load(f)
            for key in scores:
                if key not in ['permutation', 'combination']:
                    for metric in scores[key]:
                        if metric in reg_scores[key]:
                            tol = atol.get(metric, 1e-4) if isinstance(atol, dict) else atol
                            assert np.allclose(
                                scores[key][metric],
                                reg_scores[key][metric],
                                atol=tol
                            )
    return check

