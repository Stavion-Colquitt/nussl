# Changes in this fork (since nussl 1.1.9)

Unofficial modernization of nussl, which the Interactive Audio Lab archived in December 2024.
Goal: the same library, running on current tooling, plus new algorithms. Nothing here is
endorsed by the original authors.

## Runs on current Python / libraries
Tested on Python 3.14 (Windows) with NumPy 2.4, SciPy 1.17, librosa 0.11, PyTorch 2.15, and
on a clean Linux install (Python 3.13, NumPy 2.5, SciPy 1.18, librosa 1.0, scikit-learn 1.9,
PyTorch 2.15, SoX 14.4) where the whole suite runs, scaper / SoX tests included;
CI on Python 3.10 / 3.12 / 3.13 (Linux).

- NumPy 2: `np.bool` -> `bool`, `np.mat` -> `np.asmatrix`, `np.alltrue` -> `np.all` (tests).
- SciPy: window names no longer read from `scipy.signal.hamming` etc. (removed); the old
  `'hanning'` alias maps to `'hann'`.
- librosa >= 0.10 (keyword-only arguments): `resample`, `feature.rms` (keeps the old
  `pad_mode='reflect'`), `feature.melspectrogram`, `filters.mel`; `display.waveplot` ->
  `display.waveshow`; private axis helpers take the new tempo arguments.
- PyTorch >= 2.6 loads with `weights_only=True` by default; nussl checkpoints carry metadata,
  so `SafeModelLoader` passes `weights_only=False` (load only checkpoints you trust).
- pandas: report-card means / stds over the metric columns only.
- zarr pinned below 3 (the cache uses zarr 2's synchronizers); numcodecs >= 0.16 refuses to
  unpickle, so the cache (written by nussl itself) opts back in.
- `museval` is imported lazily: it imports musdb -> stempeg, which needs ffmpeg at import time,
  so `import nussl` no longer requires ffmpeg.
- `importlib_resources` -> standard-library `importlib.resources`.
- Windows: temporary files that other programs (ffmpeg, soundfile) write to are created with
  `delete_on_close=False`; tests do the same and use `num_workers=0` for DataLoaders.
- Packaging: `pyproject.toml` (setuptools); `setup.py` is a shim.

## Found on a clean install (librosa 1.0, scikit-learn 1.9, ffmpeg 6)
Windows skips the SoX-dependent third of the suite, so these only showed up on Linux:

- `audioread` is now a declared dependency. nussl imports it directly; librosa used to pull it
  in, librosa 1.0 doesn't, and `import nussl` failed on a fresh install.
- `MelProjection(direction='backward')`: with more mel bands than the low-frequency FFT bins
  support, the filter bank is rank-deficient and NumPy's default `pinv` cutoff inverted float32
  round-off (weights ~1e13). Now inverted in float64 with a cutoff at float32 precision.
- `ClusteringSeparationBase`: non-float features (e.g. one-hot booleans) are clustered as
  float64; recent scikit-learn's `GaussianMixture` mis-clusters boolean input.
- Tests: `librosa.filters.mel` keyword arguments; `assert not empty_array` (an error in NumPy
  2.5) -> `.size == 0`; the conftest downloads a static ffmpeg only when none is on `PATH`.
- Regression checks that compared things the library doesn't control:
  * effects: the raw sample *sum* of an ffmpeg-filtered signal differs by a few 16-bit LSBs
    over the clip between ffmpeg 6 and 8 (variance agrees to 1e-4) -> sum checked at 1e-2;
  * ICA: sources come back only up to scale, and scikit-learn changed FastICA's whitening, so
    every scale-dependent metric (SD-SDR, SNR, SRR) moved while every scale-invariant one is
    identical -> only scale-invariant metrics are compared;
  * timbre clustering: drift of a few thousandths of a dB -> checked at 1e-2.
- CI installs `libsox-dev` (soxbindings builds from source on Python 3.13). The upstream
  release workflow (PyPI release on every push to master) is manual-only in the fork.

## The External File Zoo is gone
`nussl.ci.northwestern.edu` no longer resolves. `efz_utils` now falls back to a cached copy,
then to the mirrors in `constants.NUSSL_EFZ_MIRRORS` (default: the Internet Archive, overridable
with the `NUSSL_EFZ_MIRRORS` / `NUSSL_EFZ_BASE_URL` environment variables). Mirrored files can't
be hash-checked because the EFZ metadata wasn't archived.

- `babywsj_oW0F0H9.zip` (the toy dataset most tests build mixtures from) is on the Archive.
- `K0140.wav`, `K0149.wav`, `dev1_female3_inst_mix.wav` are not: the tests build stand-ins with
  the same names from babywsj speech. Tests that compare against stored scores and use them
  keep their scores apart in `regression_standin/` (generated on first run), so the original
  regression data is never overwritten by stand-in results.
- Tests of the live EFZ server skip when it's offline; `test_efz_mirror.py` covers the fallback.

## Regression data that was regenerated
Stored scores drifted with the library upgrades (same algorithm, different numerics).
Regenerated with `NUSSL_OVERWRITE_REGRESSION=1`, old -> new SI-SDR (dB):

| Test | Before | After | Likely cause |
|---|---|---|---|
| `test_nmf_mixin` | -4.53 / -2.83 | -4.56 / -2.85 | librosa resampling (soxr), scikit-learn NMF |
| `test_timbre_clustering` | -2.48 / -0.92 | -3.70 / -1.80 | scikit-learn NMF + KMeans, librosa mel filters |

## Optional dependencies now optional
- SoX: only scaper (used to build some test mixtures) needs the binary; those tests skip
  without it (CI installs it).
- vamp / Melodia: the bundled plugin is macOS / Linux only; the test skips elsewhere.
- gradio, ffplay: tests skip when missing.

## New: `nussl.separation.spatial.LeadBacking`
Splits a vocals-only recording into lead and backing vocals from three cues: pan position
(level-based pan index x inter-channel coherence), the lead's harmonic comb (pYIN on the mid
signal; tolerance never narrower than the STFT main lobe), and an optional level prior.
Cues are combined as weighted log-odds, so a cue with no evidence adds nothing.

Built on: the pan cue follows Avendano's level-based panning index and inter-channel
coherence (C. Avendano, "Frequency-domain source identification and manipulation in stereo
mixes for enhancement, suppression and re-panning applications," WASPAA 2003); the pitch cue
uses pYIN (M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using
probabilistic threshold distributions," ICASSP 2014) through librosa.

Defaults were tuned on real multitrack stems (an independent artist's own sessions, two songs,
60 s each; the lead is recorded mono and centered, backing layers panned around it). SI-SDR in
dB, each estimate against the true stem; "baseline" = all vocals used as the estimate:

| | Lead | Backing |
|---|---|---|
| true vocals, baseline | +2.5 / +7.5 | -2.6 / -7.6 |
| true vocals, first version (pan width 0.25, pan + pitch) | +3.6 / +8.9 | -1.3 / -0.7 |
| true vocals, pan only, width 0.01 | +13.5 / +15.6 | +10.9 / +7.5 |
| Demucs vocals of the full mix, baseline | +1.6 | -3.0 |
| Demucs vocals, first version | +2.4 | -2.1 |
| Demucs vocals, **defaults (pan only, width 0.03)** | **+7.6** | **+6.7** |

(Song A / Song B where two numbers are given; Demucs rows are song A.) The pitch cue lowered
every real-data score -- the backing in these songs is mostly doubles at the lead's pitch, and
pYIN slips onto harmonies -- so it is off for stereo input and kept as the mono fallback. On
Demucs output a slightly wider window (0.03) beats 0.01: separation leaves some stereo residue
on the lead. `lead_pan` is a DAW pan position (constant-power law). A unison double panned with
the lead is out of reach for any of these cues.

### Residue gate and "can't tell" flags (`residue_gate_db`, `confidence`, `uncertain_regions()`)
A mask-based split leaves some of the lead -- and the lead's reverb -- in the backing. Anything
downstream that decides "is there backing here?" relative to the backing's own level then
hears backing almost everywhere. `residue_gate_db` hands it back to the lead: a frame keeps
its backing only when the backing is within that many dB of the lead and, over a beat-length
window (`residue_window`, 0.5 s), within `residue_window_db` (20 dB) -- or when the lead is
quiet (call-and-response, ad-libs in the gaps). The two masks still sum to one. Off by default.

Scored on three finished masters by the same artist (Demucs vocals of the master -> LeadBacking
-> backing split left / right), against the true backing stems: per 0.5 s window, is there
backing (each layer within 40 dB of its own peak)?

| Song | no gate: agree / recall | gate 5 dB + 20 dB window: agree / recall |
|---|---|---|
| A | 40 % / 99 % | 84 % / 83 % |
| B | 59 % / 100 % | 79 % / 91 % |
| C | 50 % / 92 % | 73 % / 70 % |

(4 dB and 6 dB were within a couple of points; 5 dB balanced agreement and recall.) The cost:
real backing mixed more than ~5 dB under the lead *while the lead sings* is handed to the
lead too (song C's recall).

`confidence` (per frame, 0..1) and `uncertain_regions()` (stretches with a reason --
`between_pans`, `decorrelated`, `quiet_backing`, `no_pitch`, `mixed`) say where the separator
couldn't decide. They do what they say on constructed mixtures (tests), but **on the three
masters they are weak error predictors**: the pan-cue certainty is as high where the split is
wrong as where it's right (the errors are confident: reverb and leakage sit clearly off-center),
and after gating the flags caught only 4-12 % of the wrong windows. Use them as a hint, not a
check list. Two other signals were tried and dropped: "the backing is a quieter copy of the
lead" (spectral similarity) scored a harmony a third away above an echo of the lead.
