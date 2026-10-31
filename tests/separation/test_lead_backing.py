import numpy as np
import pytest
import nussl
from nussl.separation import spatial, SeparationException

SR = 22050


def voice(f0, dur=2.0, vibrato=5.0, n=12, seed=0):
    """A sung-vowel stand-in: harmonic tone with vibrato and a soft attack."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    inst_f = f0 * (1 + 0.01 * np.sin(2 * np.pi * vibrato * t + rng.uniform(0, 6.28)))
    phase = 2 * np.pi * np.cumsum(inst_f) / SR
    y = sum(np.sin(k * phase) / k for k in range(1, n + 1))
    return y * np.minimum(1, t / 0.05)


def si_sdr(est, ref):
    est, ref = est.ravel(), ref.ravel()
    n = min(len(est), len(ref)); est, ref = est[:n], ref[:n]
    a = np.dot(est, ref) / np.dot(ref, ref)
    return 10 * np.log10(np.sum((a * ref) ** 2) / np.sum((a * ref - est) ** 2))


def make(lead_pan=0.0, harmony_ratio=2 ** (4 / 12), back_pans=(-0.8, 0.8), mono=False):
    """Lead + backing parts; each backing pan position is its own take (own vibrato, a few
    cents off), like real stacked backing vocals -- one signal panned to both sides would
    just be a phantom center. Default harmony: a major third."""
    lead = voice(220.0, seed=1)
    def pan(y, p):  # constant-power pan, p in -1..1
        a = (p + 1) * np.pi / 4
        return np.stack([np.cos(a) * y, np.sin(a) * y])
    lead_st = pan(lead, lead_pan)
    takes = [voice(220.0 * harmony_ratio * 2 ** (c / 1200), seed=2 + i) * 0.5
             for i, c in enumerate([-8, 6, -3, 4][:len(back_pans)])]
    back_st = sum(pan(y, p) for y, p in zip(takes, back_pans)) / len(back_pans)
    if mono:
        lead_st, back_st = lead_st.mean(0, keepdims=True), back_st.mean(0, keepdims=True)
    mix = nussl.AudioSignal(audio_data_array=lead_st + back_st, sample_rate=SR)
    return mix, lead_st, back_st


def improvement(est, ref, mix):
    return si_sdr(est.audio_data, ref) - si_sdr(mix.audio_data, ref)


@pytest.mark.parametrize('mask_type', ['soft', 'binary'])
def test_lead_backing_pan_and_pitch(mask_type):
    mix, lead, back = make()
    sep = spatial.LeadBacking(mix, mask_type=mask_type, pan_width=0.25, pitch_weight=1.0)
    sep.run()
    est_lead, est_back = sep.make_audio_signals()
    assert improvement(est_lead, lead, mix) > (4 if mask_type == 'soft' else 3)
    assert improvement(est_back, back, mix) > (10 if mask_type == 'soft' else 6)
    assert np.nanmedian(sep.f0) == pytest.approx(220, rel=0.05)


def test_lead_backing_pitch_only_mono():
    mix, lead, back = make(mono=True)
    sep = spatial.LeadBacking(mix)
    sep.run()
    est_lead, est_back = sep.make_audio_signals()
    assert sep.pan_index is None
    assert improvement(est_lead, lead, mix) > 5
    assert improvement(est_back, back, mix) > 10


def test_lead_backing_defaults_stereo_are_pan_only():
    # defaults on stereo: narrow pan window, pitch off (tuned on real stems)
    mix, lead, back = make()
    sep = spatial.LeadBacking(mix)
    sep.run()
    est_lead, est_back = sep.make_audio_signals()
    assert sep.pitch_weight == 0 and sep.f0 is None
    assert improvement(est_lead, lead, mix) > 2
    assert improvement(est_back, back, mix) > 8


def test_lead_backing_defaults_mono_use_pitch():
    mix, _, _ = make(mono=True)
    assert spatial.LeadBacking(mix).pitch_weight == 1.0


def test_lead_backing_off_center_lead():
    mix, lead, back = make(lead_pan=0.5, back_pans=(-0.9,))
    sep = spatial.LeadBacking(mix, lead_pan=0.5)
    sep.run()
    est_lead, _ = sep.make_audio_signals()
    assert improvement(est_lead, lead, mix) > 4


def test_lead_backing_masks_sum_to_one():
    mix, _, _ = make()
    sep = spatial.LeadBacking(mix, level_weight=0.5)
    lead_m, back_m = sep.run()
    assert np.allclose(lead_m.mask + back_m.mask, 1.0)
    assert lead_m.mask.shape == sep.stft.shape


def test_lead_backing_bad_args():
    mix, _, _ = make()
    pytest.raises(SeparationException, spatial.LeadBacking, mix, lead_pan=2)
    pytest.raises(SeparationException, spatial.LeadBacking, mix, pan_width=0)
    pytest.raises(SeparationException, spatial.LeadBacking, mix, pan_weight=0, pitch_weight=0)
    mono, _, _ = make(mono=True)
    sep = spatial.LeadBacking(mono, pitch_weight=0)
    pytest.raises(SeparationException, sep.run)
