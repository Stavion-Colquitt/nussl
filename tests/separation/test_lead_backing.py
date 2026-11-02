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


def _pan(y, p):
    a = (p + 1) * np.pi / 4
    return np.stack([np.cos(a) * y, np.sin(a) * y])


def _unsure_mix():
    """0-1 s: lead in the center + backing at +/-0.8 (clear). 1-2 s: one part panned
    just off the lead (pan index ~1.2 x pan_width -- can't be told apart).
    2-2.5 s: silence (nothing to flag)."""
    clear, _, _ = make()
    clear = clear.audio_data[:, :SR]
    vague = _pan(voice(330.0, dur=1.0, seed=7), 0.045)
    silence = np.zeros((2, SR // 2))
    return nussl.AudioSignal(audio_data_array=np.concatenate([clear, vague, silence], axis=1),
                             sample_rate=SR)


def test_lead_backing_confidence_and_uncertain_regions():
    sep = spatial.LeadBacking(_unsure_mix())
    sep.run()
    t = sep.frame_times()
    conf = sep.confidence
    assert conf.shape == t.shape
    assert np.nanmedian(conf[(t > 0.2) & (t < 0.8)]) > 0.7          # clear part: confident
    assert np.nanmedian(conf[(t > 1.2) & (t < 1.8)]) < 0.3          # ambiguous part: unsure
    assert np.isnan(conf[(t > 2.15) & (t < 2.4)]).all()             # silence: not judged
    regions = sep.uncertain_regions()
    assert len(regions) == 1
    r = regions[0]
    assert r['start'] == pytest.approx(1.0, abs=0.15)
    assert r['end'] == pytest.approx(2.0, abs=0.15)
    assert r['confidence'] < 0.3
    assert r['reason'] == 'between_pans'


def test_lead_backing_decorrelated_center_is_flagged():
    # two independent takes, one per side, both at the lead's level-based center:
    # pan index ~0 but low coherence (what reverb or wide doubles look like)
    a, b = voice(220.0, dur=1.5, seed=3), voice(220.0 * 2 ** (7 / 1200), dur=1.5, seed=4)
    sig = nussl.AudioSignal(audio_data_array=np.stack([a, b]) * 0.5, sample_rate=SR)
    sep = spatial.LeadBacking(sig)
    sep.run()
    regions = sep.uncertain_regions()
    assert regions and regions[0]['reason'] == 'decorrelated'


def test_lead_backing_clear_mix_has_no_flags():
    mix, _, _ = make()
    sep = spatial.LeadBacking(mix)
    sep.run()
    assert sep.uncertain_regions() == []


def test_lead_backing_uncertain_regions_needs_run():
    mix, _, _ = make()
    pytest.raises(SeparationException, spatial.LeadBacking(mix).uncertain_regions)


def _residue_mix():
    """0-1.5 s: lead alone, plus a faint (-24 dB) off-center copy of it -- what a separator's
    leakage / the lead's reverb looks like. 1.5-3 s: lead + a real harmony at +/-0.8."""
    lead = voice(220.0, dur=3.0, seed=1)
    rng = np.random.default_rng(0)
    leak = rng.standard_normal(len(lead))                        # diffuse, like reverb
    leak *= 10 ** (-24 / 20) * np.std(lead) / np.std(leak)
    n = len(lead) // 2
    leak[n:] = 0
    harm = 2 * voice(220.0 * 2 ** (4 / 12), dur=3.0, seed=5)    # at about the lead's level
    harm[:n] = 0
    mix = _pan(lead, 0.0) + _pan(leak, 0.7) + 0.5 * (_pan(harm, -0.8) + _pan(np.roll(harm, 300), 0.8))
    return nussl.AudioSignal(audio_data_array=mix, sample_rate=SR), n / SR


def test_lead_backing_residue_gate():
    mix, split = _residue_mix()
    plain = spatial.LeadBacking(mix)
    plain.run()
    gated = spatial.LeadBacking(mix, residue_gate_db=5)
    lead_m, back_m = gated.run()
    assert np.allclose(lead_m.mask + back_m.mask, 1.0)
    t = gated.frame_times()
    first, second = (t > 0.4) & (t < split - 0.5), (t > split + 0.3) & (t < 2.8)
    _, b_plain = plain.make_audio_signals()
    _, b_gated = gated.make_audio_signals()
    sr = mix.sample_rate
    rms = lambda x, a, b: np.sqrt(np.mean(x[:, int(a * sr):int(b * sr)] ** 2))
    # leakage alone: the gate hands it back to the lead
    assert gated.backing_kept[first].mean() < 0.1
    assert rms(b_gated.audio_data, 0.2, split - 0.3) < 0.2 * rms(b_plain.audio_data, 0.2, split - 0.3)
    # a real harmony: kept
    assert gated.backing_kept[second].mean() > 0.85
    assert rms(b_gated.audio_data, split + 0.3, 2.8) > 0.9 * rms(b_plain.audio_data, split + 0.3, 2.8)


def test_lead_backing_quiet_backing_is_flagged():
    # a harmony mixed ~14 dB under a lead that keeps stopping: the gate keeps it in the
    # lead's gaps, but over a beat it sits far under the lead -- as likely the lead's own
    # leakage / reverb as a real quiet part, so it is flagged rather than trusted
    lead = voice(220.0, dur=3.0, seed=1)
    t = np.arange(len(lead)) / SR
    lead *= (np.sin(2 * np.pi * 1.0 * t) > -0.5)                  # sings 2/3 of each second
    # two takes of the harmony, one per side (one signal panned both ways is a phantom center)
    takes = [voice(220.0 * 2 ** (4 / 12) * 2 ** (c / 1200), dur=3.0, seed=k) for c, k in ((-6, 5), (5, 6))]
    harm = 0.5 * (_pan(takes[0], -0.8) + _pan(takes[1], 0.8))
    mix = nussl.AudioSignal(audio_data_array=_pan(lead, 0.0) + 10 ** (-14 / 20) * harm, sample_rate=SR)
    sep = spatial.LeadBacking(mix, residue_gate_db=5)
    sep.run()
    rel = sep.backing_rel_db
    assert np.isfinite(rel).any() and np.nanmedian(rel) < -8
    regions = sep.uncertain_regions()
    assert regions and regions[0]['reason'] == 'quiet_backing'
    assert sum(r['end'] - r['start'] for r in regions) > 0.6     # the lead's three gaps
    # the same harmony up at the lead's level is trusted
    loud = nussl.AudioSignal(audio_data_array=_pan(lead, 0.0) + harm, sample_rate=SR)
    sep = spatial.LeadBacking(loud, residue_gate_db=5)
    sep.run()
    assert sep.uncertain_regions() == []


def test_lead_backing_residue_gate_bad_arg():
    mix, _, _ = make()
    pytest.raises(SeparationException, spatial.LeadBacking, mix, residue_gate_db=0)


# ---- level rule: the louder part is the lead ----------------------------------------------
def _double_tracked_lead_mix(center_db=-10.0):
    """The lead panned well off center, with a quieter harmony in the center: pan alone
    calls the center part the lead. (A lead double-tracked hard L/R is NOT a clean test:
    two near-identical takes share bins and read as a phantom center, so pan already keeps
    much of it as lead.)"""
    lead_st = _pan(voice(220.0, seed=11), -0.9)
    center = _pan(voice(220.0 * 2 ** (4 / 12), seed=13), 0.0) * 10 ** (center_db / 20)
    mix = nussl.AudioSignal(audio_data_array=lead_st + center, sample_rate=SR)
    return mix, lead_st, center


def test_lead_backing_level_rule_swaps_a_louder_backing():
    mix, lead, center = _double_tracked_lead_mix()
    off = spatial.LeadBacking(mix); off.run()
    on = spatial.LeadBacking(mix, level_rule_db=6.0); on.run()
    off_lead, _ = off.make_audio_signals()
    on_lead, on_back = on.make_audio_signals()
    assert np.mean(on.swapped) > 0.8                       # swapped nearly everywhere
    assert not off.swapped.any()                           # rule off: nothing swapped
    # with the rule the lead estimate is the loud double-tracked lead, not the center harmony
    assert si_sdr(on_lead.audio_data, lead) > si_sdr(off_lead.audio_data, lead) + 10
    assert si_sdr(on_back.audio_data, center) > 5


def test_lead_backing_level_rule_leaves_a_normal_mix_alone():
    mix, lead, back = make()                               # centered lead, quieter panned backing
    off = spatial.LeadBacking(mix); off.run()
    on = spatial.LeadBacking(mix, level_rule_db=6.0); on.run()
    assert np.mean(on.swapped) < 0.05
    assert np.allclose(on.lead_score, off.lead_score, atol=0.05)


def test_lead_backing_level_rule_keeps_a_lone_backing_part_backing():
    """1 s of backing alone (panned), then 1 s of a centered lead over it: the lone backing
    is louder than the (silent) lead's leakage but must not be swapped to lead."""
    back = _pan(voice(330.0, seed=21), -0.8) * 0.5 + _pan(voice(331.0, seed=22), 0.8) * 0.5
    lead = _pan(voice(220.0, seed=23), 0.0)
    half = back.shape[1] // 2
    lead[:, :half] = 0.0                                   # the lead only sings in the second half
    mix = nussl.AudioSignal(audio_data_array=lead + back, sample_rate=SR)
    sep = spatial.LeadBacking(mix, level_rule_db=6.0); sep.run()
    t = sep.frame_times()
    first = t < (half / SR) - 0.3                          # clear of the window around the entry
    assert np.mean(sep.swapped[first]) < 0.05
    est_lead, est_back = sep.make_audio_signals()
    assert si_sdr(est_back.audio_data, back) > 5


def test_lead_backing_level_rule_bad_arg():
    mix, _, _ = make()
    with pytest.raises(SeparationException):
        spatial.LeadBacking(mix, level_rule_db=0)
