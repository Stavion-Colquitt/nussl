import numpy as np
import librosa

from .. import MaskSeparationBase, SeparationException


class LeadBacking(MaskSeparationBase):
    """
    Splits a vocals-only recording into the **lead vocal** and the **backing vocals**,
    using the cues mix engineers leave behind:

    * **Pan.** The lead usually sits at one pan position (the center, by default) while
      backing parts -- doubles, harmonies, ad-libs -- are spread left and right. Each
      time-frequency bin gets a score for how close its level-based pan index is to the
      lead's position, scaled by inter-channel phase coherence (panned copies of one
      source are coherent; reverb and decorrelated doubles are not).
    * **Pitch.** The predominant f0 of the center (mid) signal is tracked with pYIN and
      the lead's harmonic comb is scored per bin: a harmony a third or a fifth away
      falls between the lead's partials. Unvoiced frames carry no pitch evidence.
    * **Level.** Optionally, bins much quieter than the loudest bin in their frame lean
      toward backing (backing vocals are usually mixed under the lead).

    The cues are combined as weighted log-odds into one lead score in [0, 1] (a cue with no
    evidence, 0.5, contributes nothing); the lead mask is that score and the backing mask
    its complement (soft), or the larger of the two (binary).

    Limits: a unison double panned to the same place as the lead, at the same pitch, is
    invisible to every cue here and stays with the lead. A mono input has no pan cue.

    Run a vocals / accompaniment separator first (e.g. Demucs) -- this expects vocals only.

    Defaults were tuned on real multitrack stems (two songs, 60 s each: true lead + backing
    summed, and the Demucs vocals of the full mix); see CHANGES_FORK.md for the numbers.

    Parameters:
        input_audio_signal (AudioSignal): vocals to split (stereo for the pan cue).
        lead_pan (float): pan position of the lead, -1 (left) .. 0 (center) .. 1 (right).
        pan_width (float): how far from ``lead_pan`` a bin can sit and still count as lead.
          Default 0.03: a lead recorded mono and placed in the center has a pan index of
          ~0 even after vocal separation, so anything off-center is backing.
        pan_weight, pitch_weight, level_weight (float): relative weight of each cue.
          ``pitch_weight=None`` (default) means 0 for stereo input and 1 for mono: on real
          mixes the pitch cue mostly costs (doubles share the lead's pitch, pYIN slips onto
          harmonies), so it is the fallback for mono only.
        fmin, fmax (float): f0 search range for the lead (Hz).
        n_harmonics (int): partials of the lead's comb.
        harmonic_width (float): tolerance around each partial, in cents.
        level_floor_db (float): bins this far below their frame's loudest bin count fully
          toward backing on the level cue.
        mask_type, mask_threshold: see :class:`MaskSeparationBase`.

    Attributes after ``run()``:
        f0 (np.ndarray): lead f0 per frame (nan when unvoiced).
        pan_index (np.ndarray): per-bin pan index, -1 .. 1 (stereo input only).
        lead_score (np.ndarray): combined score per bin (freq x time).
    """

    def __init__(self, input_audio_signal, lead_pan=0.0, pan_width=0.03, pan_weight=1.0,
                 pitch_weight=None, level_weight=0.0, fmin=70.0, fmax=1000.0, n_harmonics=20,
                 harmonic_width=50.0, level_floor_db=30.0, mask_type='soft', mask_threshold=0.5):
        super().__init__(input_audio_signal=input_audio_signal, mask_type=mask_type,
                         mask_threshold=mask_threshold)
        if pitch_weight is None:   # auto: pan beats pitch on real stereo mixes; mono has only pitch
            pitch_weight = 0.0 if self.audio_signal.num_channels >= 2 else 1.0
        if not -1.0 <= lead_pan <= 1.0:
            raise SeparationException('lead_pan must be between -1 (left) and 1 (right)')
        if pan_width <= 0 or harmonic_width <= 0:
            raise SeparationException('pan_width and harmonic_width must be positive')
        if pan_weight + pitch_weight + level_weight <= 0:
            raise SeparationException('at least one cue needs a positive weight')
        self.lead_pan, self.pan_width = lead_pan, pan_width
        self.pan_weight, self.pitch_weight, self.level_weight = pan_weight, pitch_weight, level_weight
        self.fmin, self.fmax, self.n_harmonics = fmin, fmax, n_harmonics
        self.harmonic_width, self.level_floor_db = harmonic_width, level_floor_db
        self.f0 = self.pan_index = self.lead_score = None

    # ---- cues -------------------------------------------------------------------------
    def pan_cue(self):
        """Per-bin closeness to the lead's pan position (freq x time), or None for mono."""
        if self.audio_signal.num_channels < 2:
            return None
        L, R = self.stft[..., 0], self.stft[..., 1]
        aL, aR = np.abs(L), np.abs(R)
        self.pan_index = (aR - aL) / (aL + aR + 1e-12)               # -1 left .. +1 right
        # lead_pan is a DAW pan position (constant-power law); this is its level-based pan index
        a = (self.lead_pan + 1) * np.pi / 4
        target = (np.sin(a) - np.cos(a)) / (np.sin(a) + np.cos(a))
        closeness = np.exp(-0.5 * ((self.pan_index - target) / self.pan_width) ** 2)
        coherence = np.clip(np.cos(np.angle(L) - np.angle(R)), 0.0, 1.0)
        return closeness * (0.5 + 0.5 * coherence)

    def pitch_cue(self):
        """Per-bin closeness to the lead's harmonic comb (freq x time); 0.5 when unvoiced."""
        sr = self.audio_signal.sample_rate
        params = self.audio_signal.stft_params
        mid = self.audio_signal.audio_data.mean(axis=0)
        f0, voiced, _ = librosa.pyin(
            mid, fmin=self.fmin, fmax=self.fmax, sr=sr,
            frame_length=max(params.window_length, 2048), hop_length=params.hop_length, center=True)
        n_frames = self.stft.shape[1]
        f0 = f0[:n_frames] if len(f0) >= n_frames else np.pad(f0, (0, n_frames - len(f0)), constant_values=np.nan)
        self.f0 = f0
        freqs = librosa.fft_frequencies(sr=sr, n_fft=params.window_length)[:self.stft.shape[0]]
        bin_hz = sr / params.window_length
        cue = np.full((len(freqs), n_frames), 0.5)
        k = np.arange(1, self.n_harmonics + 1)
        for t in np.where(np.isfinite(f0))[0]:
            partials = f0[t] * k
            # tolerance: harmonic_width cents, but never narrower than the STFT's main lobe
            tol = np.maximum(partials * (2 ** (self.harmonic_width / 1200) - 1), 1.5 * bin_hz)
            z = np.abs(freqs[:, None] - partials[None, :]) / tol[None, :]
            cue[:, t] = np.exp(-0.5 * z.min(axis=1) ** 2)
        cue[freqs < self.fmin * 0.9, :] = 0.5                          # below the lead: no evidence
        cue[freqs > f0[np.isfinite(f0)].max() * (self.n_harmonics + 0.5) if np.isfinite(f0).any() else freqs > np.inf, :] = 0.5
        return cue

    def level_cue(self):
        """1 near the loudest bin of each frame, 0 at ``level_floor_db`` below it."""
        db = 20 * np.log10(np.abs(self.stft).mean(axis=-1) + 1e-12)
        rel = db - db.max(axis=0, keepdims=True)
        return np.clip(1 + rel / self.level_floor_db, 0.0, 1.0)

    # ---- separation ---------------------------------------------------------------------
    def run(self):
        cues = []
        pan = self.pan_cue()
        if pan is not None and self.pan_weight > 0:
            cues.append((self.pan_weight, pan))
        if self.pitch_weight > 0:
            cues.append((self.pitch_weight, self.pitch_cue()))
        if self.level_weight > 0:
            cues.append((self.level_weight, self.level_cue()))
        if not cues:
            raise SeparationException('no usable cue: mono input with pitch and level turned off')
        # combine as weighted log-odds: a cue at 0.5 ("no evidence") adds nothing, and two
        # confident cues reinforce each other instead of averaging out
        total = sum(w for w, _ in cues)
        eps = 1e-4
        logit = sum(w * np.log((np.clip(c, eps, 1 - eps)) / (1 - np.clip(c, eps, 1 - eps)))
                    for w, c in cues) / total
        self.lead_score = 1.0 / (1.0 + np.exp(-logit))

        lead = np.repeat(self.lead_score[..., None], self.audio_signal.num_channels, axis=-1)
        backing = 1.0 - lead
        if self.mask_type == self.MASKS['binary']:
            lead, backing = lead >= backing, backing > lead
        self.result_masks = [self.mask_type(lead), self.mask_type(backing)]
        return self.result_masks
