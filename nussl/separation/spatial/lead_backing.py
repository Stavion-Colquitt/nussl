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

    References:
        The pan cue builds on the level-based panning index and inter-channel coherence of
        Avendano's frequency-domain stereo source identification; the pitch cue tracks the
        lead with pYIN (via librosa).

        C. Avendano, "Frequency-domain source identification and manipulation in stereo
        mixes for enhancement, suppression and re-panning applications," IEEE Workshop on
        Applications of Signal Processing to Audio and Acoustics (WASPAA), 2003.

        M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using probabilistic
        threshold distributions," IEEE International Conference on Acoustics, Speech and
        Signal Processing (ICASSP), 2014.

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
        residue_gate_db (float or None): hand leftover lead back to the lead. A mask-based
          split leaves some of the lead (and its reverb) in the backing, and anything that
          thresholds "is backing present" relative to the backing's own level will read that
          residue as backing almost everywhere. With a gate, a frame keeps its backing only
          when the backing is within ``residue_gate_db`` of the lead there AND, over a
          ``residue_window``-second window around it, within ``residue_window_db`` -- or when
          the lead is quiet (call-and-response, ad-libs in the gaps). Gated-out backing goes
          to the lead mask, so the two masks still sum to one. ``None`` (default): off.
          5 dB worked best on three real masters scored against their true stems.
        residue_window (float), residue_window_db (float): the windowed half of the gate.
        mask_type, mask_threshold: see :class:`MaskSeparationBase`.

    Attributes after ``run()``:
        f0 (np.ndarray): lead f0 per frame (nan when unvoiced).
        pan_index (np.ndarray): per-bin pan index, -1 .. 1 (stereo input only).
        lead_score (np.ndarray): combined score per bin (freq x time).
        confidence (np.ndarray): per frame, how sure the split is, 0 .. 1 -- the
          energy-weighted mean of ``|2 * lead_score - 1|`` over the frame's bins (1: every
          loud bin is clearly lead or clearly backing; 0: every loud bin is a coin flip).
          ``nan`` for silent frames, which have nothing to separate. Also lowered where
          the backing that is left sits far under the lead (``backing_rel_db`` between
          ``QUIET_HIGH_DB`` and ``QUIET_LOW_DB``): on real mixes that, more than a borderline
          pan, is where the split goes wrong.
        backing_rel_db (np.ndarray): per frame, backing minus lead level in dB over
          ``residue_window`` (``nan`` where no backing is kept, or silence).
        backing_kept (np.ndarray): per frame, 0 .. 1, how much backing the residue gate kept
          (all ones with the gate off).

    Knowing where it failed: :meth:`uncertain_regions` lists the stretches where the
    separator could not tell lead from backing (e.g. a part panned just off the lead,
    reverb or decorrelated doubles at the lead's position, or unvoiced mono passages),
    so a person -- or a downstream labeler -- can check them instead of trusting them.
    """

    def __init__(self, input_audio_signal, lead_pan=0.0, pan_width=0.03, pan_weight=1.0,
                 pitch_weight=None, level_weight=0.0, fmin=70.0, fmax=1000.0, n_harmonics=20,
                 harmonic_width=50.0, level_floor_db=30.0, residue_gate_db=None,
                 residue_window=0.5, residue_window_db=20.0, mask_type='soft', mask_threshold=0.5):
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
        if residue_gate_db is not None and residue_gate_db <= 0:
            raise SeparationException('residue_gate_db must be positive (or None for off)')
        self.residue_gate_db = residue_gate_db
        self.residue_window, self.residue_window_db = residue_window, residue_window_db
        self.backing_kept = self.backing_rel_db = None
        self.f0 = self.pan_index = self.lead_score = self.confidence = None
        self._coherence = None

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
        self._coherence = coherence
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
        self._mask_score = self.lead_score            # before gating: how sure the cues are
        self.backing_kept = np.ones(self.lead_score.shape[1])
        if self.residue_gate_db:
            self.backing_kept = self._residue_keep()
            self.lead_score = self.lead_score + (1.0 - self.lead_score) * (1.0 - self.backing_kept)[None, :]
        self.backing_rel_db = self._backing_rel_db()
        self.confidence = self._frame_confidence()

        lead = np.repeat(self.lead_score[..., None], self.audio_signal.num_channels, axis=-1)
        backing = 1.0 - lead
        if self.mask_type == self.MASKS['binary']:
            lead, backing = lead >= backing, backing > lead
        self.result_masks = [self.mask_type(lead), self.mask_type(backing)]
        return self.result_masks

    # ---- knowing where it is unsure ---------------------------------------------------
    def _frame_energy(self):
        return (np.abs(self.stft) ** 2).sum(axis=-1)                 # freq x time

    def _frame_confidence(self, silence_db=-50.0):
        energy = self._frame_energy()
        frame_e = energy.sum(axis=0)
        certainty = np.abs(2.0 * self._mask_score - 1.0)
        conf = (energy * certainty).sum(axis=0) / (frame_e + 1e-20)
        # backing left far under the lead: the lead's leakage / reverb, or a real quiet part
        rel = self.backing_rel_db
        quiet = np.clip((rel - self.QUIET_LOW_DB) / (self.QUIET_HIGH_DB - self.QUIET_LOW_DB), 0.0, 1.0)
        conf = conf * np.where(np.isfinite(rel), quiet, 1.0)
        loud = 10 * np.log10(frame_e + 1e-20)
        conf[loud < loud.max() - abs(silence_db)] = np.nan          # silence: nothing to split
        return conf

    # windowed backing-vs-lead level where backing is kept: confidence is full above
    # QUIET_HIGH_DB and drops to 0 at QUIET_LOW_DB (on three real masters, backing labeled
    # wrongly sat ~-30 dB under the lead over a beat; right ones between -10 and +4 dB)
    QUIET_LOW_DB, QUIET_HIGH_DB = -20.0, -8.0

    def _lead_backing_energy(self, score):
        mag2 = np.abs(self.stft) ** 2                                   # freq x time x ch
        s = score[..., None]
        return (mag2 * s ** 2).sum(axis=(0, 2)), (mag2 * (1 - s) ** 2).sum(axis=(0, 2))

    def _residue_keep(self, gate_db=None, window_db=None, lead_quiet_db=-45.0, smooth=0.1):
        """Per frame, 0 .. 1: how much of the backing to keep (see ``residue_gate_db``)."""
        from scipy.ndimage import uniform_filter1d
        gate_db = self.residue_gate_db if gate_db is None else gate_db
        window_db = self.residue_window_db if window_db is None else window_db
        hop_s = self.audio_signal.stft_params.hop_length / self.audio_signal.sample_rate
        el, eb = self._lead_backing_energy(self.lead_score)
        quiet = el < el.max() * 10 ** (lead_quiet_db / 10)
        keep = (eb >= el * 10 ** (-gate_db / 10)) | quiet
        n = max(1, int(round(self.residue_window / hop_s)))
        elw, ebw = uniform_filter1d(el, n), uniform_filter1d(eb, n)
        keep &= (ebw >= elw * 10 ** (-window_db / 10)) | (elw < el.max() * 10 ** (lead_quiet_db / 10))
        return np.clip(uniform_filter1d(keep.astype(float), max(1, int(round(smooth / hop_s)))), 0, 1)

    def _backing_rel_db(self):
        """Per frame: backing minus lead level (dB) over ``residue_window``, where the
        backing is kept; nan elsewhere (no backing, or silence)."""
        from scipy.ndimage import uniform_filter1d
        hop_s = self.audio_signal.stft_params.hop_length / self.audio_signal.sample_rate
        el, eb = self._lead_backing_energy(self.lead_score)
        n = max(1, int(round(self.residue_window / hop_s)))
        elw, ebw = uniform_filter1d(el, n), uniform_filter1d(eb, n)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = 10 * np.log10(ebw + 1e-20) - 10 * np.log10(elw + 1e-20)
        total = el + eb
        rel[(self.backing_kept < 0.5) | (total < total.max() * 1e-5)] = np.nan
        return rel

    def frame_times(self):
        """Time (s) of each STFT frame."""
        hop = self.audio_signal.stft_params.hop_length
        n = self.stft.shape[1]
        return np.arange(n) * hop / self.audio_signal.sample_rate

    def uncertain_regions(self, threshold=0.5, min_duration=0.25, max_gap=0.1):
        """
        Stretches where the split could not be decided -- flag these for a human.

        Args:
            threshold (float): frames whose :attr:`confidence` is below this are unsure.
            min_duration (float): drop unsure stretches shorter than this (s).
            max_gap (float): merge unsure stretches separated by less than this (s).

        Returns:
            list of dict: ``start``, ``end`` (s), ``confidence`` (mean over the stretch),
            and ``reason``, one of

            * ``'between_pans'`` -- energy sits between the lead's pan position and the
              backing's (a part panned just off the lead, or the lead itself drifting);
            * ``'decorrelated'`` -- energy at the lead's position but with little
              left/right coherence (reverb, or decorrelated doubles on the lead);
            * ``'quiet_backing'`` -- backing left far under the lead: the lead's own
              leakage or reverb, or a real part mixed low (a mix alone can't tell which);
            * ``'no_pitch'`` -- mono input and the lead's pitch was not tracked;
            * ``'mixed'`` -- the cues disagree with each other.
        """
        if self.lead_score is None:
            raise SeparationException('call run() before uncertain_regions()')
        conf = self.confidence
        times = self.frame_times()
        hop_s = times[1] - times[0] if len(times) > 1 else 0.0
        unsure = np.nan_to_num(conf, nan=1.0) < threshold
        runs, start = [], None
        for i, u in enumerate(unsure):
            if u and start is None:
                start = i
            elif not u and start is not None:
                runs.append([start, i]); start = None
        if start is not None:
            runs.append([start, len(unsure)])
        merged = []
        for a, b in runs:                                            # merge short gaps
            if merged and (a - merged[-1][1]) * hop_s < max_gap:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        regions = []
        for a, b in merged:
            if (b - a) * hop_s < min_duration:
                continue
            regions.append({
                'start': float(times[a]),
                'end': float(times[b - 1] + hop_s),
                'confidence': float(np.nanmean(conf[a:b])) if np.isfinite(conf[a:b]).any() else 0.0,
                'reason': self._reason(a, b),
            })
        return regions

    def _reason(self, a, b):
        energy = self._frame_energy()[:, a:b]
        w = energy / (energy.sum() + 1e-20)
        rel = self.backing_rel_db[a:b]
        if np.isfinite(rel).any() and np.nanmedian(rel) < self.QUIET_HIGH_DB:
            return 'quiet_backing'            # what's left of the backing is far under the lead
        if self.pan_index is None:
            if self.f0 is not None and not np.isfinite(self.f0[a:b]).any():
                return 'no_pitch'
            return 'mixed'
        a_ = (self.lead_pan + 1) * np.pi / 4
        target = (np.sin(a_) - np.cos(a_)) / (np.sin(a_) + np.cos(a_))
        dist = np.abs(self.pan_index[:, a:b] - target)
        coh = self._coherence[:, a:b]
        near = (w * (dist < 2 * self.pan_width)).sum()
        if near > 0.5 and (w * coh).sum() < 0.6:
            return 'decorrelated'
        if (w * ((dist >= 0.5 * self.pan_width) & (dist < 4 * self.pan_width))).sum() > 0.5:
            return 'between_pans'
        return 'mixed'
