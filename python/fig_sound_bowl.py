"""Audio to accompany the chapter 4 bowl animation (python/fig_resonance.py).

Uses the animation's own timeline: the "mallet rate" W sweeps from 0.2 to 1.45
(in units of the bowl's natural tone) with the same smoothstep easing over 300
frames at 30 fps, then holds for 45 frames.

Main sound: the bowl rings at its own natural tone F_TONE (with a few overtones)
the whole time, and its loudness follows the damped-resonance curve raised to
AUDIO_POWER (a steeper curve, so it is much quieter away from resonance),
peaking when W = 1. Background cue: a quiet plain tone at W * F_TONE, i.e. the
mallet's rate expressed as a pitch. It glides up through the bowl's tone at
resonance, so the match is heard as a unison. SWEEP_LEVEL = 0 removes it.

Output: audio/bowl_resonance.wav (same length as the video). If the video exists
(site/assets/bowl_resonance.mp4), the audio is also muxed into it.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

import style
import fig_resonance_params as P
from anim import add_audio

F_TONE = 417.0           # the bowl's natural tone, Hz (a small-to-medium singing bowl)
SR = 44100
OVERTONES = [(2.71, 0.25), (5.15, 0.08)]   # soft bowl-like overtones (ratio, relative level)
SWEEP_LEVEL = 0.08       # background mallet-rate tone, relative to the bowl tone at its peak (~ -22 dB)
OUT = style.ROOT / "audio"


def timeline(t):
    frame = np.minimum(t * P.FPS, P.N - 1)
    s = frame / (P.N - 1)
    ease = s * s * (3 - 2 * s)
    return 0.2 + 1.25 * ease


def main():
    dur = (P.N + P.HOLD) / P.FPS
    t = np.arange(int(dur * SR)) / SR
    W = timeline(t)
    A_curve = P.response(W) / P.response(np.linspace(0.05, 1.6, 4000)).max()   # 0..1, the curve in the animation
    A = A_curve ** P.AUDIO_POWER      # steeper: much quieter away from resonance (peak level unchanged)
    f = W * F_TONE                                         # mallet rate as a pitch (background cue)
    bowl_phase = 2 * np.pi * F_TONE * t                    # the bowl always rings at its own tone
    bowl = np.sin(bowl_phase)
    for ratio, level in OVERTONES:
        bowl += level * np.sin(ratio * bowl_phase)
    sweep = np.sin(2 * np.pi * np.cumsum(f) / SR)
    y = A * bowl + SWEEP_LEVEL * sweep
    # gentle fade-in/out so there are no clicks
    n_in, n_out = int(0.2 * SR), int(0.6 * SR)
    y[:n_in] *= np.linspace(0, 1, n_in)
    y[-n_out:] *= np.linspace(1, 0, n_out)
    pcm = (0.6 * y / np.abs(y).max() * 32767).astype(np.int16)

    OUT.mkdir(exist_ok=True)
    wav = OUT / "bowl_resonance.wav"
    wavfile.write(wav, SR, pcm)
    i_pk = np.argmax(A)
    print(f"bowl tone {F_TONE:.0f} Hz constant (+ overtones x{', x'.join(str(r) for r, _ in OVERTONES)}); "
          f"background sweep {f[0]:.0f} -> {f[-1]:.0f} Hz at {SWEEP_LEVEL:.2f} of peak ({20 * np.log10(SWEEP_LEVEL):.0f} dB)")
    print(f"loudest at t = {t[i_pk]:.2f} s (video frame {t[i_pk] * P.FPS:.0f}); sweep passes the bowl tone at W = 1")
    print(f"wrote {wav} ({len(pcm) / SR:.1f} s, {wav.stat().st_size / 1e6:.1f} MB)")

    video = style.ASSETS / "bowl_resonance.mp4"
    if video.exists():
        add_audio(video, wav)

    # check: RMS envelope vs the response curve
    win = int(0.1 * SR)
    rms = np.array([np.sqrt(np.mean(pcm[k:k + win].astype(float) ** 2)) for k in range(0, len(pcm) - win, win)])
    tc = (np.arange(len(rms)) + .5) * win / SR
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(t, A_curve, color=style.INK_SOFT, lw=1.5, ls=":", label="resonance curve (animation)")
    ax.plot(t, A, color=style.INK, lw=2, label=f"target loudness (curve$^{{{P.AUDIO_POWER:g}}}$)")
    ax.plot(tc, rms / rms.max(), color=style.CLAY, lw=2, ls="--", label="audio loudness (RMS)")
    ax.axvline(t[i_pk], color=style.SAND, ls=":")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("relative loudness")
    ax.legend()
    fig.savefig(OUT / "bowl_resonance_envelope.png")
    print("corr(audio RMS, curve) =", round(float(np.corrcoef(rms, np.interp(tc, t, A))[0, 1]), 4))


if __name__ == "__main__":
    main()
