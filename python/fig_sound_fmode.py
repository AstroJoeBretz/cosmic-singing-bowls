"""The star's f-mode as a tone: textbook, nonlinear-shifted, and both together.

Textbook pitch: f_a0 = omega_a0 / (2 pi), from the paper's universal relation
for the primary NS of the baseline system (1.29 + 1.37 Msun).
Nonlinear shift (paper Eq. Delta_wnl): the effective pitch falls as the orbit
speeds up, f_eff = f_a0 * (1 + A_nl * (f_gw / 1 kHz)^2), A_nl = -0.03 for the
canonical star (mass-adjusted for the primary).

The real inspiral takes under a second to sweep this range, so the glide is
slowed down: the orbit's GW frequency follows the chirp law but is stretched to
GLIDE_S seconds. Output: audio/fmode_textbook_nonlinear_combined.wav
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

import style

G, C, MSUN = 6.67430e-11, 299792458.0, 1.98847e30
MC_CANO, W_CANO, A_CANO = 1.33, 0.0786, -0.03
M1, M2 = 1.29, 1.37
F_START, F_END = 300.0, 1553.0      # orbit's GW frequency sweep (Hz)
SR = 44100
STEADY_S, GLIDE_S, GAP_S = 4.0, 8.0, 1.0
OUT = style.ROOT / "audio"

# --- the physics ------------------------------------------------------------
tM1 = G * M1 * MSUN / C ** 3
w_a0 = W_CANO * (M1 / MC_CANO) ** 1.5 / tM1            # rad/s
F_A0 = w_a0 / (2 * np.pi)                              # textbook f-mode, Hz
A_NL = A_CANO * (M2 / M1) * 2 * MC_CANO / (M1 + M2)    # primary's mass-adjusted amplitude


def f_eff(f_gw):
    return F_A0 * (1 + A_NL * (f_gw / 1000.0) ** 2)


def orbit_fgw(t, T):
    """GW frequency vs time: chirp law stretched so F_START -> F_END takes T seconds."""
    # tau ~ f^(-8/3): map normalised time u in [0,1] onto that curve
    a, b = F_START ** (-8 / 3), F_END ** (-8 / 3)
    u = t / T
    return (a + u * (b - a)) ** (-3 / 8)


# --- synthesis ----------------------------------------------------------------
def tone(freq, sr=SR):
    """Constant-amplitude tone following a (possibly time-varying) frequency array."""
    return np.sin(2 * np.pi * np.cumsum(freq) / sr)


def envelope(n, fade=0.15, sr=SR):
    k = int(fade * sr)
    e = np.ones(n)
    e[:k] = np.sin(np.linspace(0, np.pi / 2, k)) ** 2
    e[-k:] = np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    return e


def main():
    t_s = np.arange(int(STEADY_S * SR)) / SR
    t_g = np.arange(int(GLIDE_S * SR)) / SR
    gap = np.zeros(int(GAP_S * SR))

    f_steady = np.full_like(t_s, F_A0)
    f_glide = f_eff(orbit_fgw(t_g, GLIDE_S))

    seg1 = tone(f_steady) * envelope(len(t_s))
    seg2 = tone(f_glide) * envelope(len(t_g))
    seg3 = 0.5 * (tone(np.full_like(t_g, F_A0)) + tone(f_glide)) * envelope(len(t_g))

    h = np.concatenate([seg1, gap, seg2, gap, seg3])
    pcm = (0.5 * h / np.abs(h).max() * 32767).astype(np.int16)     # moderate level: 1-2 kHz is piercing
    OUT.mkdir(exist_ok=True)
    wav = OUT / "fmode_textbook_nonlinear_combined.wav"
    wavfile.write(wav, SR, pcm)
    print(f"textbook f-mode {F_A0:.1f} Hz; A_nl(primary) {A_NL:.4f}; "
          f"nonlinear pitch ends at {f_glide[-1]:.1f} Hz (shift {f_glide[-1] - F_A0:+.1f} Hz)")
    print(f"wrote {wav} ({len(pcm) / SR:.1f} s: steady {STEADY_S:.0f} s | glide {GLIDE_S:.0f} s | both {GLIDE_S:.0f} s)")

    # check + picture
    starts = [0, STEADY_S + GAP_S, STEADY_S + GLIDE_S + 2 * GAP_S]
    fr, tt, S = spectrogram(h, SR, nperseg=8192, noverlap=7168)
    for name, t0, dur in [("textbook", starts[0], STEADY_S), ("nonlinear", starts[1], GLIDE_S)]:
        for frac in (0.1, 0.5, 0.95):
            i = np.argmin(abs(tt - (t0 + frac * dur)))
            m = (fr > 1000) & (fr < 2200)
            print(f"  {name:9s} at {frac:4.0%}: measured {fr[m][np.argmax(S[m, i])]:7.1f} Hz")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    sel = (fr > 1300) & (fr < 2100)
    ax.pcolormesh(tt, fr[sel], 10 * np.log10(S[sel] + 1e-20), vmin=-110, vmax=-20, cmap="magma_r", shading="auto")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz]")
    for x0, lab in zip(starts, ["textbook", "nonlinear", "both"]):
        ax.text(x0 + 0.1, 2060, lab, color=style.INK, fontsize=11, va="top")
    fig.savefig(OUT / "fmode_spectrogram.png")
    print("wrote", OUT / "fmode_spectrogram.png")


if __name__ == "__main__":
    main()
