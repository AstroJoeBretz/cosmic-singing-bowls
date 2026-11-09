"""What a binary neutron star inspiral sounds like.

Gravitational waves from the inspiral are in the audio band, so we can play
the strain directly. Leading-order (Newtonian) chirp for the paper's baseline
system (M1 = 1.29, M2 = 1.37 solar masses; chirp mass 1.157), starting at
30 Hz and stopping at contact, f_gw = 1553 Hz (paper Eq. fc). Amplitude
grows as f^(2/3). No tides, no post-merger remnant: the signal simply ends
at contact. Written to audio/ (not part of the site yet).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.optimize import brentq
from scipy.signal import spectrogram

import style
from anim import add_audio
from fig_inspiral_params import DURATION_S, CONTACT_S

G, C, MSUN = 6.67430e-11, 299792458.0, 1.98847e30
M1, M2 = 1.29, 1.37
MCHIRP = (M1 * M2) ** 0.6 / (M1 + M2) ** 0.2          # solar masses
TM = G * MCHIRP * MSUN / C ** 3                        # chirp-mass time, seconds
F_START, F_END = 30.0, 1553.0
SR = 22050
OUT = style.ROOT / "audio"


def tau_of_f(f):
    """Time before coalescence at GW frequency f (Newtonian)."""
    return 5.0 / 256.0 * TM ** (-5 / 3) * (np.pi * f) ** (-8 / 3)


def f_of_tau(tau):
    return (1 / np.pi) * (5.0 / (256.0 * tau)) ** (3 / 8) * TM ** (-5 / 8)


def synth(f0, f1, sr=SR):
    """Return (t, f, h) from frequency f0 to f1 at true speed."""
    t_end = tau_of_f(f0) - tau_of_f(f1)               # duration, seconds
    t = np.arange(0, t_end, 1 / sr)
    tau = tau_of_f(f0) - t                             # time to coalescence
    f = f_of_tau(tau)
    phase = 2 * np.pi * np.cumsum(f) / sr              # GW phase
    amp = (f / f1) ** (2 / 3)                          # strain ~ f^(2/3)
    h = amp * np.sin(phase)
    return t, f, h


def finish(h, sr=SR, fade_in=0.4, fade_out=0.03):
    h = h.copy()
    n_in, n_out = int(fade_in * sr), int(fade_out * sr)
    h[:n_in] *= np.sin(np.linspace(0, np.pi / 2, n_in)) ** 2
    h[-n_out:] *= np.cos(np.linspace(0, np.pi / 2, n_out)) ** 2
    return (0.85 * h / np.abs(h).max() * 32767).astype(np.int16)


def write(name, h):
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    wavfile.write(p, SR, finish(h))
    print(f"wrote {p}  ({len(h) / SR:.1f} s, {p.stat().st_size / 1e6:.1f} MB)")


def main():
    t, f, h = synth(F_START, F_END)
    print(f"chirp mass {MCHIRP:.3f} Msun; {F_START:.0f} -> {F_END:.0f} Hz in {t[-1]:.1f} s")
    for fs in (30, 50, 100, 200, 500, 1000, 1500):
        print(f"  {fs:>5} Hz reached {tau_of_f(fs) - tau_of_f(F_END):6.2f} s before contact")

    write("bns_inspiral_realtime.wav", h)                      # true speed, from 30 Hz

    # Chapter 1 soundtrack: same chirp, starting at the frequency that makes it reach contact
    # exactly when the animation's stars reach their final pose (CONTACT_S); silence follows
    # while the video holds that pose. File length = video length.
    f_start = brentq(lambda f0: tau_of_f(f0) - tau_of_f(F_END) - CONTACT_S, 20.0, 400.0)
    _, f_ch1, h_end = synth(f_start, F_END)
    n = int(round(DURATION_S * SR))
    h_end = np.pad(h_end, (0, max(0, n - len(h_end))))[:n]
    print(f"chapter 1 audio: starts at {f_start:.1f} Hz, contact at {CONTACT_S:.2f} s, "
          f"silence until {DURATION_S:.2f} s (video length)")
    write("bns_inspiral_last_seconds.wav", h_end)
    video = style.ASSETS / "inspiral_chirp.mp4"          # chapter 1 video carries this soundtrack
    if video.exists():
        add_audio(video, OUT / "bns_inspiral_last_seconds.wav")

    # Spectrogram of the full signal, as a check and for a look
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fr, tt, S = spectrogram(h, SR, nperseg=4096, noverlap=3584)
    ax.pcolormesh(tt, fr, 10 * np.log10(S + 1e-20), vmin=-90, vmax=-20, cmap="magma_r", shading="auto")
    ax.plot(t, f, color=style.CLAY, lw=1, ls="--", alpha=.8, label="expected frequency")
    ax.set_yscale("log")
    ax.set_ylim(20, 2000)
    ax.set_yticks([30, 100, 300, 1000])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz]")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "bns_inspiral_spectrogram.png")
    print("wrote", OUT / "bns_inspiral_spectrogram.png")

    # Check: measured peak frequency vs formula at a few times
    for tq in (0.5 * t[-1], 0.9 * t[-1], t[-1] - 2.0, t[-1] - 0.3):
        i = np.argmin(abs(tt - tq))
        meas = fr[np.argmax(S[:, i])]
        print(f"  t={tq:6.1f}s measured {meas:7.1f} Hz vs formula {np.interp(tq, t, f):7.1f} Hz")


if __name__ == "__main__":
    main()
