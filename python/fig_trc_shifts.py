"""Phase differences Psi_x - Psi_b for each tidal resonance correction (TRC).

Uses the paper's own `get_dpsi_qchirp`. Writes:
  - site/assets/trc_grid.json  precomputed grid for the interactive widget
  - site/assets/trc_phase_shifts.png  static fallback (paper Fig. 1 style)
"""
import json

import numpy as np
import matplotlib.pyplot as plt

import style
from bns_import import jw

BASE = dict(
    M_c=1.33, R1=1.17e6, R2=1.17e6, Mchirp=1.157, q=0.942,
    tc=0.0, phic=0.0, lam_c=400.0, wab_c=0.0786,
)
F_C = 1553.0
W_REF = np.pi * 1000


def dpsi(freq, Anl=0.0, S=0.0, gr=0):
    return np.asarray(jw.get_dpsi_qchirp(
        freq, **BASE, Anl_c=Anl, wnl_ref=W_REF, S1=S, S2=0.0, gr_flag=gr))


def diff(freq, **kw):
    return dpsi(freq, **kw) - dpsi(freq)


def resonance_hz(Anl=0.0, S=0.0, gr=0):
    """GW frequency of the primary's l=m=2 resonance (paper Eqs. 28-30)."""
    Mc, q = BASE["Mchirp"], BASE["q"]
    M1 = float(jw.M1fromMchq(Mc, q))
    M2 = float(jw.M2fromMchq(Mc, q))
    wa0 = float(jw.Omega_x(M1, BASE["M_c"], BASE["wab_c"])) / float(jw.tMx(M1 * jw.Ms))
    Anl1 = float(jw.Anl_x(M1, BASE["M_c"], Anl, M2))
    Mt_t = float(jw.tMx((M1 + M2) * jw.Ms))
    eta = M1 * M2 / (M1 + M2) ** 2
    w0 = wa0 / 2 + np.pi * S                      # Omega_A = 2*pi*S
    w1 = (1 + eta * (Mt_t * w0) ** (2 / 3) * gr) * (w0 + Anl1 * (w0 / W_REF) ** 2 * wa0 / 2)
    return w1 / np.pi


def build_grid():
    # The tidal phase integrates from the first frequency, so compute from 100 Hz
    # (as the paper does) and keep the 800 Hz+ part for display.
    f_full = np.arange(100.0, F_C + 1, 1.0)
    keep = np.unique(np.concatenate([np.linspace(700, len(f_full) - 1, 120).astype(int)]))
    f = f_full[keep]
    S_vals = np.arange(-300, 301, 50)
    A_vals = np.round(np.arange(0.0, -0.0601, -0.01), 3)
    base = dpsi(f_full)
    curves, res = [], []
    for gr in (0, 1):
        for S in S_vals:
            for A in A_vals:
                d = dpsi(f_full, Anl=float(A), S=float(S), gr=gr) - base
                curves.append(np.round(d[keep], 4).tolist())
                res.append(round(resonance_hz(float(A), float(S), gr), 1))
    grid = dict(
        f=np.round(f, 2).tolist(), S=S_vals.tolist(), Anl=A_vals.tolist(),
        f_merger=F_C, f_res_base=round(resonance_hz(), 1),
        order="gr,S,Anl", curves=curves, f_res=res)
    out = style.ASSETS / "trc_grid.json"
    out.write_text(json.dumps(grid, separators=(",", ":")))
    print("wrote", out, f"{out.stat().st_size/1e3:.0f} kB; baseline resonance {grid['f_res_base']} Hz")


def static_fig():
    style.apply()
    f = np.arange(100.0, F_C + 1, 1.0)
    sel = f >= 800
    curves = [
        ("nonlinear motion", diff(f, Anl=-0.03), style.CLAY, "--"),
        ("spin, against the orbit (200 Hz)", diff(f, S=-200.0), style.INK, ":"),
        ("spin, with the orbit (200 Hz)", diff(f, S=200.0), style.GOLD, ":"),
        ("relativity", diff(f, gr=1), style.INK_SOFT, "-."),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhline(0, color=style.SAND, lw=1)
    for lab, y, c, ls in curves:
        ax.plot(f[sel], y[sel], color=c, ls=ls, label=lab)
    ax.set_xlabel("gravitational-wave frequency [Hz]")
    ax.set_ylabel("phase change vs. textbook model [rad]")
    ax.legend(loc="lower left")
    out = style.ASSETS / "trc_phase_shifts.png"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    build_grid()
    static_fig()
