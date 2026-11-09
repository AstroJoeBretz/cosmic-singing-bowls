"""Chapter 6: a Hamiltonian Monte Carlo sampler exploring the posterior.

Uses the real chain (textbook model fit to an SNR-500 signal that includes the
nonlinear effect; samples are shown in the order the sampler produced them).
Left: the walker's path in (tidal deformability, mass ratio). Right: the
histogram of tidal deformability builds up from the visits.
"""
import numpy as np
import matplotlib.pyplot as plt

import style
import hmc_data as hd
from anim import save_mp4

style.apply()
s = hd.samples("Anl_SNR500/qtpl")
lam, q = s.lam_cano.to_numpy(), s.q.to_numpy()
M = len(lam)
F = 270                 # frames of sampling
HOLD = 60               # frames held on the final state with the true values marked (2 s at 30 fps)

lo, hi = np.percentile(lam, [0.05, 99.95])
qlo, qhi = q.min(), q.max()
pad = 0.04 * (qhi - qlo)
qlo, qhi = qlo - pad, qhi + pad
bins = np.linspace(lo, hi, 46)
n_at = np.minimum(M, (3 * np.arange(F) + M * (np.arange(F) / (F - 1)) ** 4).astype(int) + 2)

fig = plt.figure(figsize=(9, 4.2), dpi=100)
ax = fig.add_axes([0.07, 0.17, 0.43, 0.68])
axh = fig.add_axes([0.58, 0.17, 0.40, 0.68])

ax.set_xlim(lo, hi)
ax.set_ylim(qlo, qhi)
ax.set_xlabel("tidal deformability  →", color=style.INK_SOFT)
ax.set_ylabel("mass ratio  →", color=style.INK_SOFT)
ax.set_xticks([]), ax.set_yticks([])
ax.set_title("the sampler's path", loc="left", fontsize=12)
vis = ax.scatter([], [], s=4, color=style.INK, alpha=.18, lw=0)
trail, = ax.plot([], [], color=style.CLAY, lw=1.3, alpha=.8)
head, = ax.plot([], [], "o", color=style.CLAY, ms=9, zorder=5)

axh.set_xlim(lo, hi)
axh.set_xticks([])
axh.set_yticks([])
axh.spines["left"].set_visible(False)
axh.set_xlabel("tidal deformability  →", color=style.INK_SOFT)
axh.set_title("where the sampler spends its time", loc="left", fontsize=12)
counts_max = np.histogram(lam, bins)[0].max()
axh.set_ylim(0, counts_max * 1.15)
bars = axh.bar(bins[:-1], np.zeros(len(bins) - 1), width=np.diff(bins), align="edge",
               color=style.INK, alpha=.85)
truth_a = ax.axvline(hd.TRUE_LAM, color=style.GOLD, lw=2, ls="--", alpha=0)
truth_h = axh.axvline(hd.TRUE_LAM, color=style.GOLD, lw=2, ls="--", alpha=0)
note = axh.text(hd.TRUE_LAM + 1.5, counts_max * 0.55, "true\nvalue", color=style.GOLD_TEXT,
                ha="left", va="center", fontsize=10, alpha=0)


def update(i):
    k = min(i, F - 1)
    n = n_at[k]
    vis.set_offsets(np.column_stack([lam[:n], q[:n]]))      # every accumulated sample, no thinning
    t0 = max(0, n - 40)
    trail.set_data(lam[t0:n], q[t0:n])
    head.set_data([lam[n - 1]], [q[n - 1]])
    h = np.histogram(lam[:n], bins)[0]
    for b, v in zip(bars, h):
        b.set_height(v)
    show = 1.0 if i >= F else 0.0
    for a in (truth_a, truth_h):
        a.set_alpha(show)
    note.set_alpha(show)
    return ()


save_mp4(fig, update, F + HOLD, style.ASSETS / "hmc_sampler.mp4",
         poster=(F + 5, style.ASSETS / "hmc_sampler_poster.png"))
