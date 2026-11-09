"""Chapter 1: stylised inspiral (left) and the gravitational-wave chirp (right).

Schematic time: separation shrinks as r ~ (1-s)^(1/4) and the orbital rate
follows Kepler, w ~ r^(-3/2); strain amplitude ~ w^(2/3). Real orbits are
hundreds of Hz, so this is slowed down for the eye.
"""
import numpy as np
import matplotlib.pyplot as plt

import style
from anim import save_mp4

from fig_inspiral_params import N, HOLD  # shared with fig_sound.py

SUB = 4             # curve sampling: points per video frame (smooth trails and waveform)
R0 = 1.9
# Star size is tied to the plot scale: the orbit axes span 2.5 data units across 0.38*9 in,
# so 1 data unit = 98.5 pt, and a marker of ms points has radius ms/2 pt.
PT_PER_UNIT = 0.38 * 9 * 72 / 2.5
MS = 18
R_STAR = (MS / 2) / PT_PER_UNIT                # stellar radius in data units (~0.091)
R_END = 3.0 * R_STAR                           # final centre-to-centre separation: 3 stellar radii

style.apply()

# Fine time grid: index k*SUB is exactly frame k.
L = SUB * (N - 1) + 1
s = np.arange(L) / (L - 1)
x_t = np.arange(L) / SUB                  # time axis in frame units
r = R0 * (1 - s * (1 - (R_END / R0) ** 4)) ** 0.25
w = 0.1 * (R0 / r) ** 1.5                 # orbital rate per frame
phase = np.cumsum(w) / SUB
amp = (w / w[0]) ** (2 / 3)
h = amp * np.cos(2 * phase)

fig = plt.figure(figsize=(9, 4.2), dpi=100)
ax_o = fig.add_axes([0.02, 0.04, 0.38, 0.92])
ax_h = fig.add_axes([0.47, 0.18, 0.50, 0.68])

ax_o.set_xlim(-1.25, 1.25)
ax_o.set_ylim(-1.25, 1.25)
ax_o.set_aspect("equal")
ax_o.axis("off")
ax_o.add_patch(plt.Circle((0, 0), 1.18, fill=False, ec=style.SAND, lw=1, ls=":"))
star_a, = ax_o.plot([], [], "o", color=style.INK, ms=MS, zorder=3)
star_b, = ax_o.plot([], [], "o", color=style.CLAY, ms=MS, zorder=3)
trail_a, = ax_o.plot([], [], color=style.INK, lw=1.2, alpha=.35, solid_joinstyle="round")
trail_b, = ax_o.plot([], [], color=style.CLAY, lw=1.2, alpha=.35, solid_joinstyle="round")
ax_o.text(0, -1.22, "two neutron stars, each ~23 km wide", ha="center", va="top",
          fontsize=10, color=style.INK_SOFT)

ax_h.set_xlim(0, N + HOLD)
ax_h.set_ylim(-amp.max() * 1.15, amp.max() * 1.15)
ax_h.set_yticks([])
ax_h.set_xticks([])
ax_h.spines["left"].set_visible(False)
ax_h.set_xlabel("time  →", color=style.INK_SOFT)
ax_h.set_title("the signal that reaches Earth", loc="left", fontsize=12)
ax_h.axhline(0, color=style.SAND, lw=1)
line, = ax_h.plot([], [], color=style.CLAY, lw=1.6)


def update(i):
    k = min(i, N - 1) * SUB               # fine-grid index of this frame
    th = phase[: k + 1]
    rr = r[: k + 1]
    # equal masses: opposite sides of the centre of mass
    xa, ya = 0.5 * rr * np.cos(th), 0.5 * rr * np.sin(th)
    star_a.set_data([xa[-1]], [ya[-1]])
    star_b.set_data([-xa[-1]], [-ya[-1]])
    t0 = max(0, k - 60 * SUB)
    trail_a.set_data(xa[t0:], ya[t0:])
    trail_b.set_data(-xa[t0:], -ya[t0:])
    line.set_data(x_t[: k + 1], h[: k + 1])
    return ()


save_mp4(fig, update, N + HOLD, style.ASSETS / "inspiral_chirp.mp4",
         poster=(N - 1, style.ASSETS / "inspiral_chirp_poster.png"))

# The video carries its own soundtrack (muted by default on the page).
import fig_sound  # noqa: E402
fig_sound.main()
