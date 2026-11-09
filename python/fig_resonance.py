"""Chapter 4: a mallet circling a singing bowl's rim, and the resonance curve.

Left: top view of the rim. The mallet circles at rate W; the rim bulges
(a quadrupole, like the star's f-mode) with an amplitude set by the
response curve. Right: that response versus the mallet rate, with the
current rate marked. Schematic: a damped driven oscillator.
"""
import numpy as np
import matplotlib.pyplot as plt

import style
from anim import save_mp4

from fig_resonance_params import N, HOLD, ZETA, SING_THRESHOLD, response  # shared with fig_sound_bowl.py

style.apply()


Wgrid = np.linspace(0.05, 1.6, 400)
Rgrid = response(Wgrid)
Rmax = Rgrid.max()

s = np.linspace(0, 1, N)
ease = s * s * (3 - 2 * s)
W = 0.2 + 1.25 * ease                       # mallet rate, in units of the bowl's tone
# Mallet angle. 0.15 rad per frame per unit W (30 fps): ~0.13 rev/s at the start, ~0.7 rev/s at
# resonance (W = 1, about 1.4 s per lap) and ~1 rev/s at the end, like a real mallet circuit.
# Only the picture depends on this; the soundtrack uses W directly.
phase = np.cumsum(W) * 0.15

fig = plt.figure(figsize=(9, 4.2), dpi=100)
ax_b = fig.add_axes([0.01, 0.04, 0.40, 0.90])
ax_r = fig.add_axes([0.51, 0.20, 0.46, 0.66])

ax_b.set_xlim(-2.0, 2.0)
ax_b.set_ylim(-2.0, 2.0)
ax_b.set_aspect("equal")
ax_b.axis("off")
th = np.linspace(0, 2 * np.pi, 361)
rim, = ax_b.plot([], [], color=style.INK, lw=5, solid_capstyle="round")
bowl_fill = ax_b.fill([], [], color=style.INK, alpha=.10)[0]


def stadium(r0, r1, half_w, n=14):
    """Outline of a rounded bar along +x from radius r0 to r1 (a top view of a mallet part)."""
    c = np.linspace(-np.pi / 2, np.pi / 2, n)
    right = np.column_stack([r1 - half_w + half_w * np.cos(c), half_w * np.sin(c)])
    left = np.column_stack([r0 + half_w - half_w * np.cos(c), -half_w * np.sin(c)])
    return np.vstack([right, left])


def rot(pts, ph):
    cph, sph = np.cos(ph), np.sin(ph)
    return pts @ np.array([[cph, sph], [-sph, cph]])


# Suede-wrapped mallet seen from above: wooden handle pointing out, clay suede head against the rim.
HEAD_LEN, HEAD_HW = 0.40, 0.105
HANDLE_END, HANDLE_HW = 1.95, 0.042
WOOD = "#9C8150"
handle = plt.Polygon(stadium(1.4, HANDLE_END, HANDLE_HW), closed=True, fc=WOOD, ec="none", zorder=4)
head = plt.Polygon(stadium(1.1, 1.1 + HEAD_LEN, HEAD_HW), closed=True, fc=style.CLAY, ec="#8C3C28", lw=1, zorder=5)
ax_b.add_patch(handle)
ax_b.add_patch(head)
arc_a, = ax_b.plot([], [], color=style.CLAY, lw=2, alpha=0)
rings = [ax_b.plot([], [], color=style.CLAY, lw=1.6, alpha=0)[0] for _ in range(3)]
label = ax_b.text(0, -1.98, "", ha="center", va="top", fontsize=11, color=style.INK_SOFT)

ax_r.plot(Wgrid, Rgrid, color=style.INK, lw=2.4)
ax_r.fill_between(Wgrid, 0, Rgrid, color=style.INK, alpha=.08)
ax_r.axvline(1.0, color=style.SAND, ls="--", lw=1.2)
ax_r.text(1.0, Rmax * 1.04, "bowl's natural tone", ha="center", va="bottom",
          fontsize=10, color=style.INK_SOFT)
dot, = ax_r.plot([], [], "o", color=style.CLAY, ms=12, zorder=5)
ax_r.set_xlim(0, 1.6)
ax_r.set_ylim(0, Rmax * 1.2)
ax_r.set_yticks([])
ax_r.set_xticks([0, 1])
ax_r.set_xticklabels(["slow", "matches\nthe tone"])
ax_r.set_xlabel("how fast the mallet circles  →", color=style.INK_SOFT)
ax_r.set_ylabel("how loudly the bowl rings", color=style.INK_SOFT)
ax_r.spines["left"].set_visible(False)


def update(i):
    k = min(i, N - 1)
    Wk, ph = W[k], phase[k]
    A = response(Wk) / Rmax                              # 0..1
    a = 0.02 + 0.15 * A
    r_rim = 1 + a * np.cos(2 * (th - ph))
    x, y = r_rim * np.cos(th), r_rim * np.sin(th)
    rim.set_data(x, y)
    bowl_fill.set_xy(np.column_stack([x, y]))
    rim.set_color(style.CLAY if A > 0.6 else style.INK)
    r_in = 1.0 + a + 0.015                      # head touches the rim, which bulges toward the mallet
    head.set_xy(rot(stadium(r_in, r_in + HEAD_LEN, HEAD_HW), ph))
    handle.set_xy(rot(stadium(r_in + HEAD_LEN - 0.04, HANDLE_END, HANDLE_HW), ph))
    dot.set_data([Wk], [response(Wk)])
    for j, ring in enumerate(rings):
        rr = 1.5 + 0.45 * ((k * 0.05 + j / 3) % 1)
        fade = max(0.0, (A - 0.35) / 0.65) * (1 - ((k * 0.05 + j / 3) % 1))
        ring.set_data(rr * np.cos(th), rr * np.sin(th))
        ring.set_alpha(0.55 * fade)
    label.set_text("rubbing at the bowl's tone: it sings!" if A > SING_THRESHOLD
                   else ("rubbing too slowly: faint murmur" if Wk < 1 else "rubbing too fast: fades again"))
    return ()


save_mp4(fig, update, N + HOLD, style.ASSETS / "bowl_resonance.mp4",
         poster=(int(N * 0.62), style.ASSETS / "bowl_resonance_poster.png"))

# The video carries its own soundtrack (muted by default on the page).
import fig_sound_bowl  # noqa: E402
fig_sound_bowl.main()
