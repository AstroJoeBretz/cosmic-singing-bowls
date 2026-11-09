"""Chapters 7-8: posteriors, pitch-vs-squishiness, SNR widget data, stacking, detectability."""
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import style
import hmc_data as hd
from anim import save_mp4

style.apply()
T = hd.TRUE_LAM


def kde(x, grid, sub=5):
    return gaussian_kde(np.asarray(x)[::sub])(grid)


def pct(x, v):
    return 100.0 * float(np.mean(np.asarray(x) < v))


def fill_post(ax, x, grid, color, label, alpha=.28):
    d = kde(x, grid)
    ax.fill_between(grid, d, color=color, alpha=alpha, lw=0)
    ax.plot(grid, d, color=color, lw=2.2, label=label)
    return d


def truth(ax, label="true value", x=T, y=None, side="left"):
    """Dashed true-value line with its label on the given side ('left' or 'right') of the line."""
    ax.axvline(x, color=style.GOLD, lw=2, ls="--")
    if label:
        off = 0.006 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        ax.text(x + off if side == "right" else x - off, 0.97, label, color=style.GOLD_TEXT,
                transform=ax.get_xaxis_transform(), ha="left" if side == "right" else "right",
                va="top", fontsize=10, bbox=dict(fc=style.BG, ec="none", alpha=.85, pad=1.5))


def fig_posteriors():
    base = hd.samples("Anl_SNR500/qtpl").lam_cano
    eff = hd.samples("Anl_SNR500/qtplA_negA").lam_cano
    g = np.linspace(340, 520, 600)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fill_post(ax, base, g, style.INK, "textbook model (leaves the effect out)")
    fill_post(ax, eff, g, style.CLAY, "model that includes the effect")
    truth(ax)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("tidal deformability (how squishy the star is)")
    ax.set_xlim(340, 520)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
    out = style.ASSETS / "posterior_lambda.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out, f"| standard: median {np.median(base):.0f}, truth at {pct(base, T):.1f} pct;",
          f"effect: median {np.median(eff):.0f}, truth at {pct(eff, T):.1f} pct")


def fig_pitch_vs_squish():
    s = hd.samples("Anl_SNR500/qtplo")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
    gl = np.linspace(300, 520, 500)
    fill_post(a, s.lam_cano, gl, style.INK, None)
    truth(a)
    a.set_title("squishiness, pitch left free", loc="left", fontsize=11)
    a.set_xlabel("tidal deformability")
    gw = np.linspace(s.wab_cano.min() - .002, s.wab_cano.max() + .002, 500)
    fill_post(b, s.wab_cano, gw, style.CLAY, None)
    truth(b, x=hd.TRUE_WAB, side="right")      # the pitch posterior lies left of the line, so keep the label clear of it
    b.set_title("the pitch itself", loc="left", fontsize=11)
    b.set_xlabel("star's natural pitch (dimensionless)")
    for ax_ in (a, b):
        ax_.set_yticks([])
        ax_.spines["left"].set_visible(False)
    fig.tight_layout()
    out = style.ASSETS / "pitch_vs_squish.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out, f"| free-pitch fit: lam median {s.lam_cano.median():.0f}, pitch median {s.wab_cano.median():.4f}")


def snr_widget_data():
    base = hd.samples("Anl_SNR500/qtpl").lam_cano.to_numpy()
    # Anchor on the best-fit (max-likelihood) value, which is what the paper quotes (432, +8%),
    # not the posterior median (429).
    best = float(hd.best_fit("Anl_SNR500/qtpl").lam_cano)
    dev = np.sort(base - best)
    qs = np.quantile(dev, np.linspace(0, 1, 1001))
    dg = np.linspace(-50, 50, 201)
    pdf = gaussian_kde(base[::5] - best)(dg)
    out = dict(snr_ref=500, truth=T, best_fit=round(best, 2), sigma=round(float(base.std()), 3),
               dev_quantiles=np.round(qs, 2).tolist(),
               dev_grid=dg.tolist(), dev_pdf=np.round(pdf, 6).tolist(),
               note="Rescaled from the SNR-500 chain about the best-fit value: width scales as 1/SNR, bias held fixed.")
    p = style.ASSETS / "snr_posterior.json"
    p.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", p, f"best fit {best:.2f} (bias {best - T:+.1f}) sigma {out['sigma']}")


def fig_detectability():
    rows = [  # value, minus, plus for CE, ET, A#   (paper Table 2)
        ("Nonlinear motion", [(85, 37, 65), (118, 51, 90), (62, 27, 49)]),
        ("Spin against the orbit", [(61, 27, 82), (83, 36, 111), (46, 20, 62)]),
        ("Spin with the orbit", [(107, 27, 82), (147, 36, 111), (82, 20, 62)]),
        ("Relativity", [(215, 0, 0), (300, 0, 0), (158, 0, 0)]),
    ]
    dets = [("Cosmic Explorer", style.INK), ("Einstein Telescope", style.CLAY), ("Advanced LIGO A♯", style.GOLD)]
    GAP = 1.5  # spacing between effect rows (detector spacing within a row is unchanged)
    fig, ax = plt.subplots(figsize=(8, 5.6))
    for r, (name, vals) in enumerate(rows):
        y0 = (len(rows) - 1 - r) * GAP
        for d, ((v, lo, hi), (dn, col)) in enumerate(zip(vals, dets)):
            y = y0 + (1 - d) * 0.22
            if lo or hi:
                ax.plot([v - lo, v + hi], [y, y], color=col, lw=2, alpha=.45, solid_capstyle="round")
            ax.plot([v], [y], "o", color=col, ms=9, label=dn if r == 0 else None)
    ax.set_yticks([i * GAP for i in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xscale("log")
    ax.set_xlim(25, 420)
    ax.set_xticks([30, 50, 100, 200, 400])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("signal loudness (SNR) needed to tell the effect apart")
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=10)
    out = style.ASSETS / "detectability.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def anim_stacking():
    runs = hd.population("qtpl")
    g = np.linspace(330, 560, 700)
    dens = [kde(r.lam_cano, g, sub=6) for r in runs]
    logs = np.cumsum([np.log(d + 1e-300) for d in dens], axis=0)
    stacks = []
    for L in logs:
        p = np.exp(L - L.max())
        stacks.append(p / np.trapezoid(p, g))
    cdf = np.cumsum(stacks[-1]) / stacks[-1].sum()
    print(f"stacked truth percentile: {100*np.interp(T, g, cdf):.2f}  (paper: ~1.25 at SNR 175)",
          f"| stacked median {np.interp(0.5, cdf, g):.0f}")
    ymax = max(stacks[-1].max(), max(d.max() for d in dens)) * 1.08
    fig = plt.figure(figsize=(9, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.17, 0.94, 0.68])
    ax.set_xlim(340, 520)
    ax.set_ylim(0, ymax)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("tidal deformability  →", color=style.INK_SOFT)
    ax.axvline(T, color=style.GOLD, lw=2, ls="--")
    ax.text(T - 2, ymax * 0.62, "true\nvalue", color=style.GOLD_TEXT, ha="right", fontsize=10)
    thin = [ax.plot([], [], color=style.INK_SOFT, lw=1.4, alpha=.55)[0] for _ in runs]
    stk, = ax.plot([], [], color=style.CLAY, lw=3)
    fill = [None]
    title = ax.set_title("", loc="left", fontsize=12)
    per = 12
    N = len(runs)
    frames = N * per + 60

    def update(i):
        k = min(N, i // per + 1) if i < N * per else N
        frac = (i % per) / per if i < N * per else 1
        for j, ln in enumerate(thin):
            if j < k:
                ln.set_data(g, dens[j])
                ln.set_alpha(.55 if j < k - 1 else .25 + .3 * frac)
            else:
                ln.set_data([], [])
        stk.set_data(g, stacks[k - 1])
        if fill[0] is not None:
            fill[0].remove()
        fill[0] = ax.fill_between(g, stacks[k - 1], color=style.CLAY, alpha=.18, lw=0)
        title.set_text(f"{k} event{'s' if k > 1 else ''} stacked")
        return ()

    save_mp4(fig, update, frames, style.ASSETS / "stacking.mp4",
             poster=(frames - 1, style.ASSETS / "stacking_poster.png"))


if __name__ == "__main__":
    fig_posteriors()
    fig_pitch_vs_squish()
    snr_widget_data()
    fig_detectability()
    anim_stacking()
