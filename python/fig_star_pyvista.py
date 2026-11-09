"""Chapters 2-3: neutron-star cutaway, soft-vs-stiff sizes, and tidal deformation.

All pyvista renders are schematic illustrations (layer boundaries and
deformation amplitudes are exaggerated for clarity).
"""
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import style

pv.OFF_SCREEN = True
BG = style.BG

# Layer colors in the 3D render. The crust is pale, so its name in the title uses a deeper
# tan of the same hue (CRUST_TEXT) to stay readable (3.5:1 on the page background).
CRUST, OUTER, INNER = "#D8CBB3", style.INK_SOFT, style.CLAY
CRUST_SURFACE = "#E4D9C4"     # lit outer skin of the cutaway sphere
CRUST_TEXT = "#9C8150"
CONNECT = "#6B6558"   # neutral gray for the commas and "and", distinct from every layer color


def colored_title(fig, y, parts, fontsize=17):
    """Centered title built from (text, color) parts, laid out left to right."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = [fig.text(0, y, t, color=c, fontsize=fontsize, fontweight="bold", va="center", ha="left")
             for t, c in parts]
    widths = [t.get_window_extent(rend).width / fig.bbox.width for t in texts]
    x = 0.5 - sum(widths) / 2
    for t, w in zip(texts, widths):
        t.set_x(x)
        x += w


def cutaway():
    sph = pv.Sphere(radius=1.0, theta_resolution=160, phi_resolution=160)
    sph = sph.clip_box(bounds=(0, 1.2, -1.2, 0, 0, 1.2), invert=True)
    pl = pv.Plotter(off_screen=True, window_size=(1100, 900))
    pl.set_background(BG)
    pl.add_mesh(sph, color=CRUST_SURFACE, smooth_shading=True, specular=0.2)

    rings = [(0.0, 0.5, INNER), (0.5, 0.94, OUTER), (0.94, 1.0, CRUST)]
    eps = 0.015
    faces = [  # (normal, clip bounds that keep the quarter-plane)
        ((1, 0, 0), (-eps, eps, -1.2, 0, 0, 1.2)),
        ((0, 1, 0), (0, 1.2, -eps, eps, 0, 1.2)),
        ((0, 0, 1), (0, 1.2, -1.2, 0, -eps, eps)),
    ]
    for normal, bounds in faces:
        for r_in, r_out, col in rings:
            disc = pv.Disc(center=(0, 0, 0), inner=r_in, outer=r_out,
                           normal=normal, r_res=6, c_res=160)
            disc = disc.clip_box(bounds=bounds, invert=False)
            pl.add_mesh(disc, color=col, smooth_shading=False, ambient=0.5, diffuse=0.6)
    pl.camera_position = [(3.2, -3.4, 2.6), (0.15, -0.15, 0.15), (0, 0, 1)]
    pl.enable_anti_aliasing("ssaa")
    img = pl.screenshot(transparent_background=True, return_img=True)
    pl.close()

    # crop to the star, then add the colour-coded title
    ys, xs = np.where(img[..., 3] > 0)
    pad = 20
    img = img[max(ys.min() - pad, 0): ys.max() + pad, max(xs.min() - pad, 0): xs.max() + pad]
    h, w = img.shape[:2]
    fig_w, title_h = 4.2, 0.62
    fig = plt.figure(figsize=(fig_w, fig_w * (h / w) + title_h))
    ax = fig.add_axes([0, 0, 1, 1 - title_h / fig.get_figheight()])
    ax.imshow(img)
    ax.axis("off")
    colored_title(fig, 1 - title_h / 2 / fig.get_figheight(), [
        ("crust", CRUST_TEXT), (", ", CONNECT), ("outer", OUTER), (", and ", CONNECT), ("inner core", INNER)])
    out = style.ASSETS / "star_cutaway.png"
    fig.savefig(out, bbox_inches=None)
    plt.close(fig)
    print("wrote", out)


def soft_vs_stiff():
    """Same mass, different stiffness -> different size (illustration). 3D-shaded like chapter 3."""
    R_STIFF, R_SOFT, X = 1.0, 0.79, 1.75
    pl = pv.Plotter(off_screen=True, window_size=(1120, 650))   # cropped: no empty side margins
    pl.set_background(BG)
    for cx, r, col in [(-X, R_STIFF, style.CLAY), (X, R_SOFT, style.INK)]:
        s = pv.Sphere(radius=r, center=(cx, 0, 0), theta_resolution=120, phi_resolution=120)
        pl.add_mesh(s, color=col, smooth_shading=True, specular=0.25, ambient=0.35)
    pl.camera_position = [(0, -14, 0), (0, 0, 0), (0, 0, 1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 650 / 2 / 160  # half-height in world units -> 160 px per unit
    pl.enable_anti_aliasing("ssaa")
    img = pl.screenshot(return_img=True)
    pl.close()

    fig_w = 7.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * img.shape[0] / img.shape[1]))
    ax.imshow(img)
    ax.axis("off")
    ppu = 160
    cx0, cy0 = img.shape[1] / 2, img.shape[0] / 2
    for dx, r, name, note in [(-X, R_STIFF, "stiff matter", "puffier star"),
                              (X, R_SOFT, "soft matter", "tighter star")]:
        ax.text(cx0 + dx * ppu, cy0 + r * ppu + 40, f"{name}\n{note}", ha="center", va="top",
                color=style.INK, fontsize=17, linespacing=1.25)
    ax.text(cx0, 45, "same mass, different equation of state", ha="center", va="center",
            color=style.INK_SOFT, fontsize=17)
    out = style.ASSETS / "soft_vs_stiff.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def prolate_sphere(center_x, amp, d):
    """Unit sphere stretched along x (toward the other star); eps = amp*3/d^3 (same as the widget)."""
    sph = pv.Sphere(radius=1.0, theta_resolution=90, phi_resolution=90)
    p = np.asarray(sph.points)
    cos_t = p[:, 0] / np.linalg.norm(p, axis=1)
    eps = amp * 3.0 / d ** 3
    p *= (1 + eps * 0.5 * (3 * cos_t ** 2 - 1))[:, None]
    p[:, 0] += center_x
    sph.points = p
    return sph


def deformed_star(pl, d, amp, companion_color=style.CLAY):
    """Add both stars of an equal pair; each is stretched toward the other."""
    pl.add_mesh(prolate_sphere(0.0, amp, d), color=style.INK, smooth_shading=True, specular=0.25, ambient=0.35)
    pl.add_mesh(prolate_sphere(d, amp, d), color=companion_color, smooth_shading=True, specular=0.25, ambient=0.35)


def deformed_grid():
    seps = [(6.0, "far apart"), (3.8, "closer"), (2.6, "about to merge")]
    stars = [(1.5, "soft stars: stretch a lot"), (0.45, "stiff stars: stretch a little")]
    pl = pv.Plotter(shape=(2, 3), off_screen=True, window_size=(1500, 580), border=False)
    for row, (amp, _) in enumerate(stars):
        for col, (d, _) in enumerate(seps):
            pl.subplot(row, col)
            pl.set_background(BG)
            deformed_star(pl, d, amp)
            pl.camera_position = [(d / 2, -14, 0), (d / 2, 0, 0), (0, 0, 1)]   # centre each pair
            pl.camera.parallel_projection = True
            pl.camera.parallel_scale = 2.6
    img = pl.screenshot(transparent_background=False, return_img=True)
    pl.close()

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.imshow(img)
    ax.axis("off")
    h, w = img.shape[:2]
    for c, (_, lab) in enumerate(seps):
        ax.text((c + 0.5) * w / 3, -8, lab, ha="center", va="bottom",
                fontsize=12, color=style.INK_SOFT)
    for r_, (_, lab) in enumerate(stars):
        ax.text(8, (r_ + 0.07) * h / 2, lab, ha="left", va="top", fontsize=11,
                color=style.INK, fontweight="bold")
    out = style.ASSETS / "tidal_deformation.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    cutaway()
    soft_vs_stiff()
    deformed_grid()
