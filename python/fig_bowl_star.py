"""Chapter 4: the bowl-and-mallet system next to the two-star system (3D, side by side).

Left: a suede-wrapped mallet circles a singing bowl's rim, and the rim bulges
(a quadrupole) toward and away from the mallet. Right: two neutron stars orbit,
and each is stretched toward its companion. In both, a circling driver raises a
bulge that follows it. Bowl <-> primary star (teal); mallet <-> companion (clay).
Schematic: bulges are exaggerated, the stars are drawn as equal and soft.
Output: site/assets/bowl_star_analogy.mp4 (seamless loop: two orbits) + poster.
"""
import sys

import numpy as np
import pyvista as pv
import imageio.v2 as imageio
import matplotlib
from PIL import Image, ImageDraw, ImageFont

import style

pv.OFF_SCREEN = True

N_FRAMES, ORBITS, FPS = 240, 2, 30
PANEL_W, PANEL_H = 800, 720
CAP_H = 0     # captions live in the page (HTML) so they stay readable on phones; set >0 to bake them in
WOOD, SUEDE = "#9C8150", style.CLAY
# bowl
H = 0.80                 # bowl height (rim radius = 1)
A_BOWL = 0.075           # rim bulge amplitude (exaggerated)
BETA = np.radians(40)    # mallet tilt from vertical
R_HEAD = 0.10            # suede head radius
R_HANDLE = 0.042
L_HEAD, L_HANDLE = 0.38, 0.85
# stars
D_STAR, AMP_STAR = 3.4, 2.2   # separation (star radii), exaggeration -> bulge 17%


# ---------------------------------------------------------------- bowl
def make_bowl():
    t = np.linspace(0, 1, 70)
    z_out = H * t
    r_out = 0.30 + 0.70 * np.sqrt(1 - ((H - z_out) / H) ** 2)
    zf = 0.05
    z_in = zf + (H - zf) * t[::-1]
    r_in = 0.25 + (0.945 - 0.25) * np.sqrt(1 - ((H - z_in) / (H - zf)) ** 2)
    # profile starts/ends just off the axis (r = 0.02) to avoid a degenerate pole in the shading
    prof = np.vstack([[0.02, 0.0], np.column_stack([r_out, z_out]),
                      [[0.945, H]], np.column_stack([r_in, z_in])[1:], [[0.02, zf]]])
    pts = np.column_stack([prof[:, 0], np.zeros(len(prof)), prof[:, 1]])
    line = pv.PolyData(pts, lines=np.hstack([[len(pts)], np.arange(len(pts))]))
    bowl = line.extrude_rotate(resolution=200, capping=False).triangulate().clean()
    return bowl.compute_normals(split_vertices=True, feature_angle=50, auto_orient_normals=False)


def deform_bowl(base, phi_m):
    p = np.asarray(base.points).copy()
    r = np.hypot(p[:, 0], p[:, 1])
    phi = np.arctan2(p[:, 1], p[:, 0])
    delta = A_BOWL * (p[:, 2] / H) ** 2 * np.cos(2 * (phi - phi_m))
    scale = np.where(r > 1e-6, 1 + delta / np.maximum(r, 1e-6), 1.0)
    p[:, 0] *= scale
    p[:, 1] *= scale
    m = base.copy()
    m.points = p
    return m.compute_normals(inplace=False, split_vertices=False)


# ---------------------------------------------------------------- mallet
def make_mallet():
    head = pv.Cylinder(center=(0, 0, L_HEAD / 2), direction=(0, 0, 1), radius=R_HEAD, height=L_HEAD, resolution=64)
    head = head.merge(pv.Sphere(radius=R_HEAD, center=(0, 0, 0), theta_resolution=48, phi_resolution=48))
    head = head.merge(pv.Sphere(radius=R_HEAD, center=(0, 0, L_HEAD), theta_resolution=48, phi_resolution=48))
    z0 = L_HEAD + 0.02
    handle = pv.Cylinder(center=(0, 0, z0 + L_HANDLE / 2), direction=(0, 0, 1), radius=R_HANDLE,
                         height=L_HANDLE, resolution=32)
    handle = handle.merge(pv.Sphere(radius=R_HANDLE, center=(0, 0, z0 + L_HANDLE), theta_resolution=24, phi_resolution=24))
    return head, handle


def place_mallet(mesh, phi_m):
    d = np.array([np.sin(BETA) * np.cos(phi_m), np.sin(BETA) * np.sin(phi_m), np.cos(BETA)])
    ex = np.array([-np.sin(phi_m), np.cos(phi_m), 0.0])
    ey = np.cross(d, ex)
    r_c = 1.0 + A_BOWL + R_HEAD * 0.9
    centre = np.array([r_c * np.cos(phi_m), r_c * np.sin(phi_m), H - 0.015])
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = ex, ey, d, centre
    return mesh.transform(M, inplace=False)


# ---------------------------------------------------------------- stars
_SPH = pv.Sphere(radius=1.0, theta_resolution=96, phi_resolution=96)


def star(centre, toward):
    """Unit sphere stretched along `toward` (unit vector to the companion)."""
    p = np.asarray(_SPH.points).copy()
    c = (p @ toward) / np.linalg.norm(p, axis=1)
    eps = AMP_STAR * 3.0 / D_STAR ** 3
    p *= (1 + eps * 0.5 * (3 * c ** 2 - 1))[:, None]
    m = _SPH.copy()
    m.points = p + centre
    return m.compute_normals(inplace=False)


def ring(radius, z=0.0, n=240):
    th = np.linspace(0, 2 * np.pi, n)
    pts = np.column_stack([radius * np.cos(th), radius * np.sin(th), np.full(n, z)])
    return pv.lines_from_points(pts).tube(radius=0.008)


# ---------------------------------------------------------------- captions
def font(px):
    return ImageFont.truetype(str(matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf"), px)


def caption_strip(width, height):
    img = Image.new("RGB", (width, height), style.BG)
    dr = ImageDraw.Draw(img)

    def centred(x0, w, y, parts, px):
        f = font(px)
        total = sum(dr.textlength(t, font=f) for t, _ in parts)
        x = x0 + (w - total) / 2
        for t, col in parts:
            dr.text((x, y), t, font=f, fill=col)
            x += dr.textlength(t, font=f)

    half = width // 2
    clay = "#A34830"
    centred(0, half, 22, [("A ", style.INK), ("mallet", clay), (" circles the bowl", style.INK)], 32)
    centred(0, half, 68, [("and its rim bulges to follow it", style.INK_SOFT)], 25)
    centred(half, half, 22, [("A ", style.INK), ("companion star", clay), (" orbits", style.INK)], 32)
    centred(half, half, 68, [("and the star's surface bulges to follow it", style.INK_SOFT)], 25)
    return np.asarray(img)


# ---------------------------------------------------------------- render
def build_plotter():
    pl = pv.Plotter(shape=(1, 2), window_size=(2 * PANEL_W, PANEL_H), off_screen=True, border=False)
    el = np.radians(33)
    for col, (dist, focal) in enumerate([(8.0, (0, 0, 0.95)), (11.8, (0, 0, 0))]):
        pl.subplot(0, col)
        pl.set_background(style.BG)
        cam = (focal[0], focal[1] - dist * np.cos(el), focal[2] + dist * np.sin(el))
        pl.camera_position = [cam, focal, (0, 0, 1)]
        pl.camera.view_angle = 30
    pl.subplot(0, 0)
    pl.add_mesh(ring(1.0 + A_BOWL + R_HEAD * 0.9, z=H - 0.015), color=style.SAND, name="ring_l")   # the mallet's path
    pl.subplot(0, 1)
    pl.add_mesh(ring(D_STAR / 2, z=0.0), color=style.SAND, name="ring_r")
    return pl


def draw_frame(pl, bowl_base, head0, handle0, phi):
    pl.subplot(0, 0)
    pl.add_mesh(deform_bowl(bowl_base, phi), color=style.INK, smooth_shading=True, specular=0.6,
                specular_power=30, ambient=0.3, name="bowl")
    pl.add_mesh(place_mallet(handle0, phi), color=WOOD, smooth_shading=True, specular=0.15, ambient=0.35, name="handle")
    pl.add_mesh(place_mallet(head0, phi), color=SUEDE, smooth_shading=True, specular=0.0, ambient=0.4, name="head")

    pl.subplot(0, 1)
    u = np.array([np.cos(phi), np.sin(phi), 0.0])
    c = (D_STAR / 2) * u
    pl.add_mesh(star(-c, u), color=style.INK, smooth_shading=True, specular=0.25, ambient=0.35, name="starA")
    pl.add_mesh(star(c, -u), color=style.CLAY, smooth_shading=True, specular=0.25, ambient=0.35, name="starB")


def main(test_frame=None):
    bowl_base = make_bowl()
    head0, handle0 = make_mallet()
    pl = build_plotter()
    cap = caption_strip(2 * PANEL_W, CAP_H) if CAP_H else None
    phis = 2 * np.pi * ORBITS * np.arange(N_FRAMES) / N_FRAMES

    def frame(phi):
        draw_frame(pl, bowl_base, head0, handle0, phi)
        img = pl.screenshot(return_img=True)[..., :3]
        return np.vstack([img, cap]) if cap is not None else img

    if test_frame is not None:
        out = style.ROOT / "python" / "_bowl_star_test.png"
        Image.fromarray(frame(phis[test_frame])).save(out)
        print("wrote", out)
        return
    path = style.ASSETS / "bowl_star_analogy.mp4"
    w = imageio.get_writer(str(path), fps=FPS, codec="libx264", pixelformat="yuv420p",
                           ffmpeg_params=["-crf", "26", "-movflags", "+faststart"], macro_block_size=2)
    for i, phi in enumerate(phis):
        f = frame(phi)
        w.append_data(f)
        if i == 40:
            Image.fromarray(f).save(style.ASSETS / "bowl_star_analogy_poster.png")
        if i % 40 == 0:
            print("frame", i, flush=True)
    w.close()
    pl.close()
    print("wrote", path, f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(test_frame=int(sys.argv[1]) if len(sys.argv) > 1 else None)
