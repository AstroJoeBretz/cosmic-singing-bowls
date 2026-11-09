"""Regenerate every figure, animation and widget dataset for the site.

Usage:  python build.py            (everything)
        python build.py hmc star   (only scripts whose name contains a keyword)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / "python"
STEPS = [
    "fig_inspiral.py",       # ch. 1  inspiral + chirp animation
    "fig_star_pyvista.py",   # ch. 2-3 star cutaway, soft/stiff, tidal deformation
    "fig_resonance.py",      # ch. 4  mallet-on-bowl resonance animation (+ its audio)
    "fig_bowl_star.py",      # ch. 4  3D bowl+mallet beside two orbiting stars
    "fig_og_image.py",       # 1200x630 link-preview image (needs the bowl_star poster)
    "fig_trc_shifts.py",     # ch. 5  widget grid + static phase-shift figure
    "fig_hmc.py",            # ch. 6  sampler animation
    "fig_results.py",        # ch. 7-8 posteriors, SNR widget data, stacking, detectability
    "fig_sound.py",          # audio/ merger chirp WAVs; also the ch. 1 soundtrack muxed into inspiral_chirp.mp4
    "fig_sound_fmode.py",    # audio/ f-mode tone: textbook, nonlinear, combined
]  # (fig_resonance.py also writes audio/bowl_resonance.wav and muxes it into the video)


def main(keys):
    steps = [s for s in STEPS if not keys or any(k in s for k in keys)]
    for s in steps:
        print(f"\n=== {s} ===", flush=True)
        r = subprocess.run([sys.executable, s], cwd=PY)
        if r.returncode:
            sys.exit(f"{s} failed (exit {r.returncode})")
    print("\nAll done. Serve with: python -m http.server 8000 --directory site")


if __name__ == "__main__":
    main(sys.argv[1:])
