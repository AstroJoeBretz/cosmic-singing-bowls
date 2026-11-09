"""1200x630 link-preview image (Open Graph / social cards) built from the story's own poster."""
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

import style

W, H = 1200, 630
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def font(name, px):
    return ImageFont.truetype(str(FONTS / name), px)


def main():
    poster = Image.open(style.ASSETS / "bowl_star_analogy_poster.png").convert("RGB")
    img = Image.new("RGB", (W, H), style.BG)
    pw = 1040
    ph = round(poster.height * pw / poster.width)
    img.paste(poster.resize((pw, ph), Image.LANCZOS), ((W - pw) // 2, 24))

    d = ImageDraw.Draw(img)
    y = 24 + ph + 14
    title, sub = "Cosmic Singing Bowls", "A visual story about neutron stars and gravitational waves"
    ft, fs = font("DejaVuSans-Bold.ttf", 42), font("DejaVuSans.ttf", 22)
    d.text(((W - d.textlength(title, font=ft)) / 2, y), title, font=ft, fill=style.INK)
    d.text(((W - d.textlength(sub, font=fs)) / 2, y + 56), sub, font=fs, fill=style.INK_SOFT)
    out = style.ASSETS / "og-image.png"
    img.save(out, optimize=True)
    print("wrote", out, f"{out.stat().st_size / 1e3:.0f} kB, {W}x{H}")


if __name__ == "__main__":
    main()
