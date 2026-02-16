#!/usr/bin/env python3
"""Generate a minimalist phi (φ) icon for L104 app."""

import os
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont

def create_phi_icon(size):
    """Create a minimalist phi symbol icon at the given size."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background: soft warm white circle with very subtle border
    margin = int(size * 0.04)
    bg_rect = [margin, margin, size - margin, size - margin]

    # Rounded-rect background (continuous cornerRadius like macOS icons)
    corner = int(size * 0.22)
    draw.rounded_rectangle(bg_rect, radius=corner, fill=(252, 252, 253, 255))

    # Subtle border
    draw.rounded_rectangle(bg_rect, radius=corner, outline=(210, 210, 215, 80), width=max(1, size // 256))

    # Draw the phi symbol (φ) — elegant, centered
    # Try system fonts that have good phi glyphs
    phi_char = "φ"
    font_size = int(size * 0.58)

    font = None
    # Try elegant fonts in order of preference
    font_paths = [
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Times.ttc",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/NewYork.ttf",
        "/System/Library/Fonts/NewYorkItalic.ttf",
        "/Library/Fonts/Georgia.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue

    if font is None:
        font = ImageFont.load_default()

    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), phi_char, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (size - tw) / 2 - bbox[0]
    ty = (size - th) / 2 - bbox[1] - size * 0.01  # slight upward nudge

    # Gold color matching L104Theme.gold (#B38A1A readable variant)
    gold = (163, 126, 22, 255)  # Rich readable gold

    # Draw phi with slight shadow for depth
    shadow_offset = max(1, size // 256)
    shadow_color = (130, 100, 15, 40)
    draw.text((tx + shadow_offset, ty + shadow_offset), phi_char, font=font, fill=shadow_color)

    # Main phi
    draw.text((tx, ty), phi_char, font=font, fill=gold)

    return img


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # macOS icon sizes needed for .iconset
    icon_sizes = [16, 32, 64, 128, 256, 512, 1024]

    # Create iconset directory
    iconset_dir = os.path.join(script_dir, "AppIcon.iconset")
    os.makedirs(iconset_dir, exist_ok=True)

    for size in icon_sizes:
        img = create_phi_icon(size)

        # 1x
        if size <= 512:
            fname = f"icon_{size}x{size}.png"
            img.save(os.path.join(iconset_dir, fname))
            print(f"  ✓ {fname}")

        # 2x (half-size name)
        if size >= 32:
            half = size // 2
            fname2x = f"icon_{half}x{half}@2x.png"
            img.save(os.path.join(iconset_dir, fname2x))
            print(f"  ✓ {fname2x}")

    # Convert to .icns using iconutil
    icns_path = os.path.join(script_dir, "L104Native.app", "Contents", "Resources", "AppIcon.icns")
    os.makedirs(os.path.dirname(icns_path), exist_ok=True)

    result = subprocess.run(
        ["iconutil", "-c", "icns", iconset_dir, "-o", icns_path],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        print(f"\n  ✅ AppIcon.icns created: {icns_path}")
        fsize = os.path.getsize(icns_path)
        print(f"  📦 Size: {fsize:,} bytes")
    else:
        print(f"\n  ❌ iconutil failed: {result.stderr}")

    # Cleanup iconset
    import shutil
    shutil.rmtree(iconset_dir)
    print("  🧹 Cleaned up iconset directory")


if __name__ == "__main__":
    print("🎨 Generating phi (φ) icon for L104...")
    main()
    print("\n✨ Done!")
