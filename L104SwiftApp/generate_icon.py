#!/usr/bin/env python3
"""Generate a minimalist phi (φ) icon for L104 app."""

import os
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont


def main():
    """Generate the phi icon."""
    iconset_dir = "AppIcon.iconset"

    # Clean up existing directory
    if os.path.exists(iconset_dir):
        shutil.rmtree(iconset_dir)
        print("  🧹 Cleaned up iconset directory")

    os.makedirs(iconset_dir, exist_ok=True)

    # Generate icon sizes
    sizes = [16, 32, 64, 128, 256, 512, 1024]

    for size in sizes:
        img = Image.new('RGBA', (size, size), (30, 30, 30, 255))
        draw = ImageDraw.Draw(img)

        # Draw phi symbol
        try:
            font_size = int(size * 0.6)
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()

        # Calculate text position
        text = "φ"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - int(size * 0.1)

        # Draw the phi symbol
        draw.text((x, y), text, fill=(255, 215, 0, 255), font=font)

        # Save
        img.save(f"{iconset_dir}/icon_{size}x{size}.png")
        print(f"  ✓ Generated {size}x{size}")

    # Create .icns file using iconutil
    subprocess.run(["iconutil", "-c", "icns", iconset_dir, "-o", "AppIcon.icns"], check=True)
    print("  ✓ Created AppIcon.icns")


if __name__ == "__main__":
    print("🎨 Generating phi (φ) icon for L104...")
    main()
    print("\n✨ Done!")
