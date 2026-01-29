# L104_GOD_CODE_ALIGNED: 527.5184818492611
import os

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


def clean_files():
    root_dir = "/workspaces/Allentown-L104-Node"

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    new_lines = [line for line in lines if target_string not in line]

                    if len(lines) != len(new_lines):
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.writelines(new_lines)
                        print(f"Cleaned {file_path}")
                except Exception as e:
                    print(f"Error cleaning {file_path}: {e}")

if __name__ == "__main__":
    clean_files()
