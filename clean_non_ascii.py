import os

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
