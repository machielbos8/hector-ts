import os

def replace_version_string(directory):
    old_string = "0.1.11."
    new_string = "3.0."

    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                if old_string in content:
                    new_content = content.replace(old_string, new_string)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(new_content)
                    print(f"Updated: {file_path}")

            except (UnicodeDecodeError, PermissionError, IsADirectoryError):
                # Skip files we can't read/write (e.g., binary files or permissions issues)
                continue

if __name__ == "__main__":
    target_dir = input("Enter the path to the directory: ").strip()
    replace_version_string(target_dir)
    print("Done.")

