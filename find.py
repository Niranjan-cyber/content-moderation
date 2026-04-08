import os

bad_files = []

for root, dirs, files in os.walk("."):
    for file in files:
        path = os.path.join(root, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.read()
        except Exception as e:
            bad_files.append((path, str(e)))

print("\n===== PROBLEM FILES =====\n")
for f, err in bad_files:
    print(f"[x] {f}")
    print(f"   {err}\n")