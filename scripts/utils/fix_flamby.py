import pathlib
import sys

p = pathlib.Path('/home/Aarya/healthcare_project/.venv/lib/python3.11/site-packages/flamby/datasets/fed_ixi/dataset.py')
if not p.exists():
    print(f"Cannot find {p}")
    sys.exit(1)

content = p.read_text()
# Fix any 16-space indents to 8-space for this specific line
content = content.replace('                self.parent_dir_name = ""', '        self.parent_dir_name = ""')
p.write_text(content)
print("Indentation fixed.")