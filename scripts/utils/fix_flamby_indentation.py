import os

def fix_flamby():
    path = "/home/Aarya/healthcare_project/.venv_fix/lib/python3.11/site-packages/flamby/datasets/fed_ixi/dataset.py"
    if not os.path.exists(path):
        print(f"Path {path} does not exist")
        return
        
    with open(path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    found = False
    for line in lines:
        if 'self.parent_dir_name = ""' in line and line.startswith(' '):
            # Force correct indentation
            new_lines.append('        self.parent_dir_name = ""\n')
            found = True
        else:
            new_lines.append(line)
            
    if found:
        with open(path, 'w') as f:
            f.writelines(new_lines)
        print("Successfully fixed indentation.")
    else:
        print("Target line not found.")

if __name__ == "__main__":
    fix_flamby()
