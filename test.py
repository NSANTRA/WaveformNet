import os

def print_tree(start_path, prefix='', file=None):
    entries = os.listdir(start_path)
    entries.sort()  # Sort entries for consistent output
    for i, name in enumerate(entries):
        path = os.path.join(start_path, name)
        connector = '├── ' if i < len(entries) - 1 else '└── '
        line = prefix + connector + name
        file.write(line + '\n')
        if os.path.isdir(path):
            extension = '│   ' if i < len(entries) - 1 else '    '
            print_tree(path, prefix + extension, file)

output_file = 'tree.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    print_tree("../ECG Heartbeat Classification", file=f)

print(f"Folder structure saved to {output_file}")
