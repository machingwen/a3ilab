import argparse
import torch
import os

def get_loss_from_pth(file_path):
    try:
        checkpoint = torch.load(file_path, map_location='cpu')
        return checkpoint.get('loss', None)
    except Exception as e:
        print(f"[Error] Failed to load {file_path}: {e}")
        return None

def check_single_file(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return
    loss = get_loss_from_pth(file_path)
    if loss is not None:
        print(f"{os.path.basename(file_path)} → loss: {loss}")
    else:
        print(f"{os.path.basename(file_path)} → 'loss' field not found")

def check_directory(dir_path):
    if not os.path.isdir(dir_path):
        print(f"Directory not found: {dir_path}")
        return

    print(f"Scanning directory: {dir_path}")
    pth_files = [f for f in os.listdir(dir_path) if f.endswith('.pth')]

    if not pth_files:
        print("No .pth files found in the directory.")
        return

    for fname in sorted(pth_files):
        full_path = os.path.join(dir_path, fname)
        loss = get_loss_from_pth(full_path)
        if loss is not None:
            print(f"{fname} → loss: {loss}")
        else:
            print(f"{fname} → 'loss' field not found")

def main():
    parser = argparse.ArgumentParser(description="Read 'loss' value from a .pth file or a directory of .pth files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', type=str, help='Path to a single .pth file')
    group.add_argument('-d', '--dir', type=str, help='Path to a directory containing .pth files')
    args = parser.parse_args()

    if args.file:
        check_single_file(args.file)
    elif args.dir:
        check_directory(args.dir)

if __name__ == "__main__":
    main()

