import argparse
import numpy as np
import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract statistics from .npy files.")
    parser.add_argument("directory", type=str, help="Directory containing .npy files")
    args = parser.parse_args()

    files = glob.glob(f"{args.directory}/*.npz")
    names = {}
    
    for file in files:
        filename = os.path.basename(file)
        
        data = np.load(file)
        data["chamfer"]
        names[filename] = {
            "f1": data["f1"].item() * 1000,
            "chamfer": data["chamfer"].item() * 1000,
        }
    
    # Compute the space to align the names
    max_name_length = max(len(name) for name in names.keys())
    
    # Sort by chamfer value
    sorted_names = sorted(names.items(), key=lambda x: x[1]["chamfer"])
    print("Sorted statistics by chamfer distance:")
    for name, stats in sorted_names:
        print(f"{name.ljust(max_name_length)}: chamfer={stats['chamfer']:.4f}")
    
    # Save by f1 value
    sorted_by_f1 = sorted(names.items(), key=lambda x: x[1]["f1"])
    print("\nSorted statistics by f1 score:")
    for name, stats in sorted_by_f1:
        print(f"{name.ljust(max_name_length)}: f1={stats['f1']:.4f}")