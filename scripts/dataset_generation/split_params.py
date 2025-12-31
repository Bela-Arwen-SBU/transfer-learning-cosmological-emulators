import numpy as np
import os

print("Loading dataset...") 
full_data = np.load('basetruth_validation_params_20k.npy')                                 # Filepath

total = full_data.shape[0]
per_file = 2000
start_file_num = 0

num_files = (total + per_file - 1) // per_file

print(f"Total samples: {total}")
print(f"Samples per file: {per_file}")
print(f"Number of files: {num_files}")
print(f"Files will be: _{start_file_num}.npy through _{start_file_num + num_files - 1}.npy")
print(f"Parameters: {full_data.shape[1]}")

print("\nSplitting into batches...")
for i in range(num_files):
    start_idx = i * per_file
    end_idx = min((i + 1) * per_file, total)
    batch = full_data[start_idx:end_idx]
    
    file_num = start_file_num + i
    np.save(f'./bt_validation_params/_{file_num}.npy', batch)                              # Filepath
    
    if i % 50 == 0:
        print(f"  Created _{file_num}.npy ({batch.shape[0]} samples)")

print(f"\nDone! Created {num_files} files in ./bt_validation_params/")                     # Filepath
print(f"Last file _{start_file_num + num_files - 1}.npy has {end_idx - start_idx} samples")
