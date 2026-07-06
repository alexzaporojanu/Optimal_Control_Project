import numpy as np
import os

for filename in ['optimal_trajectory_task1.npy', 'optimal_trajectory_task2.npy']:
    filepath = os.path.join('/home/alex/OPCON_Proj/Project/data', filename)
    if os.path.exists(filepath):
        try:
            data = np.load(filepath, allow_pickle=True).item()
            print(f"File: {filename}")
            print(f"  Keys: {list(data.keys())}")
            if 'x' in data:
                print(f"  x shape: {data['x'].shape}")
            if 'u' in data:
                print(f"  u shape: {data['u'].shape}")
            if 'J' in data:
                print(f"  Cost history J: {data['J']}")
                print(f"  Number of iterations: {len(data['J'])}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    else:
        print(f"File not found: {filename}")
