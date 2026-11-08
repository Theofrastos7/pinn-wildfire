"""
Generate noisy versions of the fire datasets for the noise robustness study.
Adds additive Gaussian noise at sigma = 1%, 5%, 10% of the field std to T and F.
Saves new CSVs in each model's data/ folder. Always overwrites existing files.

Run from project root:  python extras/generate_noisy_data.py
"""
import pandas as pd
import numpy as np
import os

NOISE_LEVELS = [1, 5, 10]  # percent of field std

DATASETS = [
    'Fire_PINN_2D/data/fire_dataset_columns.csv',
    'Fire_PINN_Multi_variable/data/fire_dataset_columns.csv',
    'Fire_PINN_Multi_diffusion/data/fire_dataset_columns.csv',
    'Fire_PINN_Multi_variable_diffusion/data/fire_dataset_columns.csv',
]

np.random.seed(42)

for ds_path in DATASETS:
    folder = os.path.dirname(ds_path)
    model_name = ds_path.split('/')[0]
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Loading {ds_path} ...")
    df = pd.read_csv(ds_path)

    std_T = df['T'].std()
    std_F = df['F'].std()
    print(f"  std(T) = {std_T:.6f},  std(F) = {std_F:.6f}")
    n = len(df)

    for sigma_pct in NOISE_LEVELS:
        sigma_T = std_T * sigma_pct / 100.0
        sigma_F = std_F * sigma_pct / 100.0

        df_noisy = df.copy()
        df_noisy['T'] = df['T'] + np.random.normal(0, sigma_T, n)
        df_noisy['F'] = df['F'] + np.random.normal(0, sigma_F, n)

        out_name = f"fire_dataset_noise_{sigma_pct}.csv"
        out_path = os.path.join(folder, out_name)
        df_noisy.to_csv(out_path, index=False)
        print(f"  Saved {out_path}  (sigma={sigma_pct}%)")

print("\nDone. Generated 12 noisy datasets.")
