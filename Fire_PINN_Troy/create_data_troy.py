import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

from functions_troy import *


# ── PDE parameters ──────────────────────────────────────────────────────────
# k: thermal diffusivity
# B reference temperature for non-dimensionalization 
# A: reaction pre-factor, also sets time scale 
# Ta: ambient temperature (Celsius)

w1 = 3.31
w2 = 1.21
k = 0.28

#k  = np.sqrt(w1**2 + w2**2)   # thermal diffusivity as magnitude of (w1, w2)


B     = 300.0
A     = 100
Ta    = 30.0

starting_fuel = 0.3   # initial fuel fraction in burning area
outside_fuel  = 0.6   # initial fuel fraction outside the fire

# PDE reaction parameters
beta = 0.5     # fuel consumption rate 
lamb = 0.0022   # cooling coefficient
# Per-quadrant diffusion weights (split at mid_x, mid_y)
#          top-left  top-right  bottom-left  bottom-right
w1_Q   = [  0.05,      0.53,       0.05,         0.53  ]   # x-diffusion 
w2_Q   = [  0.05,      0.05,       0.28,         0.48  ]   # y-diffusion 

TOP_FRACTION = 330 / 830

# Physical domain size meters
TOTAL_X_M = 2320.0
TOTAL_Y_M = 1420.0

# Non-dimensional scale factors
space_scale = np.sqrt(A / (B * k))

print(f"Time scale factor A/B = {A/B:.4f}")
print(f"Space scale factor sqrt(A/(B*k)) = {space_scale:.4f}")

# ── Integration method ───────────────────────────────────────────────────────
# 1 = Semi-implicit  
# 2 = Explicit Euler (1st order)
# 3 = Heun / RK2     (2nd order)
# 4 = Exponential    (exact solution)
INTEGRATION_METHOD = 1

# fire_cmap is imported from functions_troy
fire_norm = plt.Normalize(ANCHORS_TEMP[0], ANCHORS_TEMP[-1])

# ── Process all images ───────────────────────────────────────────────────────
files = sorted(glob.glob("Fire_PINN_troy/troy_fire/*.gif"))
if not files:
    raise FileNotFoundError("No GIF files found in Fire_PINN_troy/troy_fire/")

os.makedirs("Fire_PINN_troy/data", exist_ok=True)

ref_img = Image.open(files[0]).convert("RGB")
REF_W, REF_H = ref_img.size

fuel_map = None
prev_t   = None
all_dfs  = []

for i, img_path in enumerate(files):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\nProcessing: {img_path}")

    img = Image.open(img_path).convert("RGB")
    if img.size != (REF_W, REF_H):
        img = img.resize((REF_W, REF_H), Image.BILINEAR)
    arr = np.array(img)
    H, W = arr.shape[:2]
    print(f"Image size: {W}x{H}")

    # Pixel to temperature
    temp_map = color_to_temp(arr)

    # Parse time from filename
    hours, mins = img_name.split("_")
    t_seconds = (int(hours) * 60 + int(mins)) * 60
    t_nondim = t_seconds * (A / B)

    # Compute fuel map via ODE
    if fuel_map is None:
        fuel_map = np.where(temp_map >= 70.0, starting_fuel, outside_fuel).astype(float)
        dt = 0.0
    else:
        dt = float(t_nondim - prev_t)

    T_nondim = (temp_map - Ta) / B

    if dt > 0:
        T_safe   = np.maximum(T_nondim, 1e-8)
        exp_term = np.exp(-1.0 / T_safe)

        if INTEGRATION_METHOD == 1:
            fuel_map = fuel_map / (1.0 + dt * beta * exp_term)

        elif INTEGRATION_METHOD == 2:
            fuel_map = fuel_map - dt * beta * fuel_map * exp_term

        elif INTEGRATION_METHOD == 3:
            k1 = -beta * fuel_map * exp_term
            k2 = -beta * np.clip(fuel_map + dt * k1, 0.0, 1.0) * exp_term
            fuel_map = fuel_map + dt / 2.0 * (k1 + k2)

        elif INTEGRATION_METHOD == 4:
            fuel_map = fuel_map * np.exp(-dt * beta * exp_term)
        fuel_map = np.clip(fuel_map, 0.0, 1.0)

    prev_t = t_nondim

    # Build CSV ─
    ys, xs = np.mgrid[0:H, 0:W]

    pixel_size_x = TOTAL_X_M / W
    pixel_size_y = TOTAL_Y_M / H
    xs_nondim = xs * pixel_size_x * space_scale
    ys_nondim = ys * pixel_size_y * space_scale

    n = H * W

    def param_col(val):
        return [val] + [''] * (n - 1)

    df = pd.DataFrame({
        "x": xs_nondim.ravel(),
        "y": ys_nondim.ravel(),
        "t": t_nondim,
        "T": T_nondim.ravel(),
        "F": fuel_map.ravel(),
        "beta":  param_col(beta),
        "lamb":  param_col(lamb),
        "Ta":    param_col(Ta),
        # Q1=top-right, Q2=top-left, Q3=bottom-left, Q4=bottom-right
        # w1_Q indices: [0]=top-left, [1]=top-right, [2]=bottom-left, [3]=bottom-right
        "w1_q1": param_col(w1_Q[1]), "w2_q1": param_col(w2_Q[1]),  # Q1=top-right
        "w1_q2": param_col(w1_Q[0]), "w2_q2": param_col(w2_Q[0]),  # Q2=top-left
        "w1_q3": param_col(w1_Q[2]), "w2_q3": param_col(w2_Q[2]),  # Q3=bottom-left
        "w1_q4": param_col(w1_Q[3]), "w2_q4": param_col(w2_Q[3]),  # Q4=bottom-right
    })

    all_dfs.append(df)
    print(f"Processed: {img_name}  ({len(df):,} rows)")
    print(f"x range: {xs_nondim.min():.2f} - {xs_nondim.max():.2f} (non-dim)")
    print(f"y range: {ys_nondim.min():.2f} - {ys_nondim.max():.2f} (non-dim)")
    print(f"Temperature range: {T_nondim.min():.4f} - {T_nondim.max():.4f} (non-dim)")
    print(f"Fuel range: {fuel_map.min():.4f} - {fuel_map.max():.4f}")

    # Plot every 12 images
    if i % 12 == 0:
        temp_grid = df.pivot(index="y", columns="x", values="T").values
        reconstructed = temp_to_color(temp_grid * B + Ta)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f"Thermal image -> CSV -> reconstructed  |  {img_name}.gif",
                     fontsize=13)

        axes[0].imshow(arr)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(reconstructed)
        axes[1].set_title("Reconstructed from CSV")
        axes[1].axis("off")
        sm = plt.cm.ScalarMappable(cmap=fire_cmap, norm=fire_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Temperature (C)")

        im_fuel = axes[2].imshow(fuel_map, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title("Fuel map")
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        fig.colorbar(im_fuel, ax=axes[2])

        plt.tight_layout()
        plt.show()


merged_df = pd.concat(all_dfs, ignore_index=True)
merged_df.to_csv("Fire_PINN_troy/data/forward_dataset.csv", index=False)
print(f"Saved: Fire_PINN_troy/data/forward_dataset.csv  ({len(merged_df):,} rows)")
