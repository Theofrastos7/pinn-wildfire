"""
PDE dataset generator: run forward PDE simulation from the first thermal image.
Takes the first frame (1:46) as IC, simulates 1 hour, saves every 1 minute.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

from functions_troy import *


# ── PDE parameters (synced with main_troy.py) ────────────────────────────────
k = 0.28
B     = 300.0
A     = 100
Ta    = 30.0

starting_fuel = 0.3
outside_fuel  = 0.6

beta = 0.5
lamb = 0.0022

#          top-left  top-right  bottom-left  bottom-right
w1_Q   = [  0.05,      0.53,       0.05,         0.53  ]
w2_Q   = [  0.05,      0.05,       0.28,         0.48  ]

TOP_FRACTION = 330 / 830

TOTAL_X_M = 2320.0
TOTAL_Y_M = 1420.0

time_scale  = A / B
space_scale = np.sqrt(A / (B * k))

print(f"Time scale factor A/B = {time_scale:.4f}")
print(f"Space scale factor sqrt(A/(B*k)) = {space_scale:.4f}")

# ── Load first image for initial condition ────────────────────────────────────
files = sorted(glob.glob("Fire_PINN_troy/troy_fire/*.gif"))
if not files:
    raise FileNotFoundError("No GIF files found in Fire_PINN_troy/troy_fire/")

os.makedirs("Fire_PINN_troy/data", exist_ok=True)

first_img = Image.open(files[0]).convert("RGB")
REF_W, REF_H = first_img.size
first_arr = np.array(first_img)
H, W = first_arr.shape[:2]
Ny, Nx = H, W

# Grid spacing (non-dimensional)
pixel_size_x = TOTAL_X_M / Nx
pixel_size_y = TOTAL_Y_M / Ny
dx = pixel_size_x * space_scale
dy = pixel_size_y * space_scale

# Coordinate arrays
ys_grid, xs_grid = np.mgrid[0:Ny, 0:Nx]
xs_nondim_grid = xs_grid * pixel_size_x * space_scale
ys_nondim_grid = ys_grid * pixel_size_y * space_scale

# Spatially varying weight maps (4 quadrants)
mid_x = Nx // 2
mid_y = int(Ny * (1 - TOP_FRACTION))

w1_map = np.zeros((Ny, Nx))
w1_map[mid_y:, :mid_x] = w1_Q[0]   # top-left
w1_map[mid_y:, mid_x:] = w1_Q[1]   # top-right
w1_map[:mid_y, :mid_x] = w1_Q[2]   # bottom-left
w1_map[:mid_y, mid_x:] = w1_Q[3]   # bottom-right

w2_map = np.zeros((Ny, Nx))
w2_map[mid_y:, :mid_x] = w2_Q[0]
w2_map[mid_y:, mid_x:] = w2_Q[1]
w2_map[:mid_y, :mid_x] = w2_Q[2]
w2_map[:mid_y, mid_x:] = w2_Q[3]

beta_map = np.full((Ny, Nx), beta)
lamb_map = np.full((Ny, Nx), lamb)

print(f"Grid: {Nx} x {Ny} (dx={dx:.4f}, dy={dy:.4f} non-dim)")
print(f"Quadrant split: mid_y={mid_y}/{Ny}, mid_x={mid_x}/{Nx}")

# Initial condition from 1:46 thermal image
T_IGNITION = 70.0
temp0 = color_to_temp(first_arr)
temp0[temp0 < T_IGNITION] = Ta
T0 = (temp0 - Ta) / B
F0 = np.where(temp0 >= T_IGNITION, starting_fuel, outside_fuel).astype(float)

# ── Simulation setup ─────────────────────────────────────────────────────────
# First frame time (for absolute t in CSV)
fname0 = os.path.splitext(os.path.basename(files[0]))[0]
hours0, mins0 = fname0.split("_")
t_start_sec = (int(hours0) * 60 + int(mins0)) * 60  # 1:46 = 6360s

TOTAL_SIM_SEC = 3600    # 1 hour
DT_SEC = 10             # 10 seconds per PDE step
SAVE_EVERY_SEC = 60     # save every 1 minute

sim_dt = DT_SEC * time_scale
n_steps = TOTAL_SIM_SEC // DT_SEC

print(f"\nSimulation: {TOTAL_SIM_SEC}s (1 hour), dt={DT_SEC}s, save every {SAVE_EVERY_SEC}s")
print(f"dt_nondim = {sim_dt:.6f}, steps = {n_steps}, snapshots = {TOTAL_SIM_SEC // SAVE_EVERY_SEC + 1}")

# ── Run simulation ────────────────────────────────────────────────────────────
Tn = T0.copy()
Fn = F0.copy()

pde_dfs = []
n_pts = Ny * Nx
snap_times_sec = []

def param_col(val):
    return [val] + [''] * (n_pts - 1)

def save_snapshot(T_arr, F_arr, elapsed_sec):
    abs_t_nondim = (t_start_sec + elapsed_sec) * time_scale
    df = pd.DataFrame({
        "x": xs_nondim_grid.ravel(),
        "y": ys_nondim_grid.ravel(),
        "t": abs_t_nondim,
        "T": T_arr.ravel(),
        "F": F_arr.ravel(),
        "beta": param_col(beta),
        "lamb": param_col(lamb),
        "Ta":   param_col(Ta),
        "w1_q1": param_col(w1_Q[1]), "w2_q1": param_col(w2_Q[1]),
        "w1_q2": param_col(w1_Q[0]), "w2_q2": param_col(w2_Q[0]),
        "w1_q3": param_col(w1_Q[2]), "w2_q3": param_col(w2_Q[2]),
        "w1_q4": param_col(w1_Q[3]), "w2_q4": param_col(w2_Q[3]),
    })
    pde_dfs.append(df)
    snap_times_sec.append(elapsed_sec)

# Save initial state (t=0)
save_snapshot(Tn, Fn, 0)
print(f"  Saved snapshot 1 at t=0s (IC: {fname0})")

for n in range(1, n_steps + 1):
    T_diff = rk4_diffusion(Tn, 1, sim_dt, dx, dy, w1_map, w2_map)
    T_next, F_next = reaction_step(T_diff, Fn, sim_dt, lamb_map, beta_map)
    Tn = np.clip(T_next, 0, 20)
    Fn = np.clip(F_next, 0, 1)

    elapsed_sec = n * DT_SEC
    if elapsed_sec % SAVE_EVERY_SEC == 0:
        save_snapshot(Tn, Fn, elapsed_sec)

    if n % (n_steps // 10) == 0:
        print(f"  Progress: {100 * n // n_steps}%  T=[{Tn.min():.4f}, {Tn.max():.4f}]")

print(f"  Saved {len(pde_dfs)} snapshots total")

# ── Save CSV ────────────────────────────────────────────────────────────────
pde_merged = pd.concat(pde_dfs, ignore_index=True)
pde_merged.to_csv("Fire_PINN_troy/data/pde_dataset.csv", index=False)
print(f"\nSaved: Fire_PINN_troy/data/pde_dataset.csv  ({len(pde_merged):,} rows)")
print(f"Temperature range: [{pde_merged['T'].min():.4f}, {pde_merged['T'].max():.4f}]")
print(f"Fuel range: [{pde_merged['F'].min():.4f}, {pde_merged['F'].max():.4f}]")

# ── Save PDE snapshot images (every 5 snapshots = every 5 minutes) ────────────
pde_img_dir = "Fire_PINN_troy/data/pdes"
os.makedirs(pde_img_dir, exist_ok=True)

x_coords = np.linspace(0, (Nx - 1) * dx, Nx)
y_coords = np.linspace(0, (Ny - 1) * dy, Ny)

for snap_idx, sec in enumerate(snap_times_sec):
    if snap_idx % 5 != 0:
        continue

    df_snap = pde_dfs[snap_idx]
    T_grid = df_snap.pivot(index="y", columns="x", values="T").values
    F_grid = df_snap.pivot(index="y", columns="x", values="F").values
    t_val = df_snap["t"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"PDE Simulation — t+{sec}s  (t_nondim={t_val:.1f})", fontsize=14)

    im1 = ax1.imshow(T_grid * B + Ta, cmap=fire_cmap, vmin=0, vmax=650,
                     origin='lower', extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    ax1.set_title("Temperature (°C)")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(F_grid, cmap='viridis', vmin=0, vmax=1,
                     origin='lower', extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    ax2.set_title("Fuel")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(pde_img_dir, f"pde_t{sec}s.png"), dpi=150, bbox_inches='tight')
    plt.close()

print(f"Saved PDE snapshot images (every 5 snapshots) to {pde_img_dir}/")
