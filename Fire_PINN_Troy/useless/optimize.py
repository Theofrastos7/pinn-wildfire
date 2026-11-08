"""
Grid search over (k, starting_fuel, w1_scale, w2_scale) to minimise
the symmetric mean contour distance between simulated and real fire outlines.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import os
import glob
import csv
from PIL import Image
from scipy.ndimage import binary_fill_holes, distance_transform_edt

from functions_troy import *

# ── Fixed constants (not optimised) ─────────────────────────────────────────
A  = 100
B  = 300.0
Ta = 30.0
outside_fuel  = 0.6
beta = 0.50
lamb = 0.0022
TOP_FRACTION = 330 / 830
TOTAL_X_M = 2320.0
TOTAL_Y_M = 1420.0
SAVE_EVERY = 50
T_IGNITION = 70.0

# ── Baseline per-quadrant weights (will be multiplied by scale factors) ─────
w1_Q_base = np.array([0.05, 0.53, 0.05, 0.53])   # x-diffusion
w2_Q_base = np.array([0.05, 0.05, 0.28, 0.48])   # y-diffusion

# ── Parameter grid ──────────────────────────────────────────────────────────
# Narrow ranges around current values in main_troy.py
k_values             = [0.18, 0.22, 0.28, 0.34, 0.40]
starting_fuel_values = [0.15, 0.25, 0.30, 0.35, 0.45]
w1_scale_values      = [0.6, 0.8, 1.0, 1.2, 1.5]      # multiplier on all w1_Q
w2_scale_values      = [0.6, 0.8, 1.0, 1.2, 1.5]      # multiplier on all w2_Q

# ── Load Troy fire frames (once) ────────────────────────────────────────────
files = sorted(glob.glob("Fire_PINN_troy/troy_fire/*.gif"))
if not files:
    raise FileNotFoundError("No GIF files found in Fire_PINN_troy/troy_fire/")

ref_img = Image.open(files[0]).convert("RGB")
REF_W, REF_H = ref_img.size

frames = []
for img_path in files:
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    hours, mins = img_name.split("_")
    t_seconds = (int(hours) * 60 + int(mins)) * 60
    img = Image.open(img_path).convert("RGB")
    if img.size != (REF_W, REF_H):
        img = img.resize((REF_W, REF_H), Image.BILINEAR)
    arr = np.array(img)
    frames.append({
        'name':         img_name,
        't_seconds':    t_seconds,
        'temp_celsius': color_to_temp(arr),
    })

print(f"Loaded {len(frames)} frames")


# ── Metric ──────────────────────────────────────────────────────────────────

def mean_contour_distance(mask_a, mask_b):
    """Symmetric mean contour distance between two binary masks.
    For each boundary pixel of A, find distance to nearest boundary of B,
    then average.  Do the same B->A and return the mean of both."""
    from scipy.ndimage import binary_erosion
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    border_a = a & ~binary_erosion(a)
    border_b = b & ~binary_erosion(b)

    if not np.any(border_a) or not np.any(border_b):
        return np.inf

    # Distance transform of the complement gives distance-to-boundary
    dist_to_b = distance_transform_edt(~border_b)
    dist_to_a = distance_transform_edt(~border_a)

    d_a2b = dist_to_b[border_a].mean()
    d_b2a = dist_to_a[border_b].mean()
    return (d_a2b + d_b2a) / 2.0


# ── Simulation function ─────────────────────────────────────────────────────

def run_simulation(k, starting_fuel, w1_scale, w2_scale, return_fields=False):
    """Run the forward PDE and return the average mean-contour-distance
    across all frames (skipping the first frame = initial condition).
    If return_fields=True, also return (T_sol, t_saved) for plotting."""

    time_scale  = A / B
    space_scale = np.sqrt(A / (B * k))

    Ny, Nx = frames[0]['temp_celsius'].shape
    dx = (TOTAL_X_M / Nx) * space_scale
    dy = (TOTAL_Y_M / Ny) * space_scale

    T_total_seconds = frames[-1]['t_seconds'] - frames[0]['t_seconds']
    T_total = T_total_seconds * time_scale
    dt = 1.0 * (A / B)
    Nt = int(T_total / dt) + 1

    # Quadrant maps
    w1_Q = w1_Q_base * w1_scale
    w2_Q = w2_Q_base * w2_scale
    mid_x = Nx // 2
    mid_y = int(Ny * (1 - TOP_FRACTION))

    w1_map = np.zeros((Ny, Nx))
    w1_map[mid_y:, :mid_x] = w1_Q[0]
    w1_map[mid_y:, mid_x:] = w1_Q[1]
    w1_map[:mid_y, :mid_x] = w1_Q[2]
    w1_map[:mid_y, mid_x:] = w1_Q[3]

    w2_map = np.zeros((Ny, Nx))
    w2_map[mid_y:, :mid_x] = w2_Q[0]
    w2_map[mid_y:, mid_x:] = w2_Q[1]
    w2_map[:mid_y, :mid_x] = w2_Q[2]
    w2_map[:mid_y, mid_x:] = w2_Q[3]

    beta_map = np.full((Ny, Nx), beta)
    lamb_map = np.full((Ny, Nx), lamb)

    # Initial condition
    temp0 = frames[0]['temp_celsius'].copy()
    temp0[temp0 < T_IGNITION] = Ta
    T0 = (temp0 - Ta) / B
    F0 = np.where(temp0 >= T_IGNITION, starting_fuel, outside_fuel).astype(float)

    # Storage
    n_saved = Nt // SAVE_EVERY + 1
    T_sol = np.zeros((n_saved, Ny, Nx))
    t_saved = np.zeros(n_saved)
    T_sol[0] = T0
    t_saved[0] = 0.0

    Tn = T0.copy()
    Fn = F0.copy()
    save_idx = 1

    for n in range(1, Nt):
        T_diff = rk4_diffusion(Tn, 1, dt, dx, dy, w1_map, w2_map)
        T_next, F_next = reaction_step(T_diff, Fn, dt, lamb_map, beta_map)
        T_next = np.clip(T_next, 0, 20)
        F_next = np.clip(F_next, 0, 1)
        Tn = T_next
        Fn = F_next
        if n % SAVE_EVERY == 0 and save_idx < n_saved:
            T_sol[save_idx] = Tn
            t_saved[save_idx] = n * dt
            save_idx += 1

    T_sol = T_sol[:save_idx]
    t_saved = t_saved[:save_idx]

    # Compute mean contour distance for each frame (skip frame 0 = IC)
    # Only score up to 3_48 (3h48m = 13680s absolute)
    t0_abs = frames[0]['t_seconds']
    t_cutoff = (3 * 60 + 48) * 60  # 3_48 absolute seconds
    distances = []
    for frame in frames[1:]:
        if frame['t_seconds'] > t_cutoff:
            break
        dt_sec = frame['t_seconds'] - t0_abs
        if dt_sec > T_total_seconds:
            break
        t_nondim_target = dt_sec * time_scale
        snap_idx = np.argmin(np.abs(t_saved - t_nondim_target))

        T_sim_C = T_sol[snap_idx] * B + Ta
        T_real_C = frame['temp_celsius']

        mask_sim  = binary_fill_holes(T_sim_C  >= T_IGNITION).astype(float)
        mask_real = binary_fill_holes(T_real_C >= T_IGNITION).astype(float)

        d = mean_contour_distance(mask_sim, mask_real)
        distances.append(d)

    score = np.mean(distances) if distances else np.inf
    if return_fields:
        return score, T_sol, t_saved, time_scale, T_total_seconds, dx, dy
    return score


# ── Grid search ─────────────────────────────────────────────────────────────

combos = list(itertools.product(
    k_values, starting_fuel_values, w1_scale_values, w2_scale_values
))
total = len(combos)
print(f"\nStarting grid search: {total} combinations\n")

os.makedirs("Fire_PINN_Troy/simulation_results", exist_ok=True)
results_path = "Fire_PINN_Troy/simulation_results/optimize_results.csv"

best_score = np.inf
best_params = None
results = []

with open(results_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["k", "starting_fuel", "w1_scale", "w2_scale", "mean_contour_dist"])

    for i, (k_val, sf_val, w1s_val, w2s_val) in enumerate(combos):
        try:
            score = run_simulation(k_val, sf_val, w1s_val, w2s_val)
        except Exception as e:
            print(f"  [{i+1}/{total}] FAILED k={k_val} sf={sf_val} w1s={w1s_val} w2s={w2s_val}: {e}")
            score = np.inf

        results.append((k_val, sf_val, w1s_val, w2s_val, score))
        writer.writerow([k_val, sf_val, w1s_val, w2s_val, f"{score:.4f}"])
        f.flush()

        tag = ""
        if score < best_score:
            best_score = score
            best_params = (k_val, sf_val, w1s_val, w2s_val)
            tag = "  <-- NEW BEST"

        if (i + 1) % 10 == 0 or tag:
            print(f"  [{i+1}/{total}] k={k_val}  sf={sf_val}  w1s={w1s_val}  w2s={w2s_val}  "
                  f"dist={score:.2f}{tag}")

# ── Report ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("OPTIMISATION COMPLETE")
print("=" * 65)
print(f"  Best mean contour distance: {best_score:.4f}")
print(f"  k              = {best_params[0]}")
print(f"  starting_fuel  = {best_params[1]}")
print(f"  w1_scale       = {best_params[2]}  (w1_Q = {w1_Q_base * best_params[2]})")
print(f"  w2_scale       = {best_params[3]}  (w2_Q = {w2_Q_base * best_params[3]})")
print(f"\nResults saved to: {results_path}")
print("=" * 65)

# ── Re-run best params and save outline plots ──────────────────────────────
print("\nGenerating outline plots for best parameters...")
out_dir = "Fire_PINN_Troy/simulation_results/optimize"
os.makedirs(out_dir, exist_ok=True)

_, T_sol, t_saved, time_scale, T_total_seconds, dx, dy = run_simulation(
    *best_params, return_fields=True
)

Ny, Nx = frames[0]['temp_celsius'].shape
x = np.linspace(0, (Nx - 1) * dx, Nx)
y = np.linspace(0, (Ny - 1) * dy, Ny)
t0_abs = frames[0]['t_seconds']

for frame in frames[1:]:
    dt_sec = frame['t_seconds'] - t0_abs
    if dt_sec > T_total_seconds:
        break
    t_nondim_target = dt_sec * time_scale
    snap_idx = np.argmin(np.abs(t_saved - t_nondim_target))

    T_sim_C = T_sol[snap_idx] * B + Ta
    T_real_C = frame['temp_celsius']

    mask_sim  = binary_fill_holes(T_sim_C  >= T_IGNITION).astype(float)
    mask_real = binary_fill_holes(T_real_C >= T_IGNITION).astype(float)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f"Optimised Outline — {frame['name']}  (t+{dt_sec}s)", fontsize=14)

    ax.contour(x, y, mask_sim,  levels=[0.5], colors='blue', linewidths=2)
    ax.contour(x, y, mask_real, levels=[0.5], colors='red',  linewidths=2)

    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.legend(
        [plt.Line2D([0], [0], color='blue', lw=2),
         plt.Line2D([0], [0], color='red',  lw=2)],
        ['Simulated', 'Real'], loc='upper right'
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{frame['name']}_outline.png", dpi=150, bbox_inches='tight')
    plt.close()

print(f"Outline plots saved to: {out_dir}/")
