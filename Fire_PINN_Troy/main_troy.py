"""
Forward simulation of the Troy fire with initial condition from the first thermal image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_fill_holes
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

#k  = np.sqrt(w1**2 + w2**2) / 10  # thermal diffusivity as magnitude of (w1, w2)
#print(f"Thermal diffusivity k = {k:.4f}")
#print("previous k =  0.28")

B    = 300.0
A    = 100
Ta   = 30.0

starting_fuel = 0.3   # initial fuel fraction in burning area
outside_fuel  = 0.6   # initial fuel fraction outside the fire

# Per-quadrant diffusion weights 
#          top-left  top-right  bottom-left  bottom-right
w1_Q   = [  0.05,      0.53,       0.05,         0.53  ]   # x-diffusion 
w2_Q   = [  0.05,      0.05,       0.28,         0.48  ]   # y-diffusion 

# PDE reaction parameters
beta = 0.50     # fuel consumption rate 
lamb = 0.0022   # cooling coefficient 

# Quadrant split: top portion is 330/830 of total height 
TOP_FRACTION = 330 / 830

# Physical domain size meters
TOTAL_X_M = 2320.0
TOTAL_Y_M = 1420.0

# Total simulation time and save interval
T_total_seconds = None    # None, cover all frames automatically
SAVE_EVERY = 50           

# ── Load Troy fire frames ───────────────────────────────────────────────────
# GIF filenames encode time as "HH_MM.gif" (hours_minutes of the day)
files = sorted(glob.glob("Fire_PINN_troy/troy_fire/*.gif"))
if not files:
    raise FileNotFoundError("No GIF files found in Fire_PINN_troy/troy_fire/")

# Use first image as reference size, resize others to match if needed
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
        't_min':        t_seconds // 60,
        'temp_celsius': color_to_temp(arr),   # RGB to Celsius via anchor colormap
        'raw_rgb':      arr,
    })

print(f"Loaded {len(frames)} frames")
print(f"Time range: {frames[0]['t_min']}min - {frames[-1]['t_min']}min")

# ── Non-dimensional scales ───────────────────────────────────────────────────
time_scale  = A / B
space_scale = np.sqrt(A / (B * k))

Ny, Nx = frames[0]['temp_celsius'].shape   # image rows = Ny, cols = Nx
dx = (TOTAL_X_M / Nx) * space_scale        # non-dim 
dy = (TOTAL_Y_M / Ny) * space_scale       

print(f"Grid: {Nx} x {Ny} (dx={dx:.4f}, dy={dy:.4f} non-dim)")

# ── Non-dimensional time ─────────────────────────────────────────────────────
if T_total_seconds is None:
    T_total_seconds = frames[-1]['t_seconds'] - frames[0]['t_seconds']
T_total = T_total_seconds * time_scale

original_dt = 1               # real-time step in seconds
dt = original_dt * (A / B)    # non-dim time step

Nt = int(T_total / dt) + 1
print(f"original_dt = {original_dt}, using dt = {dt:.6f}")
print(f"Total nondim time: {T_total:.2f}, Nt = {Nt}")

# Build spatially varying weight maps for 4 quadrants
mid_x = Nx // 2
mid_y = int(Ny * (1 - TOP_FRACTION))   # bottom portion is larger

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

print(f"Quadrant split: mid_y={mid_y}/{Ny} (top={Ny-mid_y}px), mid_x={mid_x}/{Nx}")


# ── Initial condition from first Troy frame ─────────────────────────────────
T_IGNITION = 70.0   # below this treat as ambient to kills false warm pixels, otherwise simulation fails
temp0 = frames[0]['temp_celsius'].copy()
temp0[temp0 < T_IGNITION] = Ta

T0 = (temp0 - Ta) / B    # non-dim temperature
F0 = np.where(temp0 >= T_IGNITION, starting_fuel, outside_fuel).astype(float)

# PDE operator splitting

# Store every SAVE_EVERY (default 50) step (memory efficient)
n_saved = Nt // SAVE_EVERY + 1
T_sol = np.zeros((n_saved, Ny, Nx))
F_sol = np.zeros((n_saved, Ny, Nx))
t_saved = np.zeros(n_saved)

T_sol[0] = T0
F_sol[0] = F0
t_saved[0] = 0.0

Tn = T0.copy()
Fn = F0.copy()
save_idx = 1

for n in range(1, Nt):
    # Step 1: RK4 diffusion (k=1 because it's absorbed into dx/dy via non-dim)
    T_diff = rk4_diffusion(Tn, 1, dt, dx, dy, w1_map, w2_map)

    # Step 2: reaction + cooling (semi-implicit for fuel stability)
    T_next, F_next = reaction_step(T_diff, Fn, dt, lamb_map, beta_map)

    # Clamp to physical bounds
    T_next = np.clip(T_next, 0, 20)
    F_next = np.clip(F_next, 0, 1)

    Tn = T_next
    Fn = F_next

    if n % SAVE_EVERY == 0 and save_idx < n_saved:
        T_sol[save_idx] = Tn
        F_sol[save_idx] = Fn
        t_saved[save_idx] = n * dt
        save_idx += 1

    if n % (Nt // 10) == 0:
        print(f"Progress: {100 * n // Nt}%")

# Trim in case we allocated extra
T_sol = T_sol[:save_idx]
F_sol = F_sol[:save_idx]
t_saved = t_saved[:save_idx]

print(f"Simulation done. Saved {save_idx} snapshots.")
print(f"Temperature range: [{T_sol.min():.4f}, {T_sol.max():.4f}]")
print(f"Fuel range: [{F_sol.min():.4f}, {F_sol.max():.4f}]")

# Plot: side-by-side simulated vs real for each frame 
x = np.linspace(0, (Nx - 1) * dx, Nx)
y = np.linspace(0, (Ny - 1) * dy, Ny)

t0_abs = frames[0]['t_seconds']
snap_frames = [f for f in frames if f['t_seconds'] - t0_abs <= T_total_seconds]
n = len(snap_frames)

# create output directory if it doesn't exist
os.makedirs("Fire_PINN_troy/simulation_results", exist_ok=True)

# One figure per frame
for i, frame in enumerate(snap_frames):
    dt_sec = frame['t_seconds'] - t0_abs
    t_nondim_target = dt_sec * time_scale
    snap_idx = np.argmin(np.abs(t_saved - t_nondim_target))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{frame['name']}  (t+{dt_sec}s)", fontsize=14)

    T_sim_C = T_sol[snap_idx] * B + Ta   # convert back to Celsius for display
    im1 = ax1.imshow(T_sim_C, cmap=fire_cmap, vmin=0, vmax=650,
                      origin='lower', extent=[0, x[-1], 0, y[-1]])
    ax1.set_title('Simulated')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(frame['temp_celsius'], cmap=fire_cmap, vmin=0, vmax=650,
                      origin='lower', extent=[0, x[-1], 0, y[-1]])
    ax2.set_title('Real')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    os.makedirs("Fire_PINN_troy/simulation_results/comparisons", exist_ok=True)
    plt.savefig(f"Fire_PINN_troy/simulation_results/comparisons/{frame['name']}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

# Fire outline plots: simulated vs real contours overlaid
os.makedirs("Fire_PINN_troy/simulation_results/outlines", exist_ok=True)

for i, frame in enumerate(snap_frames):
    dt_sec = frame['t_seconds'] - t0_abs
    t_nondim_target = dt_sec * time_scale
    snap_idx = np.argmin(np.abs(t_saved - t_nondim_target))

    T_sim_C = T_sol[snap_idx] * B + Ta
    T_real_C = frame['temp_celsius']

    # Create binary masks and fill interior holes so only the outer perimeter remains
    mask_sim = binary_fill_holes(T_sim_C >= T_IGNITION).astype(float)
    mask_real = binary_fill_holes(T_real_C >= T_IGNITION).astype(float)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f"Fire Outline — {frame['name']}  (t+{dt_sec}s)", fontsize=14)

    ax.contour(x, y, mask_sim, levels=[0.5], colors='blue', linewidths=2)
    ax.contour(x, y, mask_real, levels=[0.5], colors='red', linewidths=2)

    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.legend(
        [plt.Line2D([0], [0], color='blue', lw=2),
         plt.Line2D([0], [0], color='red', lw=2)],
        ['Simulated', 'Real'], loc='upper right'
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"Fire_PINN_troy/simulation_results/outlines/{frame['name']}_outline.png",
                dpi=150, bbox_inches='tight')
    plt.close()
