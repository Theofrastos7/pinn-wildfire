import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# ── Color-to-temperature anchor points ───────────────────────────────────────
# Based on the scale provided by Pacific Southwest Research Station for Troy Fire
ANCHORS_RGB = np.array([
    [0,   0,   0  ],   # 0 C   - black
    [76,  65,  52 ],   # 50 C  - dark warm brown
    [100, 81,  84 ],   # 60 C  - brownish-tan
    [136, 19,  22 ],   # 72 C  - dark red
    [179, 2,   2  ],   # 75 C  - red
    [255, 0,   0  ],   # 120 C - bright red
    [253, 85,  1  ],   # 200 C - orange
    [255, 204, 0  ],   # 350 C - yellow-orange
    [255, 255, 0  ],   # 400 C - yellow
    [255, 255, 204],   # 580 C - light cream
    [255, 255, 255],   # 650 C - white (>600 C)
], dtype=float)

ANCHORS_TEMP = np.array([0, 50, 60, 72, 75, 120, 200, 350, 400, 580, 650],
                         dtype=float)

# Shared colormap for plotting fire temperatures
fire_cmap = mcolors.LinearSegmentedColormap.from_list(
    "fire", [(c / 255).tolist() for c in ANCHORS_RGB], N=512)


def color_to_temp(rgb_array):
    """
    Convert an RGB thermal image to a 2D temperature map (Celsius).
    For each pixel, finds the closest point on the piecewise-linear anchor
    colormap in RGB space and returns the corresponding temperature.
    """
    H, W = rgb_array.shape[:2]
    pts  = rgb_array.reshape(-1, 3).astype(float)
    N    = pts.shape[0]
    n_seg = len(ANCHORS_RGB) - 1
    best_t_global = np.zeros(N)
    best_dist     = np.full(N, np.inf)
    for i in range(n_seg):
        a, b = ANCHORS_RGB[i], ANCHORS_RGB[i + 1]
        ab   = b - a
        ab_sq = np.dot(ab, ab)
        # Project each pixel onto the line segment a->b, clamp to [0,1]
        t = np.zeros(N) if ab_sq < 1e-12 else np.clip(np.dot(pts - a, ab) / ab_sq, 0, 1)
        proj = a + t[:, None] * ab
        dist = np.linalg.norm(pts - proj, axis=1)
        better = dist < best_dist
        best_dist     = np.where(better, dist, best_dist)
        best_t_global = np.where(better, i + t, best_t_global)
    # Interpolate temperature from the best-matching segment
    seg_idx = np.clip(best_t_global.astype(int), 0, n_seg - 1)
    frac    = best_t_global - seg_idx
    T_low   = ANCHORS_TEMP[seg_idx]
    T_high  = ANCHORS_TEMP[np.minimum(seg_idx + 1, n_seg)]
    return (T_low + frac * (T_high - T_low)).reshape(H, W)


def temp_to_color(temp_array):
    """
    Convert a 2D temperature map (Celsius) to an RGB image using the anchor colormap.
    Linearly interpolates between anchor points to produce a fire-like color gradient.
    """
    T = np.clip(temp_array, ANCHORS_TEMP[0], ANCHORS_TEMP[-1])
    H, W = T.shape
    T_flat = T.ravel()
    rgb = np.zeros((T_flat.size, 3), dtype=float)
    n_seg = len(ANCHORS_TEMP) - 1
    for i in range(n_seg):
        t_lo, t_hi = ANCHORS_TEMP[i], ANCHORS_TEMP[i + 1]
        mask = (T_flat >= t_lo) & (T_flat <= t_hi)
        if not np.any(mask):
            continue
        frac = (T_flat[mask] - t_lo) / (t_hi - t_lo + 1e-12)
        rgb[mask] = ANCHORS_RGB[i] * (1 - frac[:, None]) + ANCHORS_RGB[i + 1] * frac[:, None]
    return np.clip(rgb, 0, 255).astype(np.uint8).reshape(H, W, 3)


def weighted_laplacian_var(Z, dx, dy, w1m, w2m):
    """Compute w1*d²Z/dx² + w2*d²Z/dy² with spatially varying weights.
    Uses central differences on interior points and Neumann (zero-flux) BCs.
    w1m controls x-diffusion (axis 1, columns), w2m controls y-diffusion (axis 0, rows)."""
    L = np.zeros_like(Z)
    # axis 1 (columns) = x direction, divided by dx²
    # axis 0 (rows)    = y direction, divided by dy²
    L[1:-1, 1:-1] = (w1m[1:-1, 1:-1] * (Z[1:-1, 2:] - 2*Z[1:-1, 1:-1] + Z[1:-1, :-2]) / dx**2
                    + w2m[1:-1, 1:-1] * (Z[2:, 1:-1] - 2*Z[1:-1, 1:-1] + Z[:-2, 1:-1]) / dy**2)
    # Neumann BCs: copy interior neighbor to boundary
    L[0, :] = L[1, :]; L[-1, :] = L[-2, :]
    L[:, 0] = L[:, 1]; L[:, -1] = L[:, -2]
    return L


def rk4_diffusion(Tn, k, dt, dx, dy, w1m, w2m):
    """Advance temperature one step via RK4 for the diffusion term only.
    k is the base diffusion coefficient (set to 1 when absorbed into non-dim coords).
    Applies Neumann BCs after the step."""
    k1 = k * weighted_laplacian_var(Tn, dx, dy, w1m, w2m)
    k2 = k * weighted_laplacian_var(Tn + 0.5*dt*k1, dx, dy, w1m, w2m)
    k3 = k * weighted_laplacian_var(Tn + 0.5*dt*k2, dx, dy, w1m, w2m)
    k4 = k * weighted_laplacian_var(Tn + dt*k3, dx, dy, w1m, w2m)
    T_new = Tn + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    T_new[0, :] = T_new[1, :]; T_new[-1, :] = T_new[-2, :]
    T_new[:, 0] = T_new[:, 1]; T_new[:, -1] = T_new[:, -2]
    return T_new


def reaction_step(Tn, Fn, dt, lamb_m, beta_m):
    """Semi-implicit reaction update (operator-split from diffusion).
    T advances explicitly: T += dt*(F*exp(-1/T) - lamb*T)
    F advances semi-implicitly: F = F / (1 + dt*beta*exp(-1/T))
    Semi-implicit F update is unconditionally stable for any dt."""
    exp_term = np.exp(-1.0 / (Tn + 1e-8))
    T_new = Tn + dt * (Fn * exp_term - lamb_m * Tn)
    F_new = Fn / (1 + dt * beta_m * exp_term)
    return T_new, F_new
