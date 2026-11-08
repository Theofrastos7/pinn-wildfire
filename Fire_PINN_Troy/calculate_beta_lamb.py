"""
Fire PDE - Direct analytical computation of beta and lambda
w1, w2 configurable (fixed diffusion coefficients)

PDE:
  dT/dt = w1*T_xx + w2*T_yy + F*exp(-1/T) - lamb*T
  dF/dt = -beta*F*exp(-1/T)

Rearranging:
  beta = -F_t / (F * exp(-1/T))
  lamb = (-T_t + w1*T_xx + w2*T_yy + F*exp(-1/T)) / T

All derivatives computed via finite differences on the gridded data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

np.random.seed(1234)

# ── Physical constants  ────────────────
A  = 100      # reaction pre-factor
B  = 300.0    # reference temperature for non-dimensionalization

# ── Diffusion coefficients  ───────────────────────────────
w1 = 1.75
w2 = 1.072
k = 0.28
#k  = np.sqrt(w1**2 + w2**2)   # thermal diffusivity as magnitude of (w1, w2)

w_scale = np.sqrt(A * k / B)

w1 = w_scale * w1
w2 = w_scale * w2

dataset = 'Fire_PINN_troy/data/forward_dataset.csv'
os.makedirs("Fire_PINN_Troy/graphs_calculate", exist_ok=True)

# Load Data
# The CSV contains non-dimensionalized data from create_data_forward.py:
#   x, y: spatial coords 
#   t: time 
#   T: temperature 
#   F: fuel fraction [0, 1]
#   beta, lamb
print("Loading fire dataset...")
data_df = pd.read_csv(dataset)

x_all = data_df['x'].values
y_all = data_df['y'].values
t_all = data_df['t'].values
T_all = data_df['T'].values
F_all = data_df['F'].values

# Read expected values from CSV metadata used only for comparison
beta_expected = data_df['beta'].iloc[0]
lamb_expected = data_df['lamb'].iloc[0]
print(f"Expected parameters: beta={beta_expected}, lamb={lamb_expected}")
print(f"Fixed diffusion: w1={w1}, w2={w2}")

# Build structured grid
# The CSV is a flattened grid: each row is one (x, y, t) point.
# Reconstruct the 3D arrays T_grid[t, y, x] and F_grid[t, y, x].
x_unique = np.sort(np.unique(x_all))
y_unique = np.sort(np.unique(y_all))
t_unique = np.sort(np.unique(t_all))
nx, ny, nt = len(x_unique), len(y_unique), len(t_unique)

dx = x_unique[1] - x_unique[0]   # uniform x spacing (non-dim)
dy = y_unique[1] - y_unique[0]   # uniform y spacing (non-dim)
dt_vals = np.diff(t_unique)      # time steps

print(f"Grid: nx={nx}, ny={ny}, nt={nt}")
print(f"dx={dx:.6f}, dy={dy:.6f}")
print(f"dt range: [{dt_vals.min():.1f}, {dt_vals.max():.1f}]")

# Map each flat data point to its (t, y, x) index in the 3D grid
x_idx = np.searchsorted(x_unique, x_all)
y_idx = np.searchsorted(y_unique, y_all)
t_idx = np.searchsorted(t_unique, t_all)

T_grid = np.full((nt, ny, nx), np.nan)
F_grid = np.full((nt, ny, nx), np.nan)
T_grid[t_idx, y_idx, x_idx] = T_all
F_grid[t_idx, y_idx, x_idx] = F_all

print(f"T range: [{np.nanmin(T_grid):.6f}, {np.nanmax(T_grid):.6f}]")
print(f"F range: [{np.nanmin(F_grid):.6f}, {np.nanmax(F_grid):.6f}]")

# Finite differences
# Compute all partial derivatives needed for the PDE equations.
# Spatial derivatives use central differences (2nd order accurate),
# time derivatives use forward differences (1st order).
# Boundary points are left as NaN since central diff needs neighbors on both sides.

# T_xx = d²T/dx²: axis 2 = x (columns)
T_xx = np.full_like(T_grid, np.nan)
T_xx[:, :, 1:-1] = (T_grid[:, :, 2:] - 2.0 * T_grid[:, :, 1:-1] + T_grid[:, :, :-2]) / (dx ** 2)

# T_yy = d²T/dy²: axis 1 = y (rows)
T_yy = np.full_like(T_grid, np.nan)
T_yy[:, 1:-1, :] = (T_grid[:, 2:, :] - 2.0 * T_grid[:, 1:-1, :] + T_grid[:, :-2, :]) / (dy ** 2)

# T_t = dT/dt: forward difference between consecutive frames
# Loop needed because dt varies between frames (non-uniform time spacing)
T_t = np.full_like(T_grid, np.nan)
for n in range(nt - 1):
    T_t[n] = (T_grid[n + 1] - T_grid[n]) / dt_vals[n]

# F_t = dF/dt: same forward difference for fuel
F_t = np.full_like(F_grid, np.nan)
for n in range(nt - 1):
    F_t[n] = (F_grid[n + 1] - F_grid[n]) / dt_vals[n]

print("\nFinite differences computed.")

# Reaction term
# exp(-1/T) is the Arrhenius-like reaction term. At T ~ 0 (cold/ambient) it's
# essentially zero and numerically unstable, so mask those points as NaN.
T_safe = np.where(T_grid > 0.01, T_grid, np.nan)
exp_term = np.exp(-1.0 / T_safe)
reaction = F_grid * exp_term   # F * exp(-1/T), the source term in the T equation

# Solve for beta at every grid point
# From the fuel equation:  F_t = -beta * F * exp(-1/T) => beta = -F_t / (F * exp(-1/T))
# This gives a pointwise estimate of beta wherever fuel is burning (F > 0, T > 0).
# Cold/empty points are masked out since the denominator is near zero there.
denom_beta = F_grid * exp_term
valid_beta = (
    np.isfinite(F_t) &
    np.isfinite(denom_beta) &
    (np.abs(denom_beta) > 1e-10) &    # avoid division by ~0
    (F_grid > 0.05) &                 # need meaningful fuel
    (T_grid > 0.05)                   # need meaningful temperature
)

beta_field = np.full_like(T_grid, np.nan)
beta_field[valid_beta] = -F_t[valid_beta] / denom_beta[valid_beta]

# Solve for lambda at every grid point
# From the temperature equation:  T_t = w1*T_xx + w2*T_yy + F*exp(-1/T) - lamb*T => lamb = (-T_t + w1*T_xx + w2*T_yy + F*exp(-1/T)) / T
# Lambda depends on all spatial derivatives plus the reaction term, so it's noisier than beta
valid_lamb = (
    np.isfinite(T_t) &
    np.isfinite(T_xx) &
    np.isfinite(T_yy) &
    np.isfinite(reaction) &
    (np.abs(T_grid) > 0.05)   # need T > 0 to divide
)

lamb_field = np.full_like(T_grid, np.nan)
lamb_field[valid_lamb] = (
    -T_t[valid_lamb] + w1 * T_xx[valid_lamb] + w2 * T_yy[valid_lamb] + reaction[valid_lamb]
) / T_grid[valid_lamb]

# Statistics
# Outliers (|value| > 10) are clipped — these come from near-zero denominators
# or large time gaps between frames. Median is more robust than mean here.
print("\n" + "=" * 60)
print("BETA - pointwise estimates from F equation")
print("=" * 60)
beta_valid = beta_field[np.isfinite(beta_field)]
beta_clipped = beta_valid[(beta_valid > -10) & (beta_valid < 10)]
print(f"  Valid points:    {len(beta_valid):,} / {np.prod(T_grid.shape):,}")
print(f"  Mean:            {np.mean(beta_clipped):.6f}")
print(f"  Median:          {np.median(beta_clipped):.6f}")
print(f"  Std:             {np.std(beta_clipped):.6f}")
print(f"  25th percentile: {np.percentile(beta_clipped, 25):.6f}")
print(f"  75th percentile: {np.percentile(beta_clipped, 75):.6f}")
print(f"  Expected value:  {beta_expected}")

print("\n" + "=" * 60)
print("LAMBDA - pointwise estimates from T equation")
print("=" * 60)
lamb_valid = lamb_field[np.isfinite(lamb_field)]
lamb_clipped = lamb_valid[(lamb_valid > 0) & (lamb_valid < 10)]
print(f"  Valid points:    {len(lamb_valid):,} / {np.prod(T_grid.shape):,}")
print(f"  Mean:            {np.mean(lamb_clipped):.6f}")
print(f"  Median:          {np.median(lamb_clipped):.6f}")
print(f"  Std:             {np.std(lamb_clipped):.6f}")
print(f"  25th percentile: {np.percentile(lamb_clipped, 25):.6f}")
print(f"  75th percentile: {np.percentile(lamb_clipped, 75):.6f}")
print(f"  Expected value:  {lamb_expected}")

# Per-timestep analysis
# Shows how beta and lambda estimates vary across frames.
# Early frames and large time gaps tend to be noisier.
print("\n" + "=" * 60)
print("PER-TIMESTEP BREAKDOWN")
print("=" * 60)
print(f"{'t':>10s}  {'beta_med':>10s}  {'beta_mean':>10s}  {'lamb_med':>10s}  {'lamb_mean':>10s}  {'n_beta':>8s}  {'n_lamb':>8s}")
print("-" * 72)

beta_per_t = []
lamb_per_t = []

for n in range(nt - 1):
    b_slice = beta_field[n]
    b_vals = b_slice[np.isfinite(b_slice) & (b_slice > -10) & (b_slice < 10)]
    l_slice = lamb_field[n]
    l_vals = l_slice[np.isfinite(l_slice) & (l_slice > -10) & (l_slice < 10)]

    b_med = np.median(b_vals) if len(b_vals) > 0 else np.nan
    b_mean = np.mean(b_vals) if len(b_vals) > 0 else np.nan
    l_med = np.median(l_vals) if len(l_vals) > 0 else np.nan
    l_mean = np.mean(l_vals) if len(l_vals) > 0 else np.nan

    beta_per_t.append(b_med)
    lamb_per_t.append(l_med)

    print(f"{t_unique[n]:10.1f}  {b_med:10.6f}  {b_mean:10.6f}  {l_med:10.6f}  {l_mean:10.6f}  {len(b_vals):8d}  {len(l_vals):8d}")

# Visualizations

# 1. Histograms: distribution of all pointwise estimates across the domain
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(beta_clipped, bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
axes[0].axvline(np.median(beta_clipped), color='orange', linewidth=2, linestyle='-', label=f'Median = {np.median(beta_clipped):.4f}')
axes[0].set_xlabel('beta'); axes[0].set_ylabel('Count')
axes[0].set_title('Pointwise beta estimates (from F equation)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].hist(lamb_clipped, bins=100, color='seagreen', alpha=0.7, edgecolor='black', linewidth=0.3)
axes[1].axvline(np.median(lamb_clipped), color='orange', linewidth=2, linestyle='-', label=f'Median = {np.median(lamb_clipped):.4f}')
axes[1].set_xlabel('lambda'); axes[1].set_ylabel('Count')
axes[1].set_title('Pointwise lambda estimates (from T equation)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Fire_PINN_Troy/graphs_calculate/beta_lambda_analytical_hist.png', dpi=200, bbox_inches='tight')
print("\nSaved: beta_lambda_analytical_hist.png")

# 2. Per-timestep median values
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))

axes2[0].plot(t_unique[:-1], beta_per_t, 'bo-', linewidth=2, markersize=5, label='Median beta per frame')
axes2[0].set_xlabel('t (non-dim)'); axes2[0].set_ylabel('beta')
axes2[0].set_title('Beta estimate per timestep'); axes2[0].legend(); axes2[0].grid(True, alpha=0.3)

axes2[1].plot(t_unique[:-1], lamb_per_t, 'go-', linewidth=2, markersize=5, label='Median lambda per frame')
axes2[1].set_xlabel('t (non-dim)'); axes2[1].set_ylabel('lambda')
axes2[1].set_title('Lambda estimate per timestep'); axes2[1].legend(); axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Fire_PINN_Troy/graphs_calculate/beta_lambda_analytical_per_t.png', dpi=200, bbox_inches='tight')
print("Saved: beta_lambda_analytical_per_t.png")

# 3. Spatial maps: where in the domain beta/lambda take different values
#    Helps identify if parameters are spatially uniform or vary with location
mid_t = nt // 2
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 5))

im0 = axes3[0].imshow(beta_field[mid_t], cmap='coolwarm', origin='lower', aspect='auto',
                       vmin=max(-2, np.nanpercentile(beta_field[mid_t], 2)),
                       vmax=min(5, np.nanpercentile(beta_field[mid_t], 98)))
axes3[0].set_title(f'Beta field at t={t_unique[mid_t]:.1f}')
plt.colorbar(im0, ax=axes3[0], fraction=0.046, pad=0.04)

im1 = axes3[1].imshow(lamb_field[mid_t], cmap='coolwarm', origin='lower', aspect='auto',
                       vmin=max(-2, np.nanpercentile(lamb_field[mid_t], 2)),
                       vmax=min(2, np.nanpercentile(lamb_field[mid_t], 98)))
axes3[1].set_title(f'Lambda field at t={t_unique[mid_t]:.1f}')
plt.colorbar(im1, ax=axes3[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('Fire_PINN_Troy/graphs_calculate/beta_lambda_analytical_spatial.png', dpi=200, bbox_inches='tight')
print("Saved: beta_lambda_analytical_spatial.png")

plt.show()

# Final summary
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  w1 = {w1},  w2 = {w2}")
print(f"  beta:   median = {np.median(beta_clipped):.6f},  mean = {np.mean(beta_clipped):.6f}   (used = {beta_expected})")
print(f"  lambda: median = {np.median(lamb_clipped):.6f},  mean = {np.mean(lamb_clipped):.6f}   (used = {lamb_expected})")
print("=" * 60)
print("\nDone!")
