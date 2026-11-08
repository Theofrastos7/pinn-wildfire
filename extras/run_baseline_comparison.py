"""
Non-PINN baselines for the Baseline (2D) inverse problem.
  1. Classical optimisation: FD solver inside Nelder-Mead / L-BFGS-B
  2. Plain neural network: same architecture, data-only loss (no PDE)

Compares parameter recovery errors and field prediction errors against PINN.
Does NOT modify any original files.

Run from project root:  python extras/run_baseline_comparison.py
"""
import sys, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
#  FD solver (copied from Fire_PINN_2D/functions.py — not imported
#  so we don't touch the original folder)
# ═══════════════════════════════════════════════════════════════════

def weighted_laplacian(Z, dx, dy, w1, w2):
    L = np.zeros_like(Z)
    L[1:-1,1:-1] = w1*(Z[2:,1:-1] - 2*Z[1:-1,1:-1] + Z[:-2,1:-1])/dx**2 \
                  + w2*(Z[1:-1,2:] - 2*Z[1:-1,1:-1] + Z[1:-1,:-2])/dy**2
    L[0,:] = L[1,:]; L[-1,:] = L[-2,:]
    L[:,0] = L[:,1]; L[:,-1] = L[:,-2]
    return L

def rk4_diffusion(Tn, k, dt, dx, dy, w1, w2):
    k1 = k * weighted_laplacian(Tn, dx, dy, w1, w2)
    k2 = k * weighted_laplacian(Tn + 0.5*dt*k1, dx, dy, w1, w2)
    k3 = k * weighted_laplacian(Tn + 0.5*dt*k2, dx, dy, w1, w2)
    k4 = k * weighted_laplacian(Tn + dt*k3, dx, dy, w1, w2)
    T_new = Tn + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    T_new[0,:] = T_new[1,:]; T_new[-1,:] = T_new[-2,:]
    T_new[:,0] = T_new[:,1]; T_new[:,-1] = T_new[:,-2]
    return T_new

def reaction_step(Tn, Sn, dt, A, B, C, CS, Ta):
    exp_term = np.exp(-B/(Tn + 1e-8))
    T_new = Tn + dt * A * (Sn * exp_term - C * Tn)
    S_new = Sn / (1 + dt*CS*exp_term)
    return T_new, S_new

def fd_forward(T0, F0, lamb, beta, w1, w2, dx, dy, dt, Nt):
    """Run the non-dimensional FD solver from given initial conditions."""
    Nx, Ny = T0.shape
    T_all = np.zeros((Nt, Nx, Ny))
    F_all = np.zeros((Nt, Nx, Ny))
    T_all[0] = T0.copy()
    F_all[0] = F0.copy()
    for n in range(Nt - 1):
        # k=1, A=1, B=1, Ta=0 in non-dimensional form
        T_diff = rk4_diffusion(T_all[n], 1, dt, dx, dy, w1, w2)
        T_all[n+1], F_all[n+1] = reaction_step(T_diff, F_all[n], dt, 1, 1, lamb, beta, 0)
    return T_all, F_all


# ═══════════════════════════════════════════════════════════════════
#  Load dataset
# ═══════════════════════════════════════════════════════════════════

print("Loading dataset ...")
dataset_path = 'Fire_PINN_2D/data/fire_dataset_columns.csv'
df = pd.read_csv(dataset_path)

# True parameters (stored in first row)
true_lamb = float(df['lamb'].iloc[0])
true_beta = float(df['beta'].iloc[0])
true_w1   = float(df['w1'].iloc[0])
true_w2   = float(df['w2'].iloc[0])

print(f"  True params: lamb={true_lamb:.6f}  beta={true_beta:.6f}  w1={true_w1}  w2={true_w2}")

# Coordinates and fields
x_all = df['x'].values
y_all = df['y'].values
t_all = df['t'].values
T_flat = df['T'].values
F_flat = df['F'].values

# Reconstruct grid dimensions
x_unique = np.sort(np.unique(x_all))
y_unique = np.sort(np.unique(y_all))
t_unique = np.sort(np.unique(t_all))
Nx = len(x_unique)
Ny = len(y_unique)
Nt = len(t_unique)
dx = x_unique[1] - x_unique[0]
dy = y_unique[1] - y_unique[0]
dt = t_unique[1] - t_unique[0]

print(f"  Grid: Nx={Nx}, Ny={Ny}, Nt={Nt}")
print(f"  dx={dx:.4f}, dy={dy:.4f}, dt={dt:.4f}")

# Reshape to 3D arrays (Nt, Nx, Ny) — data stored as meshgrid(t, x, y, indexing='ij').flatten()
T_obs = T_flat.reshape(Nt, Nx, Ny)
F_obs = F_flat.reshape(Nt, Nx, Ny)

# Initial conditions
T0 = T_obs[0].copy()
F0 = F_obs[0].copy()


# ═══════════════════════════════════════════════════════════════════
#  PART 1: Classical FD + Nelder-Mead optimisation
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  PART 1: FD + Nelder-Mead optimisation")
print("="*60)

# Subsample observation times for speed (every 10th timestep)
obs_step = max(1, Nt // 60)
obs_t_indices = np.arange(0, Nt, obs_step)
T_obs_sub = T_obs[obs_t_indices]
F_obs_sub = F_obs[obs_t_indices]

eval_count = [0]

def fd_objective(params_vec):
    lamb, beta, w1, w2 = params_vec
    # Enforce positivity
    if lamb <= 0 or beta <= 0 or w1 <= 0 or w2 <= 0:
        return 1e10
    try:
        T_sim, F_sim = fd_forward(T0, F0, lamb, beta, w1, w2, dx, dy, dt, Nt)
        T_sim_sub = T_sim[obs_t_indices]
        F_sim_sub = F_sim[obs_t_indices]
        err = np.mean((T_sim_sub - T_obs_sub)**2 + (F_sim_sub - F_obs_sub)**2)
    except (FloatingPointError, OverflowError, RuntimeWarning):
        err = 1e10
    eval_count[0] += 1
    if eval_count[0] % 20 == 0:
        print(f"    eval {eval_count[0]:4d}: loss={err:.6e}  "
              f"lamb={lamb:.6f} beta={beta:.6f} w1={w1:.4f} w2={w2:.4f}")
    return err

# Initial guess: 50% off from true values
x0 = [true_lamb * 1.5, true_beta * 1.5, true_w1 * 1.5, true_w2 * 1.5]
print(f"  Initial guess: lamb={x0[0]:.6f}  beta={x0[1]:.6f}  w1={x0[2]:.4f}  w2={x0[3]:.4f}")

t_start = time.time()
result = minimize(
    fd_objective, x0,
    method='Nelder-Mead',
    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-10, 'adaptive': True}
)
t_fd = time.time() - t_start

fd_lamb, fd_beta, fd_w1, fd_w2 = result.x
print(f"\n  FD+Nelder-Mead finished in {t_fd:.1f}s  ({eval_count[0]} evaluations)")
print(f"  Recovered: lamb={fd_lamb:.6f}  beta={fd_beta:.6f}  w1={fd_w1:.4f}  w2={fd_w2:.4f}")

fd_errors = {
    'lambda': abs(fd_lamb - true_lamb) / true_lamb * 100,
    'beta':   abs(fd_beta - true_beta) / true_beta * 100,
    'w1':     abs(fd_w1 - true_w1) / true_w1 * 100,
    'w2':     abs(fd_w2 - true_w2) / true_w2 * 100,
}
for name, err in fd_errors.items():
    print(f"    {name}: error = {err:.2f}%")

fd_r_mean = np.mean([fd_errors['lambda'], fd_errors['beta']])
fd_d_mean = np.mean([fd_errors['w1'], fd_errors['w2']])
print(f"  => mean reaction err = {fd_r_mean:.2f}%,  mean diffusion err = {fd_d_mean:.2f}%")

# Field prediction with recovered params
T_fd_pred, F_fd_pred = fd_forward(T0, F0, fd_lamb, fd_beta, fd_w1, fd_w2, dx, dy, dt, Nt)
fd_T_l2 = np.sqrt(np.mean((T_fd_pred - T_obs)**2)) / np.sqrt(np.mean(T_obs**2))
fd_F_l2 = np.sqrt(np.mean((F_fd_pred - F_obs)**2)) / np.sqrt(np.mean(F_obs**2))
print(f"  Field L2 errors: T={fd_T_l2:.6e},  F={fd_F_l2:.6e}")


# ═══════════════════════════════════════════════════════════════════
#  PART 2: Plain Neural Network (no physics)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  PART 2: Plain NN (data-only, no PDE loss)")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

class PlainNN(nn.Module):
    """Same architecture as the PINN: [3, 20, 20, 20, 20, 2] with Tanh."""
    def __init__(self, lb, ub):
        super().__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device)
        layers = [3, 20, 20, 20, 20, 2]
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.net.add_module(f'tanh_{i}', nn.Tanh())
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Normalise to [-1, 1]
        x_norm = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(x_norm)

# Prepare data — same 10000 random points as PINN uses
np.random.seed(1234)
N_train = 10000
idx = np.random.choice(len(x_all), N_train, replace=False)

X_train = np.column_stack([x_all[idx], y_all[idx], t_all[idx]])
TF_train = np.column_stack([T_flat[idx], F_flat[idx]])

lb = X_train.min(0)
ub = X_train.max(0)

X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
TF_train_t = torch.tensor(TF_train, dtype=torch.float32, device=device)

model_nn = PlainNN(lb, ub).to(device)

# Train with L-BFGS (same as PINN, 15000 iterations)
max_iter_nn = 15000
optimizer = torch.optim.LBFGS(
    model_nn.parameters(), lr=0.6,
    max_iter=20, history_size=50, line_search_fn='strong_wolfe'
)

losses = []
iter_count = [0]

def closure():
    optimizer.zero_grad()
    pred = model_nn(X_train_t)
    loss = torch.mean((pred - TF_train_t)**2)
    loss.backward()
    iter_count[0] += 1
    if iter_count[0] % 500 == 0:
        print(f"    iter {iter_count[0]:5d}: loss = {loss.item():.6e}")
    losses.append(loss.item())
    return loss

print(f"  Training plain NN ({max_iter_nn} L-BFGS steps, {N_train} data points) ...")
t_start = time.time()
for _ in range(max_iter_nn):
    optimizer.step(closure)
    if iter_count[0] >= max_iter_nn:
        break
t_nn = time.time() - t_start
print(f"  Training finished in {t_nn:.1f}s")

# Evaluate on full dataset
X_full = np.column_stack([x_all, y_all, t_all])
X_full_t = torch.tensor(X_full, dtype=torch.float32, device=device)

model_nn.eval()
with torch.no_grad():
    # Predict in batches to avoid OOM
    batch_size = 100000
    preds = []
    for i in range(0, len(X_full_t), batch_size):
        preds.append(model_nn(X_full_t[i:i+batch_size]).cpu().numpy())
    pred_full = np.vstack(preds)

T_nn_pred = pred_full[:, 0]
F_nn_pred = pred_full[:, 1]

nn_T_l2 = np.sqrt(np.mean((T_nn_pred - T_flat)**2)) / np.sqrt(np.mean(T_flat**2))
nn_F_l2 = np.sqrt(np.mean((F_nn_pred - F_flat)**2)) / np.sqrt(np.mean(F_flat**2))
print(f"  Field L2 errors: T={nn_T_l2:.6e},  F={nn_F_l2:.6e}")
print(f"  Parameter recovery: N/A (no physics constraint)")


# ═══════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════

print("\n\n" + "="*60)
print("  COMPARISON SUMMARY (Baseline 2D formulation)")
print("="*60)

# PINN results from seed 1234 (for reference)
pinn_errors = {
    'lambda': 1.65, 'beta': 3.43, 'w1': 5.80, 'w2': 7.20,
}
pinn_r_mean = np.mean([pinn_errors['lambda'], pinn_errors['beta']])
pinn_d_mean = np.mean([pinn_errors['w1'], pinn_errors['w2']])

print(f"\n{'Method':<20} {'lambda%':>10} {'beta%':>10} {'w1%':>10} {'w2%':>10}"
      f"  {'react%':>10} {'diff%':>10}  {'T L2':>12} {'F L2':>12}  {'Time':>8}")
print("-" * 120)
print(f"{'PINN (seed 1234)':<20} {pinn_errors['lambda']:>10.2f} {pinn_errors['beta']:>10.2f}"
      f" {pinn_errors['w1']:>10.2f} {pinn_errors['w2']:>10.2f}"
      f"  {pinn_r_mean:>10.2f} {pinn_d_mean:>10.2f}"
      f"  {'(see paper)':>12} {'(see paper)':>12}  {'~5min':>8}")
print(f"{'FD + Nelder-Mead':<20} {fd_errors['lambda']:>10.2f} {fd_errors['beta']:>10.2f}"
      f" {fd_errors['w1']:>10.2f} {fd_errors['w2']:>10.2f}"
      f"  {fd_r_mean:>10.2f} {fd_d_mean:>10.2f}"
      f"  {fd_T_l2:>12.6e} {fd_F_l2:>12.6e}  {t_fd:>7.1f}s")
print(f"{'Plain NN':<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}"
      f"  {'N/A':>10} {'N/A':>10}"
      f"  {nn_T_l2:>12.6e} {nn_F_l2:>12.6e}  {t_nn:>7.1f}s")

print("\n  Notes:")
print("  - PINN recovers both physical parameters AND field predictions in one training pass.")
print("  - FD+optimizer recovers parameters but requires a separate forward solver.")
print("  - Plain NN fits the data but CANNOT recover physical parameters (no physics constraint).")
