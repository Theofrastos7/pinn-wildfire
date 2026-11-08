"""
Fire PINN - Train/Validation Split by Timestep

Same as pinn_forward.py but trains on the first N timesteps and
validates on the last `val_timesteps` held-out snapshots.
Run twice (w_p=0 vs w_p>0) to measure whether physics helps generalization.

Fire PDEs (2D non-dimensional form with multi-diffusion quadrants):
- Temperature: dT/dt = w1*T_xx + w2*T_yy + F*exp(-1/T) - lamb*T
- Fuel: dF/dt = -beta*F*exp(-1/T)
"""

import torch
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import os
import time


np.random.seed(1234)
torch.manual_seed(1234)

# ── Training configuration ──────────────────────────────────────────────────
max_iter = 20000
w_d = 1.0   # Data loss weight
w_p = 50.0   # Physics loss weight
w_i = 0.0   # Initial condition loss weight
learning_rate = 0.4
data_points = 100000
collocation_points = 2000
dataset = 'Fire_PINN_troy/data/forward_dataset.csv'
val_timesteps = 3  # Number of LAST timesteps held out for validation

TOP_FRACTION = 330 / 830  # asymmetric quadrant split
OUTPUT_DIR = 'Fire_PINN_troy/graphs_split'

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


#---------------------- Deep Neural Network ---------------------

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


#---------------------- Physics-Informed Neural Network ---------------------
class FirePINN():
    def __init__(self, X_data, TF_data, X_collocation, X_ic, TF_ic, layers, lb, ub, fire_params):
        # Domain bounds
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # Training points
        self.x_data = torch.tensor(X_data[:, 0:1], requires_grad=True).float().to(device)
        self.y_data = torch.tensor(X_data[:, 1:2], requires_grad=True).float().to(device)
        self.t_data = torch.tensor(X_data[:, 2:3], requires_grad=True).float().to(device)
        self.T_true = torch.tensor(TF_data[:, 0:1]).float().to(device)
        self.F_true = torch.tensor(TF_data[:, 1:2]).float().to(device)

        # Collocation points
        self.x_col = torch.tensor(X_collocation[:, 0:1], requires_grad=True).float().to(device)
        self.y_col = torch.tensor(X_collocation[:, 1:2], requires_grad=True).float().to(device)
        self.t_col = torch.tensor(X_collocation[:, 2:3], requires_grad=True).float().to(device)

        # Initial condition points
        self.x_ic = torch.tensor(X_ic[:, 0:1], requires_grad=True).float().to(device)
        self.y_ic = torch.tensor(X_ic[:, 1:2], requires_grad=True).float().to(device)
        self.t_ic = torch.tensor(X_ic[:, 2:3], requires_grad=True).float().to(device)
        self.T_ic = torch.tensor(TF_ic[:, 0:1]).float().to(device)
        self.F_ic = torch.tensor(TF_ic[:, 1:2]).float().to(device)

        # Fire parameters
        self.beta = fire_params['beta']
        self.lamb = fire_params['lamb']
        self.Ta = fire_params['Ta']
        self.x_mid = fire_params['x_mid']
        self.y_mid = fire_params['y_mid']

        # Per-quadrant diffusion weights
        # Q1=top-right, Q2=top-left, Q3=bottom-left, Q4=bottom-right
        self.w1_q1 = fire_params['w1_q1']
        self.w2_q1 = fire_params['w2_q1']
        self.w1_q2 = fire_params['w1_q2']
        self.w2_q2 = fire_params['w2_q2']
        self.w1_q3 = fire_params['w1_q3']
        self.w2_q3 = fire_params['w2_q3']
        self.w1_q4 = fire_params['w1_q4']
        self.w2_q4 = fire_params['w2_q4']

        # Network
        self.layers = layers
        self.dnn = DNN(layers).to(device)

        # Optimizer LBFGS
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=learning_rate,
            max_iter=max_iter,
            max_eval=max_iter,
            history_size=50,
            tolerance_grad=0,
            tolerance_change=0,
            line_search_fn="strong_wolfe"
        )

        self.iter = 0

        # Loss history tracking
        self.loss_history = []
        self.loss_data_history = []
        self.loss_pde_history = []
        self.loss_ic_history = []
        self.loss_T_data_history = []
        self.loss_F_data_history = []
        self.loss_T_pde_history = []
        self.loss_F_pde_history = []
        self.loss_T_ic_history = []
        self.loss_F_ic_history = []

    def net_TF(self, x, y, t):
        x_norm = 2.0 * (x - self.lb[0]) / (self.ub[0] - self.lb[0]) - 1.0
        y_norm = 2.0 * (y - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1.0
        t_norm = 2.0 * (t - self.lb[2]) / (self.ub[2] - self.lb[2]) - 1.0

        output = self.dnn(torch.cat([x_norm, y_norm, t_norm], dim=1))
        T = output[:, 0:1]
        F = output[:, 1:2]
        return T, F

    def net_pde(self, x, y, t):
        T, F = self.net_TF(x, y, t)

        T_t = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T),
                                  retain_graph=True, create_graph=True)[0]
        T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T),
                                  retain_graph=True, create_graph=True)[0]
        T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T),
                                  retain_graph=True, create_graph=True)[0]
        F_t = torch.autograd.grad(F, t, grad_outputs=torch.ones_like(F),
                                  retain_graph=True, create_graph=True)[0]

        T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x),
                                   retain_graph=True, create_graph=True)[0]
        T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y),
                                   retain_graph=True, create_graph=True)[0]

        T_safe = torch.clamp(T, min=0.1)
        reaction_term = F * torch.exp(-1.0 / T_safe)

        is_x_high = x >= self.x_mid
        is_y_high = y >= self.y_mid

        # Q1=top-right, Q2=top-left, Q3=bottom-left, Q4=bottom-right
        w1_spatial = torch.where(
            is_x_high,
            torch.where(is_y_high, self.w1_q1, self.w1_q4),
            torch.where(is_y_high, self.w1_q2, self.w1_q3)
        )
        w2_spatial = torch.where(
            is_x_high,
            torch.where(is_y_high, self.w2_q1, self.w2_q4),
            torch.where(is_y_high, self.w2_q2, self.w2_q3)
        )

        cooling_term = self.lamb * T
        fuel_consumption = self.beta * reaction_term

        pde_T = T_t - w1_spatial * T_xx - w2_spatial * T_yy - reaction_term + cooling_term
        pde_F = F_t + fuel_consumption

        return pde_T, pde_F

    def loss_func(self):
        self.optimizer.zero_grad()

        T_pred, F_pred = self.net_TF(self.x_data, self.y_data, self.t_data)
        loss_T_data = torch.mean((self.T_true - T_pred) ** 2)
        loss_F_data = torch.mean((self.F_true - F_pred) ** 2)
        loss_data = loss_T_data + loss_F_data

        if w_p > 0:
            pde_T, pde_F = self.net_pde(self.x_col, self.y_col, self.t_col)
            loss_T_pde = torch.mean(pde_T ** 2)
            loss_F_pde = torch.mean(pde_F ** 2)
            loss_pde = loss_T_pde + loss_F_pde
        else:
            loss_T_pde = torch.tensor(0.0).to(device)
            loss_F_pde = torch.tensor(0.0).to(device)
            loss_pde = torch.tensor(0.0).to(device)

        if w_i > 0:
            T_ic_pred, F_ic_pred = self.net_TF(self.x_ic, self.y_ic, self.t_ic)
            loss_T_ic = torch.mean((self.T_ic - T_ic_pred) ** 2)
            loss_F_ic = torch.mean((self.F_ic - F_ic_pred) ** 2)
            loss_ic = loss_T_ic + loss_F_ic
        else:
            loss_T_ic = torch.tensor(0.0).to(device)
            loss_F_ic = torch.tensor(0.0).to(device)
            loss_ic = torch.tensor(0.0).to(device)

        loss = w_d * loss_data + w_p * loss_pde + w_i * loss_ic
        loss.backward()
        self.iter += 1

        if self.iter % 20 == 0:
            self.loss_history.append(loss.item())
            self.loss_data_history.append(loss_data.item())
            self.loss_pde_history.append(loss_pde.item())
            self.loss_ic_history.append(loss_ic.item())
            self.loss_T_data_history.append(loss_T_data.item())
            self.loss_F_data_history.append(loss_F_data.item())
            self.loss_T_pde_history.append(loss_T_pde.item())
            self.loss_F_pde_history.append(loss_F_pde.item())
            self.loss_T_ic_history.append(loss_T_ic.item())
            self.loss_F_ic_history.append(loss_F_ic.item())

        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Data: %.5e (T:%.5e, F:%.5e), PDE: %.5e (T:%.5e, F:%.5e), IC: %.5e (T:%.5e, F:%.5e)'
                % (self.iter, loss.item(), loss_data.item(), loss_T_data.item(), loss_F_data.item(),
                   loss_pde.item(), loss_T_pde.item(), loss_F_pde.item(),
                   loss_ic.item(), loss_T_ic.item(), loss_F_ic.item())
            )

        return loss

    def train(self, max_iterations=max_iter):
        self.dnn.train()
        while self.iter < max_iterations:
            def closure():
                return self.loss_func()
            self.optimizer.step(closure)
            if self.iter < max_iterations:
                continue
            else:
                break

    def predict(self, X, batch_size=50000, compute_pde=False):
        self.dnn.eval()

        N = X.shape[0]
        T_all = []
        F_all = []
        pde_T_all = []
        pde_F_all = []

        num_batches = (N + batch_size - 1) // batch_size
        print(f"Processing {N} points in {num_batches} batches...")

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N)
            X_batch = X[start_idx:end_idx, :]

            x = torch.tensor(X_batch[:, 0:1], requires_grad=True).float().to(device)
            y = torch.tensor(X_batch[:, 1:2], requires_grad=True).float().to(device)
            t = torch.tensor(X_batch[:, 2:3], requires_grad=True).float().to(device)

            T_batch, F_batch = self.net_TF(x, y, t)
            T_all.append(T_batch.detach().cpu().numpy())
            F_all.append(F_batch.detach().cpu().numpy())

            if compute_pde:
                pde_T_batch, pde_F_batch = self.net_pde(x, y, t)
                pde_T_all.append(pde_T_batch.detach().cpu().numpy())
                pde_F_all.append(pde_F_batch.detach().cpu().numpy())

            if (i + 1) % 100 == 0 or (i + 1) == num_batches:
                print(f"  Processed batch {i+1}/{num_batches}")

        T = np.vstack(T_all)
        F = np.vstack(F_all)

        if compute_pde and pde_T_all:
            pde_T = np.vstack(pde_T_all)
            pde_F = np.vstack(pde_F_all)
        else:
            pde_T = None
            pde_F = None

        return T, F, pde_T, pde_F


def _fmt(v, sig=3):
    """Format a float as  m.mm x 10^n  (3 sig. figs by default)."""
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10**e
    sup = str(e).translate(str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))
    return f"{m:.{sig-1}f}×10{sup}"

#---------------------- Load Fire Data -----------------------------
print("Loading fire dataset...")

data_df = pd.read_csv(dataset)

x_all = data_df['x'].values
y_all = data_df['y'].values
t_all = data_df['t'].values
T_all = data_df['T'].values
F_all = data_df['F'].values

print(f"Total data points: {len(x_all)}")
print(f"x range: [{x_all.min():.2f}, {x_all.max():.2f}]")
print(f"y range: [{y_all.min():.2f}, {y_all.max():.2f}]")
print(f"t range: [{t_all.min():.2f}, {t_all.max():.2f}]")
print(f"T range: [{T_all.min():.4f}, {T_all.max():.4f}]")
print(f"F range: [{F_all.min():.4f}, {F_all.max():.4f}]")

X_star = np.column_stack([x_all, y_all, t_all])
TF_star = np.column_stack([T_all, F_all])

print("Size of dataset: ", X_star.shape)

# Domain bounds — use ALL data so normalization covers val timesteps too
lb = X_star.min(0)
ub = X_star.max(0)

print(f"\nDomain bounds:")
print(f"  lb = {lb}")
print(f"  ub = {ub}")

#---------------------- Configuration -----------------------------
beta = data_df['beta'].iloc[0]
lamb = data_df['lamb'].iloc[0]
Ta   = data_df['Ta'].iloc[0]
w1_q1 = data_df['w1_q1'].iloc[0]
w2_q1 = data_df['w2_q1'].iloc[0]
w1_q2 = data_df['w1_q2'].iloc[0]
w2_q2 = data_df['w2_q2'].iloc[0]
w1_q3 = data_df['w1_q3'].iloc[0]
w2_q3 = data_df['w2_q3'].iloc[0]
w1_q4 = data_df['w1_q4'].iloc[0]
w2_q4 = data_df['w2_q4'].iloc[0]

x_mid = (ub[0] + lb[0]) / 2
y_mid = lb[1] + (ub[1] - lb[1]) * (1 - TOP_FRACTION)

fire_params = {
    'beta': beta, 'lamb': lamb, 'Ta': Ta,
    'w1_q1': w1_q1, 'w2_q1': w2_q1,
    'w1_q2': w1_q2, 'w2_q2': w2_q2,
    'w1_q3': w1_q3, 'w2_q3': w2_q3,
    'w1_q4': w1_q4, 'w2_q4': w2_q4,
    'x_mid': x_mid,
    'y_mid': y_mid,
}

print(f"\nFire parameters:")
print(f"  beta={beta:.6f}, lamb={lamb:.6f}, Ta={Ta}")
print(f"  Q1 (top-right):    w1={w1_q1}, w2={w2_q1}")
print(f"  Q2 (top-left):     w1={w1_q2}, w2={w2_q2}")
print(f"  Q3 (bottom-left):  w1={w1_q3}, w2={w2_q3}")
print(f"  Q4 (bottom-right): w1={w1_q4}, w2={w2_q4}")
print(f"  x_mid={x_mid:.4f}, y_mid={y_mid:.4f} (TOP_FRACTION={TOP_FRACTION:.4f})")

N_data = data_points
N_collocation = collocation_points
layers = [3, 32, 32, 32, 32, 2]

print(f"\nTraining configuration:")
print(f"  N_data = {N_data}")
print(f"  N_collocation = {N_collocation}")
print(f"  Network layers = {layers}")

# ── Train / validation timestep split (interleaved) ──────────────
t_unique = np.unique(t_all)
n_total = len(t_unique)
# Spread val timesteps evenly: e.g. 15 total, 3 val → indices 4, 9, 14
step = n_total // val_timesteps
val_indices = [step * (i + 1) - 1 for i in range(val_timesteps)]
val_t_set   = set(t_unique[i] for i in val_indices)
train_t_set = set(t_unique) - val_t_set
n_train = len(train_t_set)
train_mask = np.array([t in train_t_set for t in t_all])
val_mask = ~train_mask

print(f"\n  Timestep split: {n_train} TRAIN, {val_timesteps} VAL (last {val_timesteps} held out)")
print(f"  Train times: {sorted(train_t_set)}")
print(f"  Val   times: {sorted(val_t_set)}")

# Sample data points only from TRAIN timesteps
train_indices = np.where(train_mask)[0]
idx_data = np.random.choice(train_indices, min(N_data, len(train_indices)), replace=False)
X_data = X_star[idx_data, :]
TF_data = TF_star[idx_data, :]

# Sample collocation from active fire regions in TRAIN timesteps
T_COLL_THRESHOLD = 0.15
fire_train_mask = (T_all > T_COLL_THRESHOLD) & train_mask
fire_train_indices = np.where(fire_train_mask)[0]
print(f"  Active fire points in train (T>{T_COLL_THRESHOLD}): {len(fire_train_indices)}")
N_col = min(N_collocation, len(fire_train_indices))
idx_col = np.random.choice(fire_train_indices, N_col, replace=False)
X_collocation = X_star[idx_col, :]

# Initial condition points (first timestep)
t_min = t_all.min()
ic_mask = (t_all == t_min)
X_ic = X_star[ic_mask, :]
TF_ic = TF_star[ic_mask, :]

print(f"\nData shapes:")
print(f"  X_data: {X_data.shape}")
print(f"  TF_data: {TF_data.shape}")
print(f"  X_collocation: {X_collocation.shape}")
print(f"  X_ic (initial conditions): {X_ic.shape}")
print(f"  TF_ic: {TF_ic.shape}")

#------------------------ Training -----------------------------
print("\n" + "="*60)
print("Starting training...")
print("="*60)

model = FirePINN(X_data, TF_data, X_collocation, X_ic, TF_ic, layers, lb, ub, fire_params)

start_time = time.time()
model.train()
elapsed = time.time() - start_time

print('\n' + "="*60)
print(f'Training complete! Time: {elapsed:.2f}s')
print(f'Parameters: beta={beta}, lamb={lamb}, Ta={Ta}')
print(f'  w_d={w_d}, w_p={w_p}, w_i={w_i}')
print("="*60)

#------------------------ Evaluation -----------------------------
print("\nEvaluating on ALL timesteps (train + val)...")

T_pred, F_pred, pde_T, pde_F = model.predict(X_star)

# ── Overall errors ─────────────────────────────────────────────────
error_T = np.linalg.norm(T_all.reshape(-1, 1) - T_pred, 2) / np.linalg.norm(T_all.reshape(-1, 1), 2)
error_F = np.linalg.norm(F_all.reshape(-1, 1) - F_pred, 2) / np.linalg.norm(F_all.reshape(-1, 1), 2)
error_T_abs = np.abs(T_all.reshape(-1, 1) - T_pred)
error_F_abs = np.abs(F_all.reshape(-1, 1) - F_pred)

print(f'\n{"="*60}')
print(f'OVERALL  Relative L2 — T: {_fmt(error_T)}   F: {_fmt(error_F)}')
print(f'         Mean abs    — T: {_fmt(np.mean(error_T_abs))}   F: {_fmt(np.mean(error_F_abs))}')
print(f'         Max  abs    — T: {_fmt(np.max(error_T_abs))}   F: {_fmt(np.max(error_F_abs))}')

# ── Train errors ──────────────────────────────────────────────────
ti = np.where(train_mask)[0]
vi = np.where(val_mask)[0]

err_T_train = np.linalg.norm(T_all[ti].reshape(-1,1) - T_pred[ti], 2) / np.linalg.norm(T_all[ti].reshape(-1,1), 2)
err_F_train = np.linalg.norm(F_all[ti].reshape(-1,1) - F_pred[ti], 2) / np.linalg.norm(F_all[ti].reshape(-1,1), 2)

print(f'\nTRAIN    Relative L2 — T: {_fmt(err_T_train)}   F: {_fmt(err_F_train)}')
print(f'         Mean abs    — T: {_fmt(np.mean(np.abs(T_all[ti].reshape(-1,1) - T_pred[ti])))}   F: {_fmt(np.mean(np.abs(F_all[ti].reshape(-1,1) - F_pred[ti])))}')

# ── Val errors ────────────────────────────────────────────────────
err_T_val = np.linalg.norm(T_all[vi].reshape(-1,1) - T_pred[vi], 2) / np.linalg.norm(T_all[vi].reshape(-1,1), 2)
err_F_val = np.linalg.norm(F_all[vi].reshape(-1,1) - F_pred[vi], 2) / np.linalg.norm(F_all[vi].reshape(-1,1), 2)

print(f'\nVAL      Relative L2 — T: {_fmt(err_T_val)}   F: {_fmt(err_F_val)}')
print(f'         Mean abs    — T: {_fmt(np.mean(np.abs(T_all[vi].reshape(-1,1) - T_pred[vi])))}   F: {_fmt(np.mean(np.abs(F_all[vi].reshape(-1,1) - F_pred[vi])))}')

print(f'\n  Val/Train ratio    — T: {err_T_val/err_T_train:.2f}x   F: {err_F_val/err_F_train:.2f}x')

# ── Per-timestep L2 errors ────────────────────────────────────────
print(f'\n{"─"*60}')
print(f'Per-timestep Relative L2:')
print(f'  {"t":>8s}  {"T error":>10s}  {"F error":>10s}  {"split":>5s}')
for t_val in t_unique:
    mask_t = (t_all == t_val)
    idx_t = np.where(mask_t)[0]
    eT = np.linalg.norm(T_all[idx_t].reshape(-1,1) - T_pred[idx_t], 2) / (np.linalg.norm(T_all[idx_t].reshape(-1,1), 2) + 1e-10)
    eF = np.linalg.norm(F_all[idx_t].reshape(-1,1) - F_pred[idx_t], 2) / (np.linalg.norm(F_all[idx_t].reshape(-1,1), 2) + 1e-10)
    tag = "TRAIN" if t_val in train_t_set else "VAL"
    print(f'  {t_val:8.4f}  {_fmt(eT):>10s}  {_fmt(eF):>10s}  {tag:>5s}')
print(f'{"="*60}')

#------------------------ Visualization -----------------------------
print("\nCreating visualizations...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

x_unique = np.unique(x_all)
y_unique = np.unique(y_all)
nx = len(x_unique)
ny = len(y_unique)
nt = len(t_unique)

print(f"Grid: {nt} time steps x {nx} x {ny} space points")

T_true_grid = T_all.reshape(nt, ny, nx)
F_true_grid = F_all.reshape(nt, ny, nx)
T_pred_grid = T_pred.reshape(nt, ny, nx)
F_pred_grid = F_pred.reshape(nt, ny, nx)

# Show snapshots: include all val indices + first and last
time_indices = sorted(set([0, nt-1] + val_indices))
time_labels = [f't={t_unique[i]:.2f} ({"TRAIN" if t_unique[i] in train_t_set else "VAL"})' for i in time_indices]

fig, axes = plt.subplots(len(time_indices), 4, figsize=(20, 4*len(time_indices)))

for i, (tidx, tlabel) in enumerate(zip(time_indices, time_labels)):
    im1 = axes[i, 0].imshow(T_true_grid[tidx], interpolation='nearest', cmap='hot',
                            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                            origin='lower', aspect='auto',
                            vmin=T_all.min(), vmax=T_all.max())
    axes[i, 0].set_title(f'Temperature True ({tlabel})', fontsize=12)
    axes[i, 0].set_xlabel('X'); axes[i, 0].set_ylabel('Y')
    divider = make_axes_locatable(axes[i, 0])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im1, cax=cax)

    im2 = axes[i, 1].imshow(T_pred_grid[tidx], interpolation='nearest', cmap='hot',
                            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                            origin='lower', aspect='auto',
                            vmin=T_all.min(), vmax=T_all.max())
    axes[i, 1].set_title(f'Temperature PINN ({tlabel})', fontsize=12)
    axes[i, 1].set_xlabel('X'); axes[i, 1].set_ylabel('Y')
    divider = make_axes_locatable(axes[i, 1])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im2, cax=cax)

    im3 = axes[i, 2].imshow(F_true_grid[tidx], interpolation='nearest', cmap='viridis',
                            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                            origin='lower', aspect='auto',
                            vmin=F_all.min(), vmax=F_all.max())
    axes[i, 2].set_title(f'Fuel True ({tlabel})', fontsize=12)
    axes[i, 2].set_xlabel('X'); axes[i, 2].set_ylabel('Y')
    divider = make_axes_locatable(axes[i, 2])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im3, cax=cax)

    im4 = axes[i, 3].imshow(F_pred_grid[tidx], interpolation='nearest', cmap='viridis',
                            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                            origin='lower', aspect='auto',
                            vmin=F_all.min(), vmax=F_all.max())
    axes[i, 3].set_title(f'Fuel PINN ({tlabel})', fontsize=12)
    axes[i, 3].set_xlabel('X'); axes[i, 3].set_ylabel('Y')
    divider = make_axes_locatable(axes[i, 3])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im4, cax=cax)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/pinn_split_snapshots.png', dpi=150, bbox_inches='tight')
print(f"Snapshots saved to '{OUTPUT_DIR}/pinn_split_snapshots.png'")

# Error visualization
fig2, axes2 = plt.subplots(len(time_indices), 2, figsize=(14, 4*len(time_indices)))

for i, (tidx, tlabel) in enumerate(zip(time_indices, time_labels)):
    error_T_grid = T_true_grid[tidx] - T_pred_grid[tidx]
    error_F_grid = F_true_grid[tidx] - F_pred_grid[tidx]

    mse_T = np.mean(error_T_grid**2)
    mse_F = np.mean(error_F_grid**2)
    rel_T = np.linalg.norm(error_T_grid) / (np.linalg.norm(T_true_grid[tidx]) + 1e-10)
    rel_F = np.linalg.norm(error_F_grid) / (np.linalg.norm(F_true_grid[tidx]) + 1e-10)

    im5 = axes2[i, 0].imshow(np.abs(error_T_grid), interpolation='nearest', cmap='Reds',
                             extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                             origin='lower', aspect='auto')
    axes2[i, 0].set_title(f'Temperature Error ({tlabel})', fontsize=12)
    axes2[i, 0].set_xlabel('X'); axes2[i, 0].set_ylabel('Y')
    axes2[i, 0].legend(handles=[
        Patch(fc='none', ec='none', label=f'Min:  {_fmt(np.abs(error_T_grid).min())}'),
        Patch(fc='none', ec='none', label=f'Max:  {_fmt(np.abs(error_T_grid).max())}'),
        Patch(fc='none', ec='none', label=f'Rel:  {rel_T*100:.2f}%'),
        Patch(fc='none', ec='none', label=f'MSE: {_fmt(mse_T)}'),
    ], loc='upper right', fontsize=8, framealpha=0.8)
    divider = make_axes_locatable(axes2[i, 0])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im5, cax=cax)

    im6 = axes2[i, 1].imshow(np.abs(error_F_grid), interpolation='nearest', cmap='Reds',
                             extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                             origin='lower', aspect='auto')
    axes2[i, 1].set_title(f'Fuel Error ({tlabel})', fontsize=12)
    axes2[i, 1].set_xlabel('X'); axes2[i, 1].set_ylabel('Y')
    axes2[i, 1].legend(handles=[
        Patch(fc='none', ec='none', label=f'Min:  {_fmt(np.abs(error_F_grid).min())}'),
        Patch(fc='none', ec='none', label=f'Max:  {_fmt(np.abs(error_F_grid).max())}'),
        Patch(fc='none', ec='none', label=f'Rel:  {rel_F*100:.2f}%'),
        Patch(fc='none', ec='none', label=f'MSE: {_fmt(mse_F)}'),
    ], loc='upper right', fontsize=8, framealpha=0.8)
    divider = make_axes_locatable(axes2[i, 1])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im6, cax=cax)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/pinn_split_error_snapshots.png', dpi=150, bbox_inches='tight')
print(f"Error snapshots saved to '{OUTPUT_DIR}/pinn_split_error_snapshots.png'")

#------------------------ Loss History Plots -----------------------------
print("\nCreating loss history plots...")

iterations = np.arange(20, len(model.loss_history) * 20 + 1, 20)

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

axes4[0, 0].plot(iterations, model.loss_history, 'b-', linewidth=2)
axes4[0, 0].set_xlabel('Iteration', fontsize=12)
axes4[0, 0].set_ylabel('Total Loss', fontsize=12)
axes4[0, 0].set_title('Total Loss History', fontsize=14)
axes4[0, 0].set_yscale('log')
axes4[0, 0].grid(True, alpha=0.3)

axes4[0, 1].plot(iterations, model.loss_data_history, 'r-', linewidth=2, label='Data Loss')
axes4[0, 1].plot(iterations, model.loss_pde_history, 'g-', linewidth=2, label='PDE Loss')
axes4[0, 1].set_xlabel('Iteration', fontsize=12)
axes4[0, 1].set_ylabel('Loss', fontsize=12)
axes4[0, 1].set_title('Data Loss vs PDE Loss', fontsize=14)
axes4[0, 1].set_yscale('log')
axes4[0, 1].legend(fontsize=11)
axes4[0, 1].grid(True, alpha=0.3)

axes4[1, 0].plot(iterations, model.loss_T_data_history, 'r-', linewidth=2, label='T Data Loss')
axes4[1, 0].plot(iterations, model.loss_T_pde_history, 'g-', linewidth=2, label='T PDE Loss')
axes4[1, 0].set_xlabel('Iteration', fontsize=12)
axes4[1, 0].set_ylabel('Loss', fontsize=12)
axes4[1, 0].set_title('Temperature Losses', fontsize=14)
axes4[1, 0].set_yscale('log')
axes4[1, 0].legend(fontsize=11)
axes4[1, 0].grid(True, alpha=0.3)

axes4[1, 1].plot(iterations, model.loss_F_data_history, 'r-', linewidth=2, label='F Data Loss')
axes4[1, 1].plot(iterations, model.loss_F_pde_history, 'g-', linewidth=2, label='F PDE Loss')
axes4[1, 1].set_xlabel('Iteration', fontsize=12)
axes4[1, 1].set_ylabel('Loss', fontsize=12)
axes4[1, 1].set_title('Fuel Losses', fontsize=14)
axes4[1, 1].set_yscale('log')
axes4[1, 1].legend(fontsize=11)
axes4[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/pinn_split_loss_history.png', dpi=150, bbox_inches='tight')
print(f"Loss history saved to '{OUTPUT_DIR}/pinn_split_loss_history.png'")

# Save model
torch.save(model.dnn.state_dict(), f'{OUTPUT_DIR}/pinn_split_model.pth')
print(f"Model saved to '{OUTPUT_DIR}/pinn_split_model.pth'")

plt.show()

print("\nDone!")
