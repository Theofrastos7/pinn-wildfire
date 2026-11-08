"""
Fire NN/PINN for Troy fire

Train on N_TRAIN full images, validate on rest
Excludes final two images (5_49, 6_30).
Set w_p > 0 to enable PDE physics loss.

Input: (x, y, t)
Output: (T, F)
"""

import torch
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import time

SEED = 1232
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Training configuration ──────────────────────────────────────────────────
max_iter = 20000
w_d = 1.0   # Data loss weight
w_p = 0.0   # Physics loss weight 
w_i = 0.0   # Initial condition loss weight
learning_rate = 0.2
sample_fraction = 0.15  # fraction of train pixels to use (0.0-1.0), e.g. 0.5 = 50%
collocation_points = 15000 # Collocation points for physics
dataset = 'Fire_PINN_troy/data/forward_dataset.csv'

N_TRAIN = 15   # train on  15 full images
N_IMAGES = 18  # use 3 more for validation

TOP_FRACTION = 330 / 830  # asymmetric quadrant split

# CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")


#---------------------- Deep Neural Network ---------------------

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        # Input shape: N x 3 (x, y, t)
        # Output shape: N x 2 (T, F)
        self.depth = len(layers) - 1

        # Activation function: Tanh
        self.activation = torch.nn.Tanh

        # Build layer list
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)

        # Deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


#---------------------- Physics-Informed Neural Network ---------------------
class FireNN():
    def __init__(self, X_data, TF_data, X_collocation, X_ic, TF_ic, layers, lb, ub, fire_params):
        """
        X_data: training data points (N_data x 3) [x, y, t]
        TF_data: known T and F values at X_data (N_data x 2) [T, F]
        X_collocation: collocation points for PDE residual (N_collocation x 3)
        X_ic: initial condition points at t=0 (N_ic x 3) [x, y, 0]
        TF_ic: initial T and F values at t=0 (N_ic x 2) [T, F]
        layers: network architecture
        lb, ub: domain bounds
        fire_params: dict with beta, lamb, Ta, w1_q1..w2_q4, x_mid, y_mid
        """
        # Domain bounds
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # Training points, where T and F are known
        self.x_data = torch.tensor(X_data[:, 0:1], requires_grad=True).float().to(device)
        self.y_data = torch.tensor(X_data[:, 1:2], requires_grad=True).float().to(device)
        self.t_data = torch.tensor(X_data[:, 2:3], requires_grad=True).float().to(device)
        self.T_true = torch.tensor(TF_data[:, 0:1]).float().to(device)
        self.F_true = torch.tensor(TF_data[:, 1:2]).float().to(device)

        # Collocation points, for PDE residual
        self.x_col = torch.tensor(X_collocation[:, 0:1], requires_grad=True).float().to(device)
        self.y_col = torch.tensor(X_collocation[:, 1:2], requires_grad=True).float().to(device)
        self.t_col = torch.tensor(X_collocation[:, 2:3], requires_grad=True).float().to(device)

        # Initial condition points
        self.x_ic = torch.tensor(X_ic[:, 0:1], requires_grad=True).float().to(device)
        self.y_ic = torch.tensor(X_ic[:, 1:2], requires_grad=True).float().to(device)
        self.t_ic = torch.tensor(X_ic[:, 2:3], requires_grad=True).float().to(device)
        self.T_ic = torch.tensor(TF_ic[:, 0:1]).float().to(device)
        self.F_ic = torch.tensor(TF_ic[:, 1:2]).float().to(device)

        # Fire parameters from dataset
        self.beta = fire_params['beta']
        self.lamb = fire_params['lamb']
        self.x_mid = fire_params['x_mid']
        self.y_mid = fire_params['y_mid']

        # Per-quadrant diffusion weights
        # Q1=top-left, Q2=top-right, Q3=bottom-left, Q4=bottom-right
        self.w1_q1 = fire_params['w1_q1']; self.w2_q1 = fire_params['w2_q1']
        self.w1_q2 = fire_params['w1_q2']; self.w2_q2 = fire_params['w2_q2']
        self.w1_q3 = fire_params['w1_q3']; self.w2_q3 = fire_params['w2_q3']
        self.w1_q4 = fire_params['w1_q4']; self.w2_q4 = fire_params['w2_q4']

        # Network
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

    def net_TF(self, x, y, t):
        """
        Network prediction: concatenate [x, y, t]
        normalize to [-1, 1] for better training
        return [T, F]
        """
        # Normalize inputs to [-1, 1]
        x_norm = 2.0 * (x - self.lb[0]) / (self.ub[0] - self.lb[0]) - 1.0
        y_norm = 2.0 * (y - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1.0
        t_norm = 2.0 * (t - self.lb[2]) / (self.ub[2] - self.lb[2]) - 1.0

        output = self.dnn(torch.cat([x_norm, y_norm, t_norm], dim=1))

        # output shape: N x 2, T and F
        T = output[:, 0:1]
        F = output[:, 1:2]
        return T, F

    def net_pde(self, x, y, t):
        """
        Compute PDE residuals at collocation points using autograd
        """
        T, F = self.net_TF(x, y, t)

        # First derivatives

        # T_t = dT/dt
        T_t = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]

        # T_x = dT/dx
        T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]

        # T_y = dT/dy
        T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]

        # F_t = dF/dt
        F_t = torch.autograd.grad(F, t, grad_outputs=torch.ones_like(F), retain_graph=True, create_graph=True)[0]

        # Second derivatives

        # T_xx = d²T/dx²
        T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), retain_graph=True, create_graph=True)[0]

        # T_yy = d²T/dy²
        T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), retain_graph=True, create_graph=True)[0]

        # Fire physics
        T_safe = torch.clamp(T, min=0.1)  # Avoid small T for exp(-1/T)
        reaction = F * torch.exp(-1.0 / T_safe)  # reaction term = F*exp(-1/T)

        # Determine position for each collocation point in order to use correct w1 and w2
        is_x_high = x >= self.x_mid   # x >= mid
        is_y_high = y >= self.y_mid   # y >= mid

        # Q1=top-left, Q2=top-right, Q3=bottom-left, Q4=bottom-right
        w1 = torch.where(is_y_high, torch.where(is_x_high, self.w1_q2, self.w1_q1),
                                     torch.where(is_x_high, self.w1_q4, self.w1_q3))
        w2 = torch.where(is_y_high, torch.where(is_x_high, self.w2_q2, self.w2_q1),
                                     torch.where(is_x_high, self.w2_q4, self.w2_q3))

        cooling_term = self.lamb * T               # cooling term = lamb*T
        fuel_consumption = self.beta * reaction     # fuel consumption = beta*F*exp(-1/T)

        # PDE residuals
        # Temperature: dT/dt = w1*T_xx + w2*T_yy + F*exp(-1/T) - lamb*T
        # Fuel: dF/dt = -beta*F*exp(-1/T)
        pde_T = T_t - w1 * T_xx - w2 * T_yy - reaction + cooling_term
        pde_F = F_t + fuel_consumption
        return pde_T, pde_F

    def loss_func(self):
        """
        Total loss = data loss + PDE residual loss + initial condition loss
        """
        self.optimizer.zero_grad()

        # Data loss is MSE between prediction and true values
        T_pred, F_pred = self.net_TF(self.x_data, self.y_data, self.t_data)
        loss_T_data = torch.mean((self.T_true - T_pred) ** 2)
        loss_F_data = torch.mean((self.F_true - F_pred) ** 2)
        loss_data = loss_T_data + loss_F_data

        # PDE residual loss is MSE of PDE residuals at collocation points
        # only calculate if the weight of pdes is greater than 0
        if w_p > 0:
            pde_T, pde_F = self.net_pde(self.x_col, self.y_col, self.t_col)
            loss_T_pde = torch.mean(pde_T ** 2)
            loss_F_pde = torch.mean(pde_F ** 2)
            loss_pde = loss_T_pde + loss_F_pde
        else:
            loss_T_pde = torch.tensor(0.0).to(device)
            loss_F_pde = torch.tensor(0.0).to(device)
            loss_pde = torch.tensor(0.0).to(device)

        # Initial condition loss (at t=0)
        # only calculate if the weight of initial conditions is greater than 0
        if w_i > 0:
            T_ic_pred, F_ic_pred = self.net_TF(self.x_ic, self.y_ic, self.t_ic)
            loss_ic = torch.mean((self.T_ic - T_ic_pred)**2) + torch.mean((self.F_ic - F_ic_pred)**2)
        else:
            loss_ic = torch.tensor(0.0).to(device)

        # Total loss = data loss + PDE loss + initial condition loss
        loss = w_d * loss_data + w_p * loss_pde + w_i * loss_ic
        loss.backward()
        self.iter += 1

        # Track losses every 20 iterations
        if self.iter % 20 == 0:
            self.loss_history.append(loss.item())
            self.loss_data_history.append(loss_data.item())
            self.loss_pde_history.append(loss_pde.item())
            self.loss_ic_history.append(loss_ic.item())
            self.loss_T_data_history.append(loss_T_data.item())
            self.loss_F_data_history.append(loss_F_data.item())
            self.loss_T_pde_history.append(loss_T_pde.item())
            self.loss_F_pde_history.append(loss_F_pde.item())

        if self.iter % 100 == 0:
            print('Iter %d, Loss: %.5e, Data: %.5e (T:%.5e, F:%.5e), PDE: %.5e (T:%.5e, F:%.5e)'
                  % (self.iter, loss.item(), loss_data.item(), loss_T_data.item(), loss_F_data.item(),
                     loss_pde.item(), loss_T_pde.item(), loss_F_pde.item()))
        return loss

    def train(self, max_iterations=max_iter):
        """
        Train the network for max_iterations iterations
        """
        self.dnn.train()

        # Ensure LBFGS runs for max_iterations
        while self.iter < max_iterations:
            def closure():
                return self.loss_func()

            self.optimizer.step(closure)

            # If LBFGS stops early, reinitialize it
            if self.iter >= max_iterations:
                break

    def predict(self, X, batch_size=50000):
        """
        Predict T and F at given points X
        Using batches to avoid memory overflowing issues
        """
        self.dnn.eval()

        N = X.shape[0]
        T_all, F_all = [], []

        # Process in batches
        num_batches = (N + batch_size - 1) // batch_size
        for i in range(num_batches):
            s, e = i * batch_size, min((i + 1) * batch_size, N)
            x = torch.tensor(X[s:e, 0:1]).float().to(device)
            y = torch.tensor(X[s:e, 1:2]).float().to(device)
            t = torch.tensor(X[s:e, 2:3]).float().to(device)

            # Forward pass
            T_b, F_b = self.net_TF(x, y, t)
            T_all.append(T_b.detach().cpu().numpy())
            F_all.append(F_b.detach().cpu().numpy())

        # Concatenate all batches
        return np.vstack(T_all), np.vstack(F_all)
    
def get_predictions(indices, t_unique, t_all, x_all, y_all, T_all, F_all, model, ny, nx):
    """
    Get true and predicted grids for a list of time indices.
    """
    results = []
    for vi in indices:
        t_i = t_unique[vi]
        mask_i = np.abs(t_all - t_i) < 1e-6
        X_i = np.column_stack([x_all[mask_i], y_all[mask_i], t_all[mask_i]])
        T_true_i = T_all[mask_i].reshape(ny, nx)
        F_true_i = F_all[mask_i].reshape(ny, nx)
        T_pred_i, F_pred_i = model.predict(X_i)
        T_pred_i = T_pred_i.reshape(ny, nx)
        F_pred_i = F_pred_i.reshape(ny, nx)
        results.append((t_i, T_true_i, T_pred_i, F_true_i, F_pred_i))
    return results

def plot_snapshots(results, tag, mode, ext, T_all, F_all):
    """
    Snapshots at selected time indices.
    """
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(24, 5 * n))
    fig.suptitle(f'{mode} - {tag} images', fontsize=14, y=1.01)
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, (t_i, T_true, T_pred, F_true, F_pred) in enumerate(results):
        tlabel = f't={t_i:.2f}'
        im = axes[i, 0].imshow(T_true, interpolation='nearest', cmap='hot',
                               extent=ext, origin='lower', aspect='auto', vmin=T_all.min(), vmax=T_all.max())
        axes[i, 0].set_title(f'T True ({tlabel}) [{tag}]'); axes[i, 0].set_ylabel('Y')
        divider = make_axes_locatable(axes[i, 0]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

        im = axes[i, 1].imshow(T_pred, interpolation='nearest', cmap='hot',
                               extent=ext, origin='lower', aspect='auto', vmin=T_all.min(), vmax=T_all.max())
        axes[i, 1].set_title(f'T {mode} ({tlabel}) [{tag}]'); axes[i, 1].set_ylabel('Y')
        divider = make_axes_locatable(axes[i, 1]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

        im = axes[i, 2].imshow(F_true, interpolation='nearest', cmap='viridis',
                               extent=ext, origin='lower', aspect='auto', vmin=F_all.min(), vmax=F_all.max())
        axes[i, 2].set_title(f'F True ({tlabel}) [{tag}]'); axes[i, 2].set_ylabel('Y')
        divider = make_axes_locatable(axes[i, 2]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

        im = axes[i, 3].imshow(F_pred, interpolation='nearest', cmap='viridis',
                               extent=ext, origin='lower', aspect='auto', vmin=F_all.min(), vmax=F_all.max())
        axes[i, 3].set_title(f'F {mode} ({tlabel}) [{tag}]'); axes[i, 3].set_ylabel('Y')
        divider = make_axes_locatable(axes[i, 3]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))
    plt.tight_layout()
    return fig

def plot_errors(results, tag, ext):
    """
    Snaposhots of T and F errors
    """
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    fig.suptitle(f'Error - {tag} images', fontsize=14, y=1.01)
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, (t_i, T_true, T_pred, F_true, F_pred) in enumerate(results):
        tlabel = f't={t_i:.2f}'
        err_T = np.abs(T_true - T_pred)
        err_F = np.abs(F_true - F_pred)
        mse_T = np.mean(err_T**2); mse_F = np.mean(err_F**2)
        rel_T = np.linalg.norm(err_T) / (np.linalg.norm(T_true) + 1e-10)
        rel_F = np.linalg.norm(err_F) / (np.linalg.norm(F_true) + 1e-10)

        im = axes[i, 0].imshow(err_T, interpolation='nearest', cmap='Reds',
                               extent=ext, origin='lower', aspect='auto')
        axes[i, 0].set_title(f'T Error ({tlabel}) [{tag}]')
        axes[i, 0].legend(handles=[
            Patch(fc='none', ec='none', label=f'Max: {err_T.max():.4e}'),
            Patch(fc='none', ec='none', label=f'Rel: {rel_T:.4e}'),
            Patch(fc='none', ec='none', label=f'MSE: {mse_T:.4e}'),
        ], loc='upper right', fontsize=8, framealpha=0.8)
        divider = make_axes_locatable(axes[i, 0]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

        im = axes[i, 1].imshow(err_F, interpolation='nearest', cmap='Reds',
                               extent=ext, origin='lower', aspect='auto')
        axes[i, 1].set_title(f'F Error ({tlabel}) [{tag}]')
        axes[i, 1].legend(handles=[
            Patch(fc='none', ec='none', label=f'Max: {err_F.max():.4e}'),
            Patch(fc='none', ec='none', label=f'Rel: {rel_F:.4e}'),
            Patch(fc='none', ec='none', label=f'MSE: {mse_F:.4e}'),
        ], loc='upper right', fontsize=8, framealpha=0.8)
        divider = make_axes_locatable(axes[i, 1]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))
    plt.tight_layout()
    return fig


#---------------------- Load Fire Data -----------------------------
print("Loading fire dataset...")
data_df = pd.read_csv(dataset)

x_all = data_df['x'].values
y_all = data_df['y'].values
t_all = data_df['t'].values
T_all = data_df['T'].values
F_all = data_df['F'].values

# Filter: keep only first N_IMAGES
t_unique_all = np.sort(np.unique(t_all))
t_max = t_unique_all[N_IMAGES - 1]
mask_keep = t_all <= t_max + 1e-6
x_all = x_all[mask_keep]
y_all = y_all[mask_keep]
t_all = t_all[mask_keep]
T_all = T_all[mask_keep]
F_all = F_all[mask_keep]

print(f"Using first {N_IMAGES} images (t_max={t_max:.2f})")
print(f"Total data points: {len(x_all)}")
print(f"t range: [{t_all.min():.2f}, {t_all.max():.2f}]")
print(f"T range: [{T_all.min():.4f}, {T_all.max():.4f}]")
print(f"F range: [{F_all.min():.4f}, {F_all.max():.4f}]")

X_star = np.column_stack([x_all, y_all, t_all])
TF_star = np.column_stack([T_all, F_all])

# Domain bounds
lb = X_star.min(0)
ub = X_star.max(0)
print(f"Domain bounds: lb={lb}, ub={ub}")

#---------------------- Configuration -----------------------------
beta = data_df['beta'].iloc[0]
lamb = data_df['lamb'].iloc[0]
Ta   = data_df['Ta'].iloc[0]
w1_q1 = data_df['w1_q1'].iloc[0]; w2_q1 = data_df['w2_q1'].iloc[0]
w1_q2 = data_df['w1_q2'].iloc[0]; w2_q2 = data_df['w2_q2'].iloc[0]
w1_q3 = data_df['w1_q3'].iloc[0]; w2_q3 = data_df['w2_q3'].iloc[0]
w1_q4 = data_df['w1_q4'].iloc[0]; w2_q4 = data_df['w2_q4'].iloc[0]

x_mid = (ub[0] + lb[0]) / 2
y_mid = lb[1] + (ub[1] - lb[1]) * (1 - TOP_FRACTION)

fire_params = {
    'beta': beta, 'lamb': lamb, 'Ta': Ta,
    'w1_q1': w1_q1, 'w2_q1': w2_q1, 'w1_q2': w1_q2, 'w2_q2': w2_q2,
    'w1_q3': w1_q3, 'w2_q3': w2_q3, 'w1_q4': w1_q4, 'w2_q4': w2_q4,
    'x_mid': x_mid, 'y_mid': y_mid,
}

print(f"\nFire parameters: beta={beta}, lamb={lamb}, Ta={Ta}")
print(f"  Q1(w1={w1_q1},w2={w2_q1}) Q2(w1={w1_q2},w2={w2_q2}) Q3(w1={w1_q3},w2={w2_q3}) Q4(w1={w1_q4},w2={w2_q4})")

# Train/val split by full images (random selection)
t_unique = np.sort(np.unique(t_all))
tol = 1e-6

# Randomly pick N_TRAIN images, remaining are validation
idx_all = np.arange(len(t_unique))
np.random.shuffle(idx_all)
idx_train = np.sort(idx_all[:N_TRAIN])
idx_val = np.sort(idx_all[N_TRAIN:])
t_train_set = set(t_unique[idx_train])
t_val_set = set(t_unique[idx_val])

t_train_times = t_unique[idx_train]
t_val_times = t_unique[idx_val]
print(f"\nTRAIN images ({N_TRAIN}): {['t=%.2f' % t for t in t_train_times]}")
print(f"VAL   images ({N_IMAGES - N_TRAIN}): {['t=%.2f' % t for t in t_val_times]}")

mask_train = np.array([t in t_train_set for t in t_all])
mask_val = np.array([t in t_val_set for t in t_all])

X_train_all = X_star[mask_train]
TF_train_all = TF_star[mask_train]
X_val = X_star[mask_val]
TF_val = TF_star[mask_val]

# Sample fraction of training data
N_train_total = X_train_all.shape[0]
N_use = int(N_train_total * sample_fraction)
if N_use < N_train_total:
    idx_sample = np.random.choice(N_train_total, N_use, replace=False)
    X_train = X_train_all[idx_sample]
    TF_train = TF_train_all[idx_sample]
else:
    X_train = X_train_all
    TF_train = TF_train_all

# Collocation points 
N_col = min(collocation_points, N_train_total)
idx_col = np.random.choice(N_train_total, N_col, replace=False)
X_collocation = X_train_all[idx_col]

# IC points (earliest training timestep)
t_min_train = t_unique[idx_train[0]]
ic_mask = np.abs(t_all - t_min_train) < tol
X_ic = X_star[ic_mask]
TF_ic = TF_star[ic_mask]

print(f"\nTrain: {X_train.shape[0]} points ({N_TRAIN} images, sample_fraction={sample_fraction})")
print(f"Validation: {X_val.shape[0]} points ({N_IMAGES - N_TRAIN} images)")
print(f"Collocation: {X_collocation.shape[0]} points")
print(f"IC: {X_ic.shape[0]} points")

layers = [3, 64, 64, 64, 64, 2]
print(f"Network layers: {layers}")
print(f"Weights: w_d={w_d}, w_p={w_p}, w_i={w_i}")

#------------------------ Training -----------------------------
print("\n" + "="*60)
mode = "PINN" if w_p > 0 else "Pure NN"
print(f"Starting training ({mode})...")
print("="*60)

model = FireNN(X_train, TF_train, X_collocation, X_ic, TF_ic, layers, lb, ub, fire_params)

start_time = time.time()
model.train()
elapsed = time.time() - start_time

print('\n' + "="*60)
print(f'Training complete! Time: {elapsed:.2f}s')
print("="*60)

#------------------------ Evaluation -----------------------------
print("\nEvaluating...")

T_pred_train, F_pred_train = model.predict(X_train_all)
T_pred_val, F_pred_val = model.predict(X_val)

# Train errors
err_T_train = np.linalg.norm(TF_train_all[:, 0:1] - T_pred_train, 2) / np.linalg.norm(TF_train_all[:, 0:1], 2)
err_F_train = np.linalg.norm(TF_train_all[:, 1:2] - F_pred_train, 2) / np.linalg.norm(TF_train_all[:, 1:2], 2)

# Validation errors
err_T_val = np.linalg.norm(TF_val[:, 0:1] - T_pred_val, 2) / np.linalg.norm(TF_val[:, 0:1], 2)
err_F_val = np.linalg.norm(TF_val[:, 1:2] - F_pred_val, 2) / np.linalg.norm(TF_val[:, 1:2], 2)

mse_T_train = np.mean((TF_train_all[:, 0] - T_pred_train.ravel()) ** 2)
mse_F_train = np.mean((TF_train_all[:, 1] - F_pred_train.ravel()) ** 2)
mse_T_val = np.mean((TF_val[:, 0] - T_pred_val.ravel()) ** 2)
mse_F_val = np.mean((TF_val[:, 1] - F_pred_val.ravel()) ** 2)

print(f'\n--- Train ({N_TRAIN} images) ---')
print(f'  Relative L2:  T={err_T_train:.6e}, F={err_F_train:.6e}')
print(f'  MSE:          T={mse_T_train:.6e}, F={mse_F_train:.6e}')
print(f'\n--- Validation ({N_IMAGES - N_TRAIN} images) ---')
print(f'  Relative L2:  T={err_T_val:.6e}, F={err_F_val:.6e}')
print(f'  MSE:          T={mse_T_val:.6e}, F={mse_F_val:.6e}')

#------------------------ Visualization -----------------------------
print("\nCreating visualizations...")
os.makedirs("Fire_PINN_troy/graphs_nn", exist_ok=True)

x_unique = np.unique(x_all)
y_unique = np.unique(y_all)
t_unique = np.sort(np.unique(t_all))
nx, ny = len(x_unique), len(y_unique)
ext = [x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()]


# Get predictions for train (3 samples) and val (all)
train_show = [idx_train[0], idx_train[len(idx_train)//2], idx_train[-1]]
res_train = get_predictions(train_show, t_unique, t_all, x_all, y_all, T_all, F_all, model, ny, nx)
res_val = get_predictions(idx_val, t_unique, t_all, x_all, y_all, T_all, F_all, model, ny, nx)

#Figure 1: Loss history
# Create iteration array, losses are recorded every 20 iterations
iterations = np.arange(20, len(model.loss_history) * 20 + 1, 20)

# Create loss plots
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

# Total Loss
axes1[0, 0].plot(iterations, model.loss_history, 'b-', linewidth=2)
axes1[0, 0].set_xlabel('Iteration', fontsize=12)
axes1[0, 0].set_ylabel('Total Loss', fontsize=12)
axes1[0, 0].set_title('Total Loss History', fontsize=14)
axes1[0, 0].set_yscale('log')
axes1[0, 0].grid(True, alpha=0.3)

# Data Loss vs PDE Loss
axes1[0, 1].plot(iterations, model.loss_data_history, 'r-', linewidth=2, label='Data Loss')
axes1[0, 1].plot(iterations, model.loss_pde_history, 'g-', linewidth=2, label='PDE Loss')
axes1[0, 1].set_xlabel('Iteration', fontsize=12)
axes1[0, 1].set_ylabel('Loss', fontsize=12)
axes1[0, 1].set_title('Data Loss vs PDE Loss', fontsize=14)
axes1[0, 1].set_yscale('log')
axes1[0, 1].legend(fontsize=11)
axes1[0, 1].grid(True, alpha=0.3)

# Temperature Losses
axes1[1, 0].plot(iterations, model.loss_T_data_history, 'r-', linewidth=2, label='T Data Loss')
axes1[1, 0].plot(iterations, model.loss_T_pde_history, 'g-', linewidth=2, label='T PDE Loss')
axes1[1, 0].set_xlabel('Iteration', fontsize=12)
axes1[1, 0].set_ylabel('Loss', fontsize=12)
axes1[1, 0].set_title('Temperature Losses', fontsize=14)
axes1[1, 0].set_yscale('log')
axes1[1, 0].legend(fontsize=11)
axes1[1, 0].grid(True, alpha=0.3)

# Fuel Losses
axes1[1, 1].plot(iterations, model.loss_F_data_history, 'r-', linewidth=2, label='F Data Loss')
axes1[1, 1].plot(iterations, model.loss_F_pde_history, 'g-', linewidth=2, label='F PDE Loss')
axes1[1, 1].set_xlabel('Iteration', fontsize=12)
axes1[1, 1].set_ylabel('Loss', fontsize=12)
axes1[1, 1].set_title('Fuel Losses', fontsize=14)
axes1[1, 1].set_yscale('log')
axes1[1, 1].legend(fontsize=11)
axes1[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Fire_PINN_troy/graphs_nn/fire_nn_loss_history.png', dpi=150, bbox_inches='tight')
print("Fig 1/5: Loss history saved")

#Figure 2: Training snapshots 
fig2 = plot_snapshots(res_train, 'Train', mode, ext, T_all, F_all)
fig2.savefig('Fire_PINN_troy/graphs_nn/fire_nn_train_snapshots.png', dpi=150, bbox_inches='tight')
print("Fig 2/5: Training snapshots saved")

#Figure 3: Training errors
fig3 = plot_errors(res_train, 'Train', ext)
fig3.savefig('Fire_PINN_troy/graphs_nn/fire_nn_train_errors.png', dpi=150, bbox_inches='tight')
print("Fig 3/5: Training errors saved")

#Figure 4: Validation snapshots
fig4 = plot_snapshots(res_val, 'Validation', mode, ext, T_all, F_all)
fig4.savefig('Fire_PINN_troy/graphs_nn/fire_nn_val_snapshots.png', dpi=150, bbox_inches='tight')
print("Fig 4/5: Validation snapshots saved")

#Figure 5: Validation errors
fig5 = plot_errors(res_val, 'Validation', ext)
fig5.savefig('Fire_PINN_troy/graphs_nn/fire_nn_val_errors.png', dpi=150, bbox_inches='tight')
print("Fig 5/5: Validation errors saved")

torch.save(model.dnn.state_dict(), 'Fire_PINN_troy/data/fire_nn_model.pth')
print("Model saved")

plt.show()
print("\nDone!")
