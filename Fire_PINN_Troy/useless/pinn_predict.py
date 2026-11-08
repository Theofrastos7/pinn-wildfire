"""
Fire NN/PINN - Train on first N_TRAIN images, predict the N_PREDICT-th.
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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Training configuration ──────────────────────────────────────────────────
max_iter = 12000
w_d = 1.0   # Data loss weight
w_p = 0.0   # Physics loss weight 
w_i = 0.0   # Initial condition loss weight
learning_rate = 0.4
sample_fraction = 0.2   # fraction of train pixels to use (0.0-1.0), e.g. 0.5 = 50%
collocation_points = 15000 # Collocation points for physics
dataset = 'Fire_PINN_troy/data/forward_dataset.csv'

N_TRAIN = 6   # train on first 14 images
N_PREDICT = 7 # predict the 15th

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


def get_frame(t_val):
    """
    Get true and predicted grids for a single timestep.
    """
    mask = np.abs(t_all - t_val) < tol
    X_i = np.column_stack([x_all[mask], y_all[mask], t_all[mask]])
    T_true = T_all[mask].reshape(ny, nx)
    F_true = F_all[mask].reshape(ny, nx)
    T_pred, F_pred = model.predict(X_i)
    return t_val, T_true, T_pred.reshape(ny, nx), F_true, F_pred.reshape(ny, nx)


def plot_snapshots(results, tag, mode, ext, T_ref, F_ref):
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(24, 5 * n))
    fig.suptitle(f'{mode} - {tag} images', fontsize=14, y=1.01)
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, (t_i, T_true, T_pred, F_true, F_pred) in enumerate(results):
        tlabel = f't={t_i:.2f}'
        im = axes[i, 0].imshow(T_true, interpolation='nearest', cmap='hot',
                               extent=ext, origin='lower', aspect='auto', vmin=T_ref.min(), vmax=T_ref.max())
        axes[i, 0].set_title(f'T True ({tlabel}) [{tag}]'); axes[i, 0].set_ylabel('Y')
        divider = make_axes_locatable(axes[i, 0]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

        im = axes[i, 1].imshow(T_pred, interpolation='nearest', cmap='hot',
                               extent=ext, origin='lower', aspect='auto', vmin=T_ref.min(), vmax=T_ref.max())
        axes[i, 1].set_title(f'T {mode} ({tlabel}) [{tag}]'); axes[i, 1].set_ylabel('Y')
        divider = make_axes_locatable(axes[i, 1]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

        im = axes[i, 2].imshow(F_true, interpolation='nearest', cmap='viridis',
                               extent=ext, origin='lower', aspect='auto', vmin=F_ref.min(), vmax=F_ref.max())
        axes[i, 2].set_title(f'F True ({tlabel}) [{tag}]'); axes[i, 2].set_ylabel('Y')
        divider = make_axes_locatable(axes[i, 2]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

        im = axes[i, 3].imshow(F_pred, interpolation='nearest', cmap='viridis',
                               extent=ext, origin='lower', aspect='auto', vmin=F_ref.min(), vmax=F_ref.max())
        axes[i, 3].set_title(f'F {mode} ({tlabel}) [{tag}]'); axes[i, 3].set_ylabel('Y')
        divider = make_axes_locatable(axes[i, 3]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))
    plt.tight_layout()
    return fig

def plot_errors(results, tag, ext):
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
print("\nLoading fire dataset...")
data_df = pd.read_csv(dataset)

x_all = data_df['x'].values
y_all = data_df['y'].values
t_all = data_df['t'].values
T_all = data_df['T'].values
F_all = data_df['F'].values

# Get unique times and split
t_unique_all = np.sort(np.unique(t_all))
t_train_max = t_unique_all[N_TRAIN - 1]
t_predict = t_unique_all[N_PREDICT - 1]
tol = 1e-6

mask_train = t_all <= t_train_max + tol
mask_predict = np.abs(t_all - t_predict) < tol

X_train_full = np.column_stack([x_all[mask_train], y_all[mask_train], t_all[mask_train]])
TF_train_full = np.column_stack([T_all[mask_train], F_all[mask_train]])

X_predict = np.column_stack([x_all[mask_predict], y_all[mask_predict], t_all[mask_predict]])
TF_predict = np.column_stack([T_all[mask_predict], F_all[mask_predict]])

# Print which images are used
t_train_times = t_unique_all[:N_TRAIN]
print(f"\nTRAIN images ({N_TRAIN}): {['t=%.2f' % t for t in t_train_times]}")
print(f"PREDICT image (#{N_PREDICT}): t={t_predict:.2f}")

print(f"Train points: {X_train_full.shape[0]}")
print(f"Predict points: {X_predict.shape[0]}")

# Domain bounds: extend t to include prediction time
lb = X_train_full.min(0)
ub = X_train_full.max(0)
ub[2] = t_predict
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

# Sample fraction of training data
N_train_total = X_train_full.shape[0]
N_use = int(N_train_total * sample_fraction)
if N_use < N_train_total:
    idx_sample = np.random.choice(N_train_total, N_use, replace=False)
    X_train = X_train_full[idx_sample]
    TF_train = TF_train_full[idx_sample]
else:
    X_train = X_train_full
    TF_train = TF_train_full

# Collocation points
N_col = min(collocation_points, N_train_total)
idx_col = np.random.choice(N_train_total, N_col, replace=False)
X_collocation = X_train_full[idx_col]

# IC points (first training timestep)
t_min_train = t_unique_all[0]
ic_mask = np.abs(t_all - t_min_train) < tol
X_ic = np.column_stack([x_all[ic_mask], y_all[ic_mask], t_all[ic_mask]])
TF_ic = np.column_stack([T_all[ic_mask], F_all[ic_mask]])

layers = [3, 64, 64, 64, 64, 2]

print(f"\nTrain: {X_train.shape[0]} points ({N_TRAIN} images, sample_fraction={sample_fraction})")
print(f"Collocation: {X_collocation.shape[0]} points")
print(f"IC: {X_ic.shape[0]} points")
print(f"Network layers: {layers}")
print(f"Weights: w_d={w_d}, w_p={w_p}, w_i={w_i}")

#------------------------ Training -----------------------------
print("\n" + "="*60)
mode = "PINN" if w_p > 0 else "Pure NN"
print(f"Starting training ({mode}), predict image #{N_PREDICT}...")
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
T_pred_tr, F_pred_tr = model.predict(X_train_full)

err_T_tr = np.linalg.norm(TF_train_full[:, 0:1] - T_pred_tr, 2) / np.linalg.norm(TF_train_full[:, 0:1], 2)
err_F_tr = np.linalg.norm(TF_train_full[:, 1:2] - F_pred_tr, 2) / np.linalg.norm(TF_train_full[:, 1:2], 2)
mse_T_tr = np.mean((TF_train_full[:, 0] - T_pred_tr.ravel()) ** 2)
mse_F_tr = np.mean((TF_train_full[:, 1] - F_pred_tr.ravel()) ** 2)

print(f'\n--- TRAIN ({N_TRAIN} images) ---')
print(f'  Relative L2:  T={err_T_tr:.6e}, F={err_F_tr:.6e}')
print(f'  MSE:          T={mse_T_tr:.6e}, F={mse_F_tr:.6e}')

T_pred_next, F_pred_next = model.predict(X_predict)

err_T_pred = np.linalg.norm(TF_predict[:, 0:1] - T_pred_next, 2) / np.linalg.norm(TF_predict[:, 0:1], 2)
err_F_pred = np.linalg.norm(TF_predict[:, 1:2] - F_pred_next, 2) / np.linalg.norm(TF_predict[:, 1:2], 2)
mse_T_pred = np.mean((TF_predict[:, 0] - T_pred_next.ravel()) ** 2)
mse_F_pred = np.mean((TF_predict[:, 1] - F_pred_next.ravel()) ** 2)

print(f'\n--- PREDICTION (image #{N_PREDICT}, t={t_predict:.2f}) ---')
print(f'  Relative L2:  T={err_T_pred:.6e}, F={err_F_pred:.6e}')
print(f'  MSE:          T={mse_T_pred:.6e}, F={mse_F_pred:.6e}')

#------------------------ Visualization -----------------------------
print("\nCreating visualizations...")
os.makedirs("Fire_PINN_troy/graphs_predict", exist_ok=True)

x_unique = np.unique(x_all[mask_predict])
y_unique = np.unique(y_all[mask_predict])
nx = len(x_unique)
ny = len(y_unique)
ext = [x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()]

t_train_unique = np.sort(np.unique(t_all[mask_train]))

# Get 3 training frames (first, mid, last)
train_show_times = [t_train_unique[0], t_train_unique[len(t_train_unique)//2], t_train_unique[-1]]
res_train = [get_frame(t) for t in train_show_times]

# Get prediction frame
res_pred = [get_frame(t_predict)]

# Figure 1: Loss history
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
plt.savefig('Fire_PINN_troy/graphs_predict/fire_predict_loss_history.png', dpi=150, bbox_inches='tight')
print("Fig 1/5: Loss history saved")

# Figure 2: Training snapshots (3 from train set)
fig2 = plot_snapshots(res_train, 'TRAIN', mode, ext, T_all, F_all)
fig2.savefig('Fire_PINN_troy/graphs_predict/fire_predict_train_snapshots.png', dpi=150, bbox_inches='tight')
print("Fig 2/5: Training snapshots saved")

# Figure 3: Training errors
fig3 = plot_errors(res_train, 'TRAIN', ext)
fig3.savefig('Fire_PINN_troy/graphs_predict/fire_predict_train_errors.png', dpi=150, bbox_inches='tight')
print("Fig 3/5: Training errors saved")

# Figure 4: Prediction snapshot (with previous frame for comparison)
t_previous = t_train_unique[-1]
res_prev = [get_frame(t_previous)]

fig4, axes = plt.subplots(2, 4, figsize=(24, 10))
fig4.suptitle(f'{mode} - Previous vs Prediction', fontsize=14, y=1.01)

# Row 0: previous frame (last training image) — true only
t_i, T_true_prev, T_pred_prev, F_true_prev, F_pred_prev = res_prev[0]
tlabel = f't={t_i:.2f}'

im = axes[0, 0].imshow(T_true_prev, interpolation='nearest', cmap='hot',
                        extent=ext, origin='lower', aspect='auto', vmin=T_all.min(), vmax=T_all.max())
axes[0, 0].set_title(f'T True ({tlabel}) [PREVIOUS]'); axes[0, 0].set_ylabel('Y')
divider = make_axes_locatable(axes[0, 0]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

im = axes[0, 1].imshow(T_pred_prev, interpolation='nearest', cmap='hot',
                        extent=ext, origin='lower', aspect='auto', vmin=T_all.min(), vmax=T_all.max())
axes[0, 1].set_title(f'T {mode} ({tlabel}) [PREVIOUS]'); axes[0, 1].set_ylabel('Y')
divider = make_axes_locatable(axes[0, 1]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

im = axes[0, 2].imshow(F_true_prev, interpolation='nearest', cmap='viridis',
                        extent=ext, origin='lower', aspect='auto', vmin=F_all.min(), vmax=F_all.max())
axes[0, 2].set_title(f'F True ({tlabel}) [PREVIOUS]'); axes[0, 2].set_ylabel('Y')
divider = make_axes_locatable(axes[0, 2]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

im = axes[0, 3].imshow(F_pred_prev, interpolation='nearest', cmap='viridis',
                        extent=ext, origin='lower', aspect='auto', vmin=F_all.min(), vmax=F_all.max())
axes[0, 3].set_title(f'F {mode} ({tlabel}) [PREVIOUS]'); axes[0, 3].set_ylabel('Y')
divider = make_axes_locatable(axes[0, 3]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

# Row 1: prediction frame
t_i, T_true_pr, T_pred_pr, F_true_pr, F_pred_pr = res_pred[0]
tlabel = f't={t_i:.2f}'

im = axes[1, 0].imshow(T_true_pr, interpolation='nearest', cmap='hot',
                        extent=ext, origin='lower', aspect='auto', vmin=T_all.min(), vmax=T_all.max())
axes[1, 0].set_title(f'T True ({tlabel}) [PREDICT]'); axes[1, 0].set_ylabel('Y')
divider = make_axes_locatable(axes[1, 0]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

im = axes[1, 1].imshow(T_pred_pr, interpolation='nearest', cmap='hot',
                        extent=ext, origin='lower', aspect='auto', vmin=T_all.min(), vmax=T_all.max())
axes[1, 1].set_title(f'T {mode} ({tlabel}) [PREDICT]'); axes[1, 1].set_ylabel('Y')
divider = make_axes_locatable(axes[1, 1]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

im = axes[1, 2].imshow(F_true_pr, interpolation='nearest', cmap='viridis',
                        extent=ext, origin='lower', aspect='auto', vmin=F_all.min(), vmax=F_all.max())
axes[1, 2].set_title(f'F True ({tlabel}) [PREDICT]'); axes[1, 2].set_ylabel('Y')
divider = make_axes_locatable(axes[1, 2]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

im = axes[1, 3].imshow(F_pred_pr, interpolation='nearest', cmap='viridis',
                        extent=ext, origin='lower', aspect='auto', vmin=F_all.min(), vmax=F_all.max())
axes[1, 3].set_title(f'F {mode} ({tlabel}) [PREDICT]'); axes[1, 3].set_ylabel('Y')
divider = make_axes_locatable(axes[1, 3]); plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.10))

plt.tight_layout()
fig4.savefig('Fire_PINN_troy/graphs_predict/fire_predict_prediction.png', dpi=150, bbox_inches='tight')
print("Fig 4/5: Prediction snapshot saved (with previous frame)")

# Figure 5: Prediction error (with previous frame for comparison)
fig5 = plot_errors(res_prev + res_pred, 'PREVIOUS+PREDICT', ext)
fig5.savefig('Fire_PINN_troy/graphs_predict/fire_predict_prediction_errors.png', dpi=150, bbox_inches='tight')
print("Fig 5/5: Prediction errors saved (with previous frame)")

torch.save(model.dnn.state_dict(), 'Fire_PINN_troy/data/fire_predict_model.pth')
print("Model saved")

plt.show()
print("\nDone!")
