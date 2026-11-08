"""
Create a 3-panel comparison figure for the Troy Fire prediction:
  (a) Observed  (b) Plain NN (w_p=0)  (c) PINN (w_p=1)
Loads saved models from graphs_predict (NN) and graphs_predict1 (PINN).
"""
import torch
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

np.random.seed(1234)
torch.manual_seed(1234)

TOP_FRACTION = 330 / 830

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# ── Network definition (must match saved models) ──────────────────
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        self.layers = torch.nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x)


# ── Load dataset ──────────────────────────────────────────────────
dataset = 'Fire_PINN_troy/data/forward_dataset.csv'
data_df = pd.read_csv(dataset)

x_all = data_df['x'].values
y_all = data_df['y'].values
t_all = data_df['t'].values
T_all = data_df['T'].values

X_star = np.column_stack([x_all, y_all, t_all])
lb_full = X_star.min(0)
ub_full = X_star.max(0)

# Use first n=10 timesteps for training, predict the 11th
n = 10
t_unique_all = np.unique(t_all)
train_t_set = set(t_unique_all[:n])
pred_t_val = t_unique_all[n]

# Keep only first n+1 timesteps
keep_mask = np.array([t in train_t_set or t == pred_t_val for t in t_all])
X_star = X_star[keep_mask]
x_all = x_all[keep_mask]
y_all = y_all[keep_mask]
t_all = t_all[keep_mask]
T_all = T_all[keep_mask]

lb = X_star.min(0)
ub = X_star.max(0)
t_unique = np.unique(t_all)

# Extract predicted frame grid
pred_mask = (t_all == pred_t_val)
x_pred = x_all[pred_mask]
y_pred = y_all[pred_mask]
T_pred_true = T_all[pred_mask]

x_unique = np.unique(x_pred)
y_unique = np.unique(y_pred)
nx = len(x_unique)
ny = len(y_unique)

T_true_grid = T_pred_true.reshape(ny, nx)

lb_t = torch.tensor(lb).float().to(device)
ub_t = torch.tensor(ub).float().to(device)


def predict_frame(model_path, x, y, t_val):
    """Load a saved model and predict T on a single frame."""
    layers = [3, 32, 32, 32, 32, 2]
    dnn = DNN(layers).to(device)
    dnn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    dnn.eval()

    t = np.full_like(x, t_val)
    x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32).to(device)
    y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32).to(device)
    t_t = torch.tensor(t.reshape(-1, 1), dtype=torch.float32).to(device)

    x_norm = 2.0 * (x_t - lb_t[0]) / (ub_t[0] - lb_t[0]) - 1.0
    y_norm = 2.0 * (y_t - lb_t[1]) / (ub_t[1] - lb_t[1]) - 1.0
    t_norm = 2.0 * (t_t - lb_t[2]) / (ub_t[2] - lb_t[2]) - 1.0

    with torch.no_grad():
        out = dnn(torch.cat([x_norm, y_norm, t_norm], dim=1))
        T = out[:, 0:1].cpu().numpy().flatten()
    return T


# ── Predict from both models ─────────────────────────────────────
nn_model_path   = 'Fire_PINN_troy/graphs_predict/pinn_predict_model.pth'
pinn_model_path = 'Fire_PINN_troy/graphs_predict1/pinn_predict_model.pth'

print("Loading plain NN model...")
T_nn = predict_frame(nn_model_path, x_pred, y_pred, pred_t_val)
T_nn_grid = T_nn.reshape(ny, nx)

print("Loading PINN model...")
T_pinn = predict_frame(pinn_model_path, x_pred, y_pred, pred_t_val)
T_pinn_grid = T_pinn.reshape(ny, nx)

# ── Figure ────────────────────────────────────────────────────────
vmin, vmax = 0.0, 2.0

bg_color = '#d0d0d0'
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=bg_color)

titles = [
    '(a) Observed — 15:08',
    '(b) Classic NN — 15:08',
    '(c) PINN — 15:08',
]
grids = [T_true_grid, T_nn_grid, T_pinn_grid]

for ax, title, grid in zip(axes, titles, grids):
    ax.set_facecolor('black')
    im = ax.imshow(grid, interpolation='nearest', cmap='hot',
                   vmin=vmin, vmax=vmax, origin='lower',
                   extent=[x_unique.min(), x_unique.max(),
                           y_unique.min(), y_unique.max()])
    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    plt.colorbar(im, cax=cax)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.suptitle(r'Observed vs Predicted Temperature $\tilde{T}$ — Unseen Frame at 15:08',
             fontsize=15, y=0.86)

out_path = 'Fire_PINN_troy/graphs_predict1/troy_prediction_comparison.png'
out_path2 = 'Latex/results/troy_prediction_comparison.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=bg_color)
plt.savefig(out_path2, dpi=200, bbox_inches='tight', facecolor=bg_color)
print(f"\nSaved: {out_path}")
print(f"Saved: {out_path2}")
