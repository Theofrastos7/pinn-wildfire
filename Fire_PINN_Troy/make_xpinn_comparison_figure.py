"""
Create a 4-panel comparison figure for the Troy Fire prediction:
  (a) Observed  (b) PINN  (c) XPINN  (d) Causal XPINN
Loads PINN model from graphs_predict1/, XPINN sub-networks from
graphs_xpinn_predict/, and Causal XPINN sub-networks from
graphs_causal_xpinn_predict/.
"""
import torch
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
torch.manual_seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Region thresholds (must match causal_xpinn_predict.py)
T_COLD = 0.10
F_DEPLETED = 0.15

REGION_NAMES = ['untouched', 'burning', 'burnt']
layers_map = {
    'untouched': [3, 20, 20, 2],
    'burning':   [3, 40, 40, 40, 40, 2],
    'burnt':     [3, 20, 20, 2],
}


class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh
        layer_list = []
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
F_all = data_df['F'].values

X_star = np.column_stack([x_all, y_all, t_all])

n = 10
t_unique_all = np.unique(t_all)
train_t_set = set(t_unique_all[:n])
pred_t_val = t_unique_all[n]

keep_mask = np.array([t in train_t_set or t == pred_t_val for t in t_all])
X_star = X_star[keep_mask]
x_all = x_all[keep_mask]; y_all = y_all[keep_mask]; t_all = t_all[keep_mask]
T_all = T_all[keep_mask]; F_all = F_all[keep_mask]

lb = X_star.min(0)
ub = X_star.max(0)
lb_t = torch.tensor(lb).float().to(device)
ub_t = torch.tensor(ub).float().to(device)

pred_mask = (t_all == pred_t_val)
x_pred = x_all[pred_mask]
y_pred = y_all[pred_mask]
T_pred_true = T_all[pred_mask]

x_unique = np.unique(x_pred)
y_unique = np.unique(y_pred)
nx = len(x_unique)
ny = len(y_unique)
T_true_grid = T_pred_true.reshape(ny, nx)


def normalize_and_predict(dnn, x, y, t_val):
    t = np.full_like(x, t_val)
    x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32).to(device)
    y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32).to(device)
    t_t = torch.tensor(t.reshape(-1, 1), dtype=torch.float32).to(device)
    x_n = 2.0 * (x_t - lb_t[0]) / (ub_t[0] - lb_t[0]) - 1.0
    y_n = 2.0 * (y_t - lb_t[1]) / (ub_t[1] - lb_t[1]) - 1.0
    t_n = 2.0 * (t_t - lb_t[2]) / (ub_t[2] - lb_t[2]) - 1.0
    with torch.no_grad():
        out = dnn(torch.cat([x_n, y_n, t_n], dim=1))
    return out[:, 0:1].cpu().numpy().flatten()


# ── PINN prediction ──────────────────────────────────────────────
print("Loading PINN model...")
pinn_dnn = DNN([3, 32, 32, 32, 32, 2]).to(device)
pinn_dnn.load_state_dict(torch.load(
    'Fire_PINN_troy/graphs_predict1/pinn_predict_model.pth',
    map_location=device, weights_only=True))
pinn_dnn.eval()
T_pinn = normalize_and_predict(pinn_dnn, x_pred, y_pred, pred_t_val)
T_pinn_grid = T_pinn.reshape(ny, nx)


# ── Helper: load XPINN sub-networks and predict ──────────────────
def load_xpinn_and_predict(model_dir, prefix):
    dnns = {}
    for r in REGION_NAMES:
        dnn = DNN(layers_map[r]).to(device)
        dnn.load_state_dict(torch.load(
            f'{model_dir}/{prefix}_{r}_model.pth',
            map_location=device, weights_only=True))
        dnn.eval()
        dnns[r] = dnn

    # Classify prediction points using last training frame
    last_train_t = max(train_t_set)
    last_mask = (t_all == last_train_t)
    last_T = T_all[last_mask]
    last_F = F_all[last_mask]

    burnt = (last_F < F_DEPLETED)
    untouched = (last_T < T_COLD) & ~burnt
    burning = ~untouched & ~burnt

    region_labels = np.empty(pred_mask.sum(), dtype=object)
    region_labels[untouched] = 'untouched'
    region_labels[burning] = 'burning'
    region_labels[burnt] = 'burnt'

    T_out = np.zeros(pred_mask.sum())
    for r in REGION_NAMES:
        mask = (region_labels == r)
        if not mask.any():
            continue
        T_out[mask] = normalize_and_predict(dnns[r], x_pred[mask], y_pred[mask], pred_t_val)

    return T_out.reshape(ny, nx)


# ── XPINN prediction ──────────────────────────────────────────────
print("Loading XPINN models...")
T_xpinn_grid = load_xpinn_and_predict(
    'Fire_PINN_troy/graphs_xpinn_predict', 'xpinn')

# ── Causal XPINN prediction ─────────────────────────────────────
print("Loading Causal XPINN models...")
T_cxpinn_grid = load_xpinn_and_predict(
    'Fire_PINN_troy/graphs_causal_xpinn_predict', 'causal_xpinn')

# ── Figure ────────────────────────────────────────────────────────
vmin, vmax = 0.0, 2.0
bg_color = '#d0d0d0'
fig, axes = plt.subplots(1, 4, figsize=(24, 5.5), facecolor=bg_color)

titles = [
    '(a) Observed — 15:08',
    '(b) PINN — 15:08',
    '(c) XPINN — 15:08',
    '(d) C-XPINN — 15:08',
]
grids = [T_true_grid, T_pinn_grid, T_xpinn_grid, T_cxpinn_grid]

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

out_path = 'Fire_PINN_troy/graphs_causal_xpinn_predict/xpinn_prediction_comparison.png'
out_path2 = 'Latex/results/xpinn_prediction_comparison.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=bg_color)
plt.savefig(out_path2, dpi=200, bbox_inches='tight', facecolor=bg_color)
print(f"\nSaved: {out_path}")
print(f"Saved: {out_path2}")
