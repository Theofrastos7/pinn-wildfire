"""
Replot the Alessandri comparison overview using saved XPINN models.
No training — just load models, predict, and regenerate the figure
with cleaned perimeters.
"""
import torch
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure, morphology
from scipy import ndimage
import math

np.random.seed(1234)
torch.manual_seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Config (must match alessandri_comparison.py) ──────────────────
T_COLD = 0.10
F_DEPLETED = 0.15
T_BURNT_THRESHOLD = 0.05

HELDOUT_T = [4020.0, 4440.0, 4700.0]
HELDOUT_LABELS = ['15:21', '15:42', '15:55']
EXCLUDE_LATE_T = [6980.0, 7800.0]

REGION_NAMES = ['untouched', 'burning', 'burnt']
layers_map = {
    'untouched': [3, 20, 20, 2],
    'burning':   [3, 40, 40, 40, 40, 2],
    'burnt':     [3, 20, 20, 2],
}

MODEL_DIR = 'Fire_PINN_troy/graphs_alessandri_comparison'
OUTPUT_DIR = MODEL_DIR
dataset = 'Fire_PINN_troy/data/forward_dataset.csv'

n = 10
data_points = 5000


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


def classify_points(T_vals, F_vals):
    burnt     = (F_vals < F_DEPLETED)
    untouched = (T_vals < T_COLD) & ~burnt
    burning   = ~untouched & ~burnt
    return untouched, burning, burnt


def compute_overlap_metrics(obs, pred):
    A = obs.astype(bool)
    B = pred.astype(bool)
    intersection = np.sum(A & B)
    union = np.sum(A | B)
    sum_AB = np.sum(A) + np.sum(B)
    N = A.size
    if sum_AB == 0:
        return 1.0, 1.0, 1.0
    SSI = 2 * intersection / sum_AB
    JSC = intersection / union if union > 0 else 1.0
    Pa = (np.sum(A & B) + np.sum(~A & ~B)) / N
    Pe = (np.sum(A) * np.sum(B) + np.sum(~A) * np.sum(~B)) / (N * N)
    KS = (Pa - Pe) / (1 - Pe) if Pe < 1 else 1.0
    return SSI, JSC, KS


# ── Load data ─────────────────────────────────────────────────────
print("Loading dataset...")
data_df = pd.read_csv(dataset)
x_all = data_df['x'].values
y_all = data_df['y'].values
t_all = data_df['t'].values
T_all = data_df['T'].values
F_all = data_df['F'].values

# Remove late frames
keep = ~np.isin(t_all, EXCLUDE_LATE_T)
x_all = x_all[keep]; y_all = y_all[keep]; t_all = t_all[keep]
T_all = T_all[keep]; F_all = F_all[keep]

X_star = np.column_stack([x_all, y_all, t_all])

t_unique_all = np.unique(t_all)
train_times = sorted([t for t in t_unique_all if t not in HELDOUT_T])
test_t_vals = [t for t in HELDOUT_T if t in t_unique_all]

lb = X_star.min(0)
ub = X_star.max(0)
lb_t = torch.tensor(lb).float().to(device)
ub_t = torch.tensor(ub).float().to(device)

x_unique = np.unique(x_all)
y_unique = np.unique(y_all)
nx = len(x_unique)
ny = len(y_unique)

# ── Load models ───────────────────────────────────────────────────
print("Loading saved models...")
dnns = {}
for r in REGION_NAMES:
    dnn = DNN(layers_map[r]).to(device)
    dnn.load_state_dict(torch.load(
        f'{MODEL_DIR}/causal_xpinn_{r}_model.pth',
        map_location=device, weights_only=True))
    dnn.eval()
    dnns[r] = dnn


def predict_xpinn(X, region_labels):
    N = X.shape[0]
    T_pred = np.zeros(N)
    F_pred = np.zeros(N)
    for r in REGION_NAMES:
        mask = (region_labels == r)
        if not mask.any():
            continue
        b = X[mask]
        x = torch.tensor(b[:, 0:1], dtype=torch.float32).to(device)
        y = torch.tensor(b[:, 1:2], dtype=torch.float32).to(device)
        t = torch.tensor(b[:, 2:3], dtype=torch.float32).to(device)
        x_n = 2.0 * (x - lb_t[0]) / (ub_t[0] - lb_t[0]) - 1.0
        y_n = 2.0 * (y - lb_t[1]) / (ub_t[1] - lb_t[1]) - 1.0
        t_n = 2.0 * (t - lb_t[2]) / (ub_t[2] - lb_t[2]) - 1.0
        with torch.no_grad():
            out = dnns[r](torch.cat([x_n, y_n, t_n], dim=1))
        T_pred[mask] = out[:, 0].cpu().numpy()
        F_pred[mask] = out[:, 1].cpu().numpy()
    return T_pred, F_pred


# ── Get region labels for test frames (from last training frame before each) ──
def get_proxy_labels(test_t):
    prior = [t for t in train_times if t < test_t]
    ref_t = max(prior) if prior else train_times[-1]
    ref_mask = (t_all == ref_t)
    T_ref = T_all[ref_mask]
    F_ref = F_all[ref_mask]
    u, b, x = classify_points(T_ref, F_ref)
    labels = np.empty(ref_mask.sum(), dtype=object)
    labels[u] = 'untouched'
    labels[b] = 'burning'
    labels[x] = 'burnt'
    return labels


# ── Evaluate + plot ───────────────────────────────────────────────
print("Predicting and plotting...")

bg_color = '#d0d0d0'
fig, axes = plt.subplots(3, 3, figsize=(18, 11.2), facecolor=bg_color)
fig.suptitle('Observed vs Causal XPINN — Fire Perimeters on Held-Out Frames',
             fontsize=15, y=0.96)

extent = [x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()]
disk = morphology.disk(5)
all_metrics = []

for i, (test_t, label) in enumerate(zip(test_t_vals, HELDOUT_LABELS)):
    print(f"  {label}  (t={test_t:.0f})...")

    test_mask = (t_all == test_t)
    X_test = X_star[test_mask]
    T_obs = T_all[test_mask]
    F_obs = F_all[test_mask]

    rl = get_proxy_labels(test_t)
    T_pred, F_pred = predict_xpinn(X_test, rl)

    T_obs_grid = T_obs.reshape(ny, nx)
    T_pred_grid = T_pred.reshape(ny, nx)

    obs_burnt = (T_obs_grid > T_BURNT_THRESHOLD)
    pred_burnt = (T_pred_grid > T_BURNT_THRESHOLD)

    # Pad to 2880m x 2880m equivalent for fair comparison with Alessandri
    # Our domain: 2314m x 1414m, target: 2880m x 2880m
    pad_x = 88   # pixels to add in x (right side)
    pad_y = 236  # pixels to add in y (top side)
    obs_burnt_padded = np.pad(obs_burnt, ((0, pad_y), (0, pad_x)), constant_values=False)
    pred_burnt_padded = np.pad(pred_burnt, ((0, pad_y), (0, pad_x)), constant_values=False)

    SSI, JSC, KS = compute_overlap_metrics(obs_burnt_padded, pred_burnt_padded)

    # Morphological clean + keep only largest component + fill holes
    obs_clean = morphology.closing(morphology.opening(obs_burnt, disk), disk)
    pred_clean = morphology.closing(pred_burnt, disk)
    # Remove small isolated patches
    obs_labeled = measure.label(obs_clean)
    if obs_labeled.max() > 0:
        largest = max(range(1, obs_labeled.max()+1), key=lambda l: (obs_labeled==l).sum())
        obs_clean = (obs_labeled == largest)
    pred_labeled = measure.label(pred_clean)
    if pred_labeled.max() > 0:
        largest = max(range(1, pred_labeled.max()+1), key=lambda l: (pred_labeled==l).sum())
        pred_clean = (pred_labeled == largest)
    # Fill internal holes so contour traces only the outer boundary
    obs_clean = ndimage.binary_fill_holes(obs_clean)
    pred_clean = ndimage.binary_fill_holes(pred_clean)
    obs_contours = measure.find_contours(obs_clean.astype(float), level=0.5)
    pred_contours = measure.find_contours(pred_clean.astype(float), level=0.5)

    all_metrics.append((label, (SSI, JSC, KS)))
    print(f"    SSI={SSI:.4f}  JSC={JSC:.4f}  KS={KS:.4f}")

    # Col 0: Observed T
    ax = axes[i, 0]
    ax.set_facecolor('black')
    # Boost dark pixels inside the observed perimeter for display
    T_obs_disp = T_obs_grid.copy()
    T_obs_disp[obs_clean & (T_obs_disp < 0.3)] = 0.3
    im = ax.imshow(T_obs_disp, interpolation='nearest', cmap='hot',
                   vmin=0.0, vmax=2.0, origin='lower', extent=extent)
    for c in obs_contours:
        cy = y_unique[0] + c[:, 0] * (y_unique[-1] - y_unique[0]) / (ny - 1)
        cx = x_unique[0] + c[:, 1] * (x_unique[-1] - x_unique[0]) / (nx - 1)
        ax.plot(cx, cy, 'c-', lw=2.5)
    ax.set_title(f'Observed — {label} ', fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    divider = make_axes_locatable(ax)
    plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.08))

    # Col 1: Predicted T with perimeter
    ax = axes[i, 1]
    ax.set_facecolor('black')
    # Boost dark pixels inside the predicted perimeter for display
    T_pred_disp = T_pred_grid.copy()
    T_pred_disp[pred_clean & (T_pred_disp < 0.3)] = 0.3
    im = ax.imshow(T_pred_disp, interpolation='nearest', cmap='hot',
                   vmin=0.0, vmax=2.0, origin='lower', extent=extent)
    for c in pred_contours:
        cy = y_unique[0] + c[:, 0] * (y_unique[-1] - y_unique[0]) / (ny - 1)
        cx = x_unique[0] + c[:, 1] * (x_unique[-1] - x_unique[0]) / (nx - 1)
        ax.plot(cx, cy, 'lime', lw=2.5, ls='--')
    ax.set_title(f'Causal XPINN — {label} ', fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    divider = make_axes_locatable(ax)
    plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.08))

    # Col 2: Perimeter overlay (original domain)
    ax = axes[i, 2]
    ax.set_facecolor('#f0f0f0')
    ax.imshow(obs_clean.astype(float), interpolation='nearest',
              cmap='Reds', vmin=0, vmax=2, origin='lower', extent=extent, alpha=0.4)
    for c in obs_contours:
        cy = y_unique[0] + c[:, 0] * (y_unique[-1] - y_unique[0]) / (ny - 1)
        cx = x_unique[0] + c[:, 1] * (x_unique[-1] - x_unique[0]) / (nx - 1)
        ax.plot(cx, cy, 'r-', lw=2, label='Observed' if c is obs_contours[0] else None)
    for c in pred_contours:
        cy = y_unique[0] + c[:, 0] * (y_unique[-1] - y_unique[0]) / (ny - 1)
        cx = x_unique[0] + c[:, 1] * (x_unique[-1] - x_unique[0]) / (nx - 1)
        ax.plot(cx, cy, 'b--', lw=2, label='Causal XPINN' if c is pred_contours[0] else None)

    ax.set_title(f'Perimeter Overlay — {label} ', fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.legend(loc='upper right', fontsize=7, framealpha=0.7)

plt.subplots_adjust(hspace=0.12, wspace=0.30, top=0.94)
fig_path = f'{OUTPUT_DIR}/alessandri_comparison_overview.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor=bg_color)
fig_path2 = 'Latex/results/alessandri_comparison_overview.png'
plt.savefig(fig_path2, dpi=200, bbox_inches='tight', facecolor=bg_color)
print(f"\nSaved: {fig_path}")
print(f"Saved: {fig_path2}")

# ── Comparison table with Alessandri ──────────────────────────────
alessandri = {
    '15:21': {'SSI': 0.9424, 'JSC': 0.9564, 'KS': 0.9703},
    '15:42': {'SSI': 0.8957, 'JSC': 0.9165, 'KS': 0.9450},
    '15:55': {'SSI': 0.8987, 'JSC': 0.9178, 'KS': 0.9466},
}

print("\n" + "=" * 70)
print("COMPARISON WITH ALESSANDRI ET AL. (2021)")
print("=" * 70)
print(f"  {'Time':>6s}  {'SSI':>7s}  {'JSC':>7s}  {'KS':>7s}  ||  "
      f"{'SSI_A':>7s}  {'JSC_A':>7s}  {'KS_A':>7s}")
print("-" * 70)
for label, (SSI, JSC, KS) in all_metrics:
    a = alessandri.get(label, {})
    print(f"  {label:>6s}  {SSI:7.4f}  {JSC:7.4f}  {KS:7.4f}  ||  "
          f"{a.get('SSI',0):7.4f}  {a.get('JSC',0):7.4f}  {a.get('KS',0):7.4f}")
print("-" * 70)
print("  Left: Causal XPINN  ||  Right: Alessandri et al. (2021)")
