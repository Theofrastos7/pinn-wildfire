"""
XPINN Fire Predict - Extended Physics-Informed Neural Network

Same domain decomposition as causal_xpinn_predict.py:
  - Untouched: cold, fuel intact  (T < T_COLD  and  F >= F_DEPLETED)
  - Burning:   active fire         (T >= T_COLD and  F >= F_DEPLETED)
  - Burnt:     fuel depleted       (F < F_DEPLETED)

Standard XPINN (no causal weighting) — all regions use pooled PDE loss.
Otherwise identical to causal_xpinn_predict.py for fair comparison.

Fire PDEs (same physics in every sub-domain, quadrant diffusion preserved):
  dT/dt = w1*T_xx + w2*T_yy + F*exp(-1/T) - lamb*T
  dF/dt = -beta*F*exp(-1/T)
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
n = 10                    # train on first n timesteps, predict (n+1)th
max_iter = 30000
w_d = 1.0                # data loss weight
w_p = 5.0                # physics (PDE) loss weight
w_i = 0.0                # initial-condition loss weight
w_intf = 5.0             # interface loss weight
learning_rate = 0.4
data_points = 5000        # per timestep
collocation_points = 2000 # total across all regions
interface_points = 500    # per interface
dataset = 'Fire_PINN_troy/data/forward_dataset.csv'

# Region thresholds
T_COLD = 0.10             # below this + fuel intact = untouched (~60°C)
F_DEPLETED = 0.15         # below this = burnt
INTF_BAND_T = 0.05        # half-width of T-based interface band
INTF_BAND_F = 0.10        # half-width of F-based interface band

# Collocation budget split (fractions of collocation_points)
COL_FRAC_UNTOUCHED = 0.10
COL_FRAC_BURNING   = 0.70
COL_FRAC_BURNT     = 0.20

TOP_FRACTION = 330 / 830
OUTPUT_DIR = 'Fire_PINN_troy/graphs_xpinn_predict'

# Per-region network architectures
layers_untouched = [3, 20, 20, 2]
layers_burning   = [3, 40, 40, 40, 40, 2]
layers_burnt     = [3, 20, 20, 2]

# CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_t(arr, grad=False):
    """Numpy -> torch tensor on device."""
    return torch.tensor(arr, requires_grad=grad).float().to(device)


def _fmt(v, sig=3):
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10**e
    sup = str(e).translate(str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))
    return f"{m:.{sig-1}f}×10{sup}"


def classify_points(T_vals, F_vals):
    """Classify each point into untouched / burning / burnt."""
    burnt     = (F_vals < F_DEPLETED)
    untouched = (T_vals < T_COLD) & ~burnt
    burning   = ~untouched & ~burnt
    return untouched, burning, burnt


REGION_NAMES = ['untouched', 'burning', 'burnt']
REGION_COLORS = {'untouched': '#2196F3', 'burning': '#FF5722', 'burnt': '#795548'}


# ── DNN ──────────────────────────────────────────────────────────────────────

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh
        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        self.layers = torch.nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x)


# ── XPINN ────────────────────────────────────────────────────────────────────

class XPINN_Fire():
    def __init__(self, region_data, interface_data, lb, ub, fire_params):
        """
        region_data : dict[region_name] -> {X_data, TF_data, X_col}
        interface_data : dict['ub'|'bb'] -> X_intf  (N x 3)
        """
        self.lb = _to_t(lb)
        self.ub = _to_t(ub)

        # ── Sub-networks ─────────────────────────────────────────────
        layer_map = {
            'untouched': layers_untouched,
            'burning':   layers_burning,
            'burnt':     layers_burnt,
        }
        self.dnn = {r: DNN(layer_map[r]).to(device) for r in REGION_NAMES}

        # ── Region data tensors ──────────────────────────────────────
        self.has_data = {}
        self.has_col  = {}

        for r in REGION_NAMES:
            rd = region_data[r]
            # Data points
            if rd['X_data'].shape[0] > 0:
                self.has_data[r] = True
                setattr(self, f'x_d_{r}', _to_t(rd['X_data'][:, 0:1], True))
                setattr(self, f'y_d_{r}', _to_t(rd['X_data'][:, 1:2], True))
                setattr(self, f't_d_{r}', _to_t(rd['X_data'][:, 2:3], True))
                setattr(self, f'T_{r}',   _to_t(rd['TF_data'][:, 0:1]))
                setattr(self, f'F_{r}',   _to_t(rd['TF_data'][:, 1:2]))
            else:
                self.has_data[r] = False

            # Collocation points (all regions use pooled collocation)
            if rd['X_col'].shape[0] > 0:
                self.has_col[r] = True
                setattr(self, f'x_c_{r}', _to_t(rd['X_col'][:, 0:1], True))
                setattr(self, f'y_c_{r}', _to_t(rd['X_col'][:, 1:2], True))
                setattr(self, f't_c_{r}', _to_t(rd['X_col'][:, 2:3], True))
            else:
                self.has_col[r] = False

        # ── Interface tensors ────────────────────────────────────────
        self.interfaces = []
        intf_pairs = {'ub': ('untouched', 'burning'), 'bb': ('burning', 'burnt')}
        for tag, (ra, rb) in intf_pairs.items():
            X_if = interface_data.get(tag, np.empty((0, 3)))
            if X_if.shape[0] > 0:
                setattr(self, f'x_i_{tag}', _to_t(X_if[:, 0:1], True))
                setattr(self, f'y_i_{tag}', _to_t(X_if[:, 1:2], True))
                setattr(self, f't_i_{tag}', _to_t(X_if[:, 2:3], True))
                self.interfaces.append((tag, ra, rb))
                print(f"  Interface {tag} ({ra}<->{rb}): {X_if.shape[0]} points")
            else:
                print(f"  Interface {tag} ({ra}<->{rb}): SKIPPED (no points)")

        # ── Fire parameters ──────────────────────────────────────────
        self.beta  = fire_params['beta']
        self.lamb  = fire_params['lamb']
        self.x_mid = fire_params['x_mid']
        self.y_mid = fire_params['y_mid']
        for q in ['q1', 'q2', 'q3', 'q4']:
            setattr(self, f'w1_{q}', fire_params[f'w1_{q}'])
            setattr(self, f'w2_{q}', fire_params[f'w2_{q}'])

        # ── Optimizer (only active networks) ─────────────────────────
        active = set()
        for r in REGION_NAMES:
            if self.has_data[r] or self.has_col[r]:
                active.add(r)
        for _, ra, rb in self.interfaces:
            active.add(ra)
            active.add(rb)
        self.active_regions = active

        all_params = []
        for r in active:
            all_params.extend(self.dnn[r].parameters())
        n_params = sum(p.numel() for p in all_params)
        print(f"  Active regions: {active}  ({n_params} total parameters)")

        self.optimizer = torch.optim.LBFGS(
            all_params,
            lr=learning_rate,
            max_iter=max_iter,
            max_eval=max_iter,
            history_size=50,
            tolerance_grad=0,
            tolerance_change=0,
            line_search_fn="strong_wolfe",
        )

        self.iter = 0
        self.loss_history = []
        self.loss_data_history = []
        self.loss_pde_history = []
        self.loss_intf_history = []

    # ── Forward through a specific sub-network ───────────────────────
    def net_TF(self, x, y, t, dnn):
        x_n = 2.0 * (x - self.lb[0]) / (self.ub[0] - self.lb[0]) - 1.0
        y_n = 2.0 * (y - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1.0
        t_n = 2.0 * (t - self.lb[2]) / (self.ub[2] - self.lb[2]) - 1.0
        out = dnn(torch.cat([x_n, y_n, t_n], dim=1))
        return out[:, 0:1], out[:, 1:2]

    # ── PDE residual through a specific sub-network ──────────────────
    def net_pde(self, x, y, t, dnn):
        T, F = self.net_TF(x, y, t, dnn)

        T_t  = torch.autograd.grad(T, t, torch.ones_like(T), True, True)[0]
        T_x  = torch.autograd.grad(T, x, torch.ones_like(T), True, True)[0]
        T_y  = torch.autograd.grad(T, y, torch.ones_like(T), True, True)[0]
        F_t  = torch.autograd.grad(F, t, torch.ones_like(F), True, True)[0]
        T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x), True, True)[0]
        T_yy = torch.autograd.grad(T_y, y, torch.ones_like(T_y), True, True)[0]

        T_safe   = torch.clamp(T, min=0.1)
        reaction = F * torch.exp(-1.0 / T_safe)

        is_x_hi = x >= self.x_mid
        is_y_hi = y >= self.y_mid
        w1 = torch.where(is_x_hi,
                 torch.where(is_y_hi, self.w1_q1, self.w1_q4),
                 torch.where(is_y_hi, self.w1_q2, self.w1_q3))
        w2 = torch.where(is_x_hi,
                 torch.where(is_y_hi, self.w2_q1, self.w2_q4),
                 torch.where(is_y_hi, self.w2_q2, self.w2_q3))

        pde_T = T_t - w1 * T_xx - w2 * T_yy - reaction + self.lamb * T
        pde_F = F_t + self.beta * reaction
        return pde_T, pde_F

    # ── Interface loss between two adjacent sub-networks ─────────────
    def _interface_loss(self, tag, ra, rb):
        x = getattr(self, f'x_i_{tag}')
        y = getattr(self, f'y_i_{tag}')
        t = getattr(self, f't_i_{tag}')

        T_a, F_a = self.net_TF(x, y, t, self.dnn[ra])
        T_b, F_b = self.net_TF(x, y, t, self.dnn[rb])

        # Solution continuity
        loss_sol = torch.mean((T_a - T_b) ** 2) + torch.mean((F_a - F_b) ** 2)

        # Gradient continuity
        dTa_x = torch.autograd.grad(T_a, x, torch.ones_like(T_a), True, True)[0]
        dTa_y = torch.autograd.grad(T_a, y, torch.ones_like(T_a), True, True)[0]
        dTb_x = torch.autograd.grad(T_b, x, torch.ones_like(T_b), True, True)[0]
        dTb_y = torch.autograd.grad(T_b, y, torch.ones_like(T_b), True, True)[0]
        loss_grad = torch.mean((dTa_x - dTb_x) ** 2) + torch.mean((dTa_y - dTb_y) ** 2)

        # Residual continuity
        pT_a, pF_a = self.net_pde(x, y, t, self.dnn[ra])
        pT_b, pF_b = self.net_pde(x, y, t, self.dnn[rb])
        loss_res = torch.mean((pT_a - pT_b) ** 2) + torch.mean((pF_a - pF_b) ** 2)

        return loss_sol + 0.1 * loss_grad + 0.01 * loss_res

    # ── Total loss ───────────────────────────────────────────────────
    def loss_func(self):
        self.optimizer.zero_grad()

        # Data loss ─ per region
        loss_data = torch.tensor(0.0, device=device)
        for r in REGION_NAMES:
            if not self.has_data[r]:
                continue
            x = getattr(self, f'x_d_{r}')
            y = getattr(self, f'y_d_{r}')
            t = getattr(self, f't_d_{r}')
            T_true = getattr(self, f'T_{r}')
            F_true = getattr(self, f'F_{r}')
            T_p, F_p = self.net_TF(x, y, t, self.dnn[r])
            loss_data = loss_data + torch.mean((T_true - T_p) ** 2) \
                                  + torch.mean((F_true - F_p) ** 2)

        # PDE loss ─ per region (all pooled, no causal weighting)
        loss_pde = torch.tensor(0.0, device=device)
        if w_p > 0:
            for r in REGION_NAMES:
                if not self.has_col[r]:
                    continue
                x = getattr(self, f'x_c_{r}')
                y = getattr(self, f'y_c_{r}')
                t = getattr(self, f't_c_{r}')
                pT, pF = self.net_pde(x, y, t, self.dnn[r])
                loss_pde = loss_pde + torch.mean(pT ** 2) + torch.mean(pF ** 2)

        # Interface loss
        loss_intf = torch.tensor(0.0, device=device)
        for tag, ra, rb in self.interfaces:
            loss_intf = loss_intf + self._interface_loss(tag, ra, rb)

        loss = w_d * loss_data + w_p * loss_pde + w_intf * loss_intf
        loss.backward()
        self.iter += 1

        if self.iter % 20 == 0:
            self.loss_history.append(loss.item())
            self.loss_data_history.append(loss_data.item())
            self.loss_pde_history.append(loss_pde.item())
            self.loss_intf_history.append(loss_intf.item())

        if self.iter % 100 == 0:
            print('Iter %d, Loss: %.5e, Data: %.5e, PDE: %.5e, Intf: %.5e'
                  % (self.iter, loss.item(), loss_data.item(),
                     loss_pde.item(), loss_intf.item()))

        return loss

    # ── Train ────────────────────────────────────────────────────────
    def train(self):
        for r in REGION_NAMES:
            self.dnn[r].train()
        while self.iter < max_iter:
            self.optimizer.step(lambda: self.loss_func())
            if self.iter >= max_iter:
                break

    # ── Predict (hard region assignment) ─────────────────────────────
    def predict(self, X, region_labels, batch_size=50000):
        for r in REGION_NAMES:
            self.dnn[r].eval()

        N = X.shape[0]
        T_pred = np.zeros((N, 1))
        F_pred = np.zeros((N, 1))

        for r in REGION_NAMES:
            mask = (region_labels == r)
            if not mask.any():
                continue
            X_r = X[mask]
            T_list, F_list = [], []
            for i in range(0, X_r.shape[0], batch_size):
                b = X_r[i:i + batch_size]
                x = _to_t(b[:, 0:1], True)
                y = _to_t(b[:, 1:2], True)
                t = _to_t(b[:, 2:3], True)
                Tb, Fb = self.net_TF(x, y, t, self.dnn[r])
                T_list.append(Tb.detach().cpu().numpy())
                F_list.append(Fb.detach().cpu().numpy())
            T_pred[mask] = np.vstack(T_list)
            F_pred[mask] = np.vstack(F_list)

        return T_pred, F_pred


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
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

X_star  = np.column_stack([x_all, y_all, t_all])
TF_star = np.column_stack([T_all, F_all])

lb = X_star.min(0)
ub = X_star.max(0)

print(f"\nDomain bounds:  lb = {lb}  |  ub = {ub}")

# ── Fire parameters from CSV ────────────────────────────────────────────────
beta  = data_df['beta'].iloc[0]
lamb  = data_df['lamb'].iloc[0]
Ta    = data_df['Ta'].iloc[0]
w1_q1 = data_df['w1_q1'].iloc[0];  w2_q1 = data_df['w2_q1'].iloc[0]
w1_q2 = data_df['w1_q2'].iloc[0];  w2_q2 = data_df['w2_q2'].iloc[0]
w1_q3 = data_df['w1_q3'].iloc[0];  w2_q3 = data_df['w2_q3'].iloc[0]
w1_q4 = data_df['w1_q4'].iloc[0];  w2_q4 = data_df['w2_q4'].iloc[0]

x_mid = (ub[0] + lb[0]) / 2
y_mid = lb[1] + (ub[1] - lb[1]) * (1 - TOP_FRACTION)

fire_params = {
    'beta': beta, 'lamb': lamb, 'Ta': Ta,
    'w1_q1': w1_q1, 'w2_q1': w2_q1,
    'w1_q2': w1_q2, 'w2_q2': w2_q2,
    'w1_q3': w1_q3, 'w2_q3': w2_q3,
    'w1_q4': w1_q4, 'w2_q4': w2_q4,
    'x_mid': x_mid, 'y_mid': y_mid,
}

print(f"\nFire params: beta={beta:.6f}, lamb={lamb:.6f}")
print(f"  Q1(TR) w1={w1_q1} w2={w2_q1}  |  Q2(TL) w1={w1_q2} w2={w2_q2}")
print(f"  Q3(BL) w1={w1_q3} w2={w2_q3}  |  Q4(BR) w1={w1_q4} w2={w2_q4}")
print(f"  x_mid={x_mid:.4f}, y_mid={y_mid:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN / PREDICT SPLIT
# ═══════════════════════════════════════════════════════════════════════════
t_unique = np.unique(t_all)
n_total  = len(t_unique)
assert n < n_total, f"n={n} must be < total timesteps ({n_total})"

train_t_set = set(t_unique[:n])
train_t_sorted = sorted(train_t_set)
pred_t_val  = t_unique[n]

keep = np.array([t in train_t_set or t == pred_t_val for t in t_all])
X_star  = X_star[keep];  TF_star = TF_star[keep]
x_all = x_all[keep]; y_all = y_all[keep]; t_all = t_all[keep]
T_all = T_all[keep]; F_all = F_all[keep]
train_mask = np.array([t in train_t_set for t in t_all])
pred_mask  = (t_all == pred_t_val)

lb = X_star.min(0)
ub = X_star.max(0)
t_unique = np.unique(t_all)

print(f"\nTrain on first {n} timesteps, PREDICT t={pred_t_val:.4f} (step {n+1})")

# ═══════════════════════════════════════════════════════════════════════════
#  REGION CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Region classification")
print("=" * 60)

u_mask, b_mask, x_mask = classify_points(T_all, F_all)

region_labels = np.empty(len(T_all), dtype=object)
region_labels[u_mask] = 'untouched'
region_labels[b_mask] = 'burning'
region_labels[x_mask] = 'burnt'

last_train_t = max(train_t_set)
last_t_mask  = (t_all == last_train_t)
last_T = T_all[last_t_mask]
last_F = F_all[last_t_mask]

pred_region_labels = np.empty(pred_mask.sum(), dtype=object)
u_last, b_last, x_last = classify_points(last_T, last_F)
pred_region_labels[u_last] = 'untouched'
pred_region_labels[b_last] = 'burning'
pred_region_labels[x_last] = 'burnt'

for r in REGION_NAMES:
    n_train = np.sum((region_labels == r) & train_mask)
    n_pred  = np.sum(region_labels[pred_mask] == r) if pred_mask.sum() > 0 else 0
    n_proxy = np.sum(pred_region_labels == r)
    print(f"  {r:>10s}:  train={n_train:>8,d}   pred(true)={n_pred:>8,d}   pred(proxy)={n_proxy:>8,d}")

# ═══════════════════════════════════════════════════════════════════════════
#  SAMPLE DATA, COLLOCATION, INTERFACES
# ═══════════════════════════════════════════════════════════════════════════
print("\nSampling training points...")

region_data = {}
for r in REGION_NAMES:
    r_train_mask = (region_labels == r) & train_mask
    idx_list = []
    for t_val in train_t_sorted:
        t_r_mask = np.where(r_train_mask & (t_all == t_val))[0]
        if len(t_r_mask) == 0:
            continue
        ns = min(data_points, len(t_r_mask))
        idx_list.append(np.random.choice(t_r_mask, ns, replace=False))

    if idx_list:
        idx_data = np.concatenate(idx_list)
        X_d  = X_star[idx_data]
        TF_d = TF_star[idx_data]
    else:
        X_d  = np.empty((0, 3))
        TF_d = np.empty((0, 2))

    # Collocation: pooled from training data in this region
    frac = {'untouched': COL_FRAC_UNTOUCHED,
            'burning':   COL_FRAC_BURNING,
            'burnt':     COL_FRAC_BURNT}[r]
    n_col_target = int(collocation_points * frac)
    r_train_indices = np.where(r_train_mask)[0]

    if len(r_train_indices) > 0:
        nc = min(n_col_target, len(r_train_indices))
        idx_col = np.random.choice(r_train_indices, nc, replace=False)
        X_c = X_star[idx_col]
    else:
        X_c = np.empty((0, 3))

    region_data[r] = {'X_data': X_d, 'TF_data': TF_d, 'X_col': X_c}
    print(f"  {r:>10s}:  data={X_d.shape[0]:>6,d}   col={X_c.shape[0]:>5,d}")

# ── Interface sampling ───────────────────────────────────────────────────
print("\nSampling interface points...")
interface_data = {}

ub_band = train_mask & (np.abs(T_all - T_COLD) < INTF_BAND_T) & (F_all >= F_DEPLETED)
ub_idx  = np.where(ub_band)[0]
if len(ub_idx) > 0:
    ns = min(interface_points, len(ub_idx))
    interface_data['ub'] = X_star[np.random.choice(ub_idx, ns, replace=False)]
else:
    interface_data['ub'] = np.empty((0, 3))
print(f"  Untouched<->Burning:  {interface_data['ub'].shape[0]} points  "
      f"(from {len(ub_idx)} candidates in band |T-{T_COLD}|<{INTF_BAND_T})")

bb_band = train_mask & (np.abs(F_all - F_DEPLETED) < INTF_BAND_F) & (T_all >= T_COLD)
bb_idx  = np.where(bb_band)[0]
if len(bb_idx) > 0:
    ns = min(interface_points, len(bb_idx))
    interface_data['bb'] = X_star[np.random.choice(bb_idx, ns, replace=False)]
else:
    interface_data['bb'] = np.empty((0, 3))
print(f"  Burning<->Burnt:      {interface_data['bb'].shape[0]} points  "
      f"(from {len(bb_idx)} candidates in band |F-{F_DEPLETED}|<{INTF_BAND_F})")

# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Starting XPINN training (no causal weighting)...")
print(f"  w_d={w_d}, w_p={w_p}, w_intf={w_intf}, max_iter={max_iter}")
print("=" * 60)

model = XPINN_Fire(region_data, interface_data, lb, ub, fire_params)

start_time = time.time()
model.train()
elapsed = time.time() - start_time

print('\n' + "=" * 60)
print(f'Training complete! Time: {elapsed:.2f}s  ({model.iter} iters)')
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
print("\nEvaluating...")

eval_labels = np.copy(region_labels)
eval_labels[pred_mask] = pred_region_labels

T_pred, F_pred = model.predict(X_star, eval_labels)

# ── Overall errors ───────────────────────────────────────────────────────
error_T = np.linalg.norm(T_all.reshape(-1, 1) - T_pred, 2) / np.linalg.norm(T_all.reshape(-1, 1), 2)
error_F = np.linalg.norm(F_all.reshape(-1, 1) - F_pred, 2) / np.linalg.norm(F_all.reshape(-1, 1), 2)
print(f'\n{"="*60}')
print(f'OVERALL  Relative L2 — T: {_fmt(error_T)}   F: {_fmt(error_F)}')

# ── Train vs Predict ─────────────────────────────────────────────────────
ti = np.where(train_mask)[0]
pi = np.where(pred_mask)[0]

err_T_train = np.linalg.norm(T_all[ti].reshape(-1,1) - T_pred[ti], 2) / np.linalg.norm(T_all[ti].reshape(-1,1), 2)
err_F_train = np.linalg.norm(F_all[ti].reshape(-1,1) - F_pred[ti], 2) / np.linalg.norm(F_all[ti].reshape(-1,1), 2)
print(f'\nTRAIN    Relative L2 — T: {_fmt(err_T_train)}   F: {_fmt(err_F_train)}')

err_T_pred = np.linalg.norm(T_all[pi].reshape(-1,1) - T_pred[pi], 2) / np.linalg.norm(T_all[pi].reshape(-1,1), 2)
err_F_pred = np.linalg.norm(F_all[pi].reshape(-1,1) - F_pred[pi], 2) / np.linalg.norm(F_all[pi].reshape(-1,1), 2)
print(f'PREDICT  Relative L2 — T: {_fmt(err_T_pred)}   F: {_fmt(err_F_pred)}')
print(f'         Mean abs    — T: {_fmt(np.mean(np.abs(T_all[pi].reshape(-1,1) - T_pred[pi])))}   F: {_fmt(np.mean(np.abs(F_all[pi].reshape(-1,1) - F_pred[pi])))}')
print(f'         Max  abs    — T: {_fmt(np.max(np.abs(T_all[pi].reshape(-1,1) - T_pred[pi])))}   F: {_fmt(np.max(np.abs(F_all[pi].reshape(-1,1) - F_pred[pi])))}')

print(f'\n  Predict/Train ratio — T: {err_T_pred/err_T_train:.2f}x   F: {err_F_pred/err_F_train:.2f}x')

# ── Per-region error breakdown ───────────────────────────────────────────
print(f'\n{"─"*60}')
print(f'Per-region Relative L2 (PREDICT timestep, hard assignment):')
for r in REGION_NAMES:
    r_mask = pred_mask & (eval_labels == r)
    idx_r = np.where(r_mask)[0]
    if len(idx_r) == 0:
        print(f'  {r:>10s}:  (no points)')
        continue
    eT = np.linalg.norm(T_all[idx_r].reshape(-1,1) - T_pred[idx_r], 2) / (np.linalg.norm(T_all[idx_r].reshape(-1,1), 2) + 1e-10)
    eF = np.linalg.norm(F_all[idx_r].reshape(-1,1) - F_pred[idx_r], 2) / (np.linalg.norm(F_all[idx_r].reshape(-1,1), 2) + 1e-10)
    print(f'  {r:>10s}:  T={_fmt(eT)}  F={_fmt(eF)}  ({len(idx_r):,d} pts)')

# ── Per-timestep L2 errors ───────────────────────────────────────────────
print(f'\n{"─"*60}')
print(f'Per-timestep Relative L2:')
print(f'  {"t":>8s}  {"T error":>10s}  {"F error":>10s}  {"split":>7s}')
for t_val in t_unique:
    idx_t = np.where(t_all == t_val)[0]
    eT = np.linalg.norm(T_all[idx_t].reshape(-1,1) - T_pred[idx_t], 2) / (np.linalg.norm(T_all[idx_t].reshape(-1,1), 2) + 1e-10)
    eF = np.linalg.norm(F_all[idx_t].reshape(-1,1) - F_pred[idx_t], 2) / (np.linalg.norm(F_all[idx_t].reshape(-1,1), 2) + 1e-10)
    tag = "PREDICT" if t_val == pred_t_val else "TRAIN"
    print(f'  {t_val:8.4f}  {_fmt(eT):>10s}  {_fmt(eF):>10s}  {tag:>7s}')
print(f'{"="*60}')

# ═══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
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

# ── 1. Region map ────────────────────────────────────────────────────────
print("  Plotting region decomposition map...")

region_id = np.zeros_like(T_all)
region_id[region_labels == 'untouched'] = 0
region_id[region_labels == 'burning']   = 1
region_id[region_labels == 'burnt']     = 2
region_grid = region_id.reshape(nt, ny, nx)

time_indices = sorted(set([0, nt // 2, nt - 2, nt - 1]))

from matplotlib.colors import ListedColormap
region_cmap = ListedColormap([REGION_COLORS['untouched'],
                               REGION_COLORS['burning'],
                               REGION_COLORS['burnt']])

fig_r, axes_r = plt.subplots(1, len(time_indices), figsize=(5 * len(time_indices), 4))
for i, tidx in enumerate(time_indices):
    t_val = t_unique[tidx]
    tag = "PREDICT" if t_val == pred_t_val else "TRAIN"
    im = axes_r[i].imshow(region_grid[tidx], cmap=region_cmap, vmin=-0.5, vmax=2.5,
                           origin='lower', aspect='auto',
                           extent=[x_unique.min(), x_unique.max(),
                                   y_unique.min(), y_unique.max()])
    axes_r[i].set_title(f't={t_val:.2f} ({tag})', fontsize=11)
    axes_r[i].set_xlabel('X'); axes_r[i].set_ylabel('Y')
legend_patches = [Patch(color=REGION_COLORS[r], label=r) for r in REGION_NAMES]
fig_r.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f'{OUTPUT_DIR}/xpinn_region_map.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/xpinn_region_map.png")

# ── 2. Snapshots (true vs predicted) ────────────────────────────────────
time_labels = [f't={t_unique[i]:.2f} ({"PREDICT" if t_unique[i]==pred_t_val else "TRAIN"})'
               for i in time_indices]

fig, axes = plt.subplots(len(time_indices), 4, figsize=(20, 4 * len(time_indices)))
for i, (tidx, tlabel) in enumerate(zip(time_indices, time_labels)):
    for j, (grid_true, grid_pred, cmap, var) in enumerate([
        (T_true_grid, T_pred_grid, 'hot',     'Temperature'),
        (F_true_grid, F_pred_grid, 'viridis', 'Fuel'),
    ]):
        col = j * 2
        vmin = T_all.min() if var == 'Temperature' else F_all.min()
        vmax = T_all.max() if var == 'Temperature' else F_all.max()
        for k, (grid, label) in enumerate([(grid_true, 'True'), (grid_pred, 'XPINN')]):
            im = axes[i, col + k].imshow(grid[tidx], interpolation='nearest', cmap=cmap,
                                          extent=[x_unique.min(), x_unique.max(),
                                                  y_unique.min(), y_unique.max()],
                                          origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
            axes[i, col + k].set_title(f'{var} {label} ({tlabel})', fontsize=10)
            axes[i, col + k].set_xlabel('X'); axes[i, col + k].set_ylabel('Y')
            div = make_axes_locatable(axes[i, col + k])
            plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.10))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/xpinn_predict_snapshots.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/xpinn_predict_snapshots.png")

# ── 3. Error maps ───────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(len(time_indices), 2, figsize=(14, 4 * len(time_indices)))
for i, (tidx, tlabel) in enumerate(zip(time_indices, time_labels)):
    for j, (true_g, pred_g, var) in enumerate([
        (T_true_grid, T_pred_grid, 'Temperature'),
        (F_true_grid, F_pred_grid, 'Fuel'),
    ]):
        err_g = true_g[tidx] - pred_g[tidx]
        mse   = np.mean(err_g ** 2)
        rel   = np.linalg.norm(err_g) / (np.linalg.norm(true_g[tidx]) + 1e-10)

        im = axes2[i, j].imshow(np.abs(err_g), interpolation='nearest', cmap='Reds',
                                 extent=[x_unique.min(), x_unique.max(),
                                         y_unique.min(), y_unique.max()],
                                 origin='lower', aspect='auto')
        axes2[i, j].set_title(f'{var} Error ({tlabel})', fontsize=10)
        axes2[i, j].set_xlabel('X'); axes2[i, j].set_ylabel('Y')
        axes2[i, j].legend(handles=[
            Patch(fc='none', ec='none', label=f'Min:  {_fmt(np.abs(err_g).min())}'),
            Patch(fc='none', ec='none', label=f'Max:  {_fmt(np.abs(err_g).max())}'),
            Patch(fc='none', ec='none', label=f'Rel:  {rel*100:.2f}%'),
            Patch(fc='none', ec='none', label=f'MSE: {_fmt(mse)}'),
        ], loc='upper right', fontsize=8, framealpha=0.8)
        div = make_axes_locatable(axes2[i, j])
        plt.colorbar(im, cax=div.append_axes("right", size="5%", pad=0.10))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/xpinn_predict_errors.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/xpinn_predict_errors.png")

# ── 4. Loss history ─────────────────────────────────────────────────────
iterations = np.arange(20, len(model.loss_history) * 20 + 1, 20)

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

axes4[0, 0].plot(iterations, model.loss_history, 'b-', lw=2)
axes4[0, 0].set_xlabel('Iteration'); axes4[0, 0].set_ylabel('Total Loss')
axes4[0, 0].set_title('Total Loss'); axes4[0, 0].set_yscale('log'); axes4[0, 0].grid(True, alpha=0.3)

axes4[0, 1].plot(iterations, model.loss_data_history, 'r-', lw=2, label='Data')
axes4[0, 1].plot(iterations, model.loss_pde_history,  'g-', lw=2, label='PDE')
axes4[0, 1].plot(iterations, model.loss_intf_history,  'm-', lw=2, label='Interface')
axes4[0, 1].set_xlabel('Iteration'); axes4[0, 1].set_ylabel('Loss')
axes4[0, 1].set_title('Loss Components'); axes4[0, 1].set_yscale('log')
axes4[0, 1].legend(fontsize=11); axes4[0, 1].grid(True, alpha=0.3)

axes4[1, 0].plot(iterations, model.loss_data_history, 'r-', lw=2, label='Data')
axes4[1, 0].plot(iterations, model.loss_intf_history,  'm-', lw=2, label='Interface')
axes4[1, 0].set_xlabel('Iteration'); axes4[1, 0].set_ylabel('Loss')
axes4[1, 0].set_title('Data vs Interface'); axes4[1, 0].set_yscale('log')
axes4[1, 0].legend(fontsize=11); axes4[1, 0].grid(True, alpha=0.3)

axes4[1, 1].plot(iterations, model.loss_pde_history,  'g-', lw=2, label='PDE')
axes4[1, 1].plot(iterations, model.loss_intf_history,  'm-', lw=2, label='Interface')
axes4[1, 1].set_xlabel('Iteration'); axes4[1, 1].set_ylabel('Loss')
axes4[1, 1].set_title('PDE vs Interface'); axes4[1, 1].set_yscale('log')
axes4[1, 1].legend(fontsize=11); axes4[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/xpinn_predict_loss_history.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/xpinn_predict_loss_history.png")

# ── Save models ──────────────────────────────────────────────────────────
for r in REGION_NAMES:
    torch.save(model.dnn[r].state_dict(), f'{OUTPUT_DIR}/xpinn_{r}_model.pth')
print(f"\nModels saved to '{OUTPUT_DIR}/'")

plt.show()
print("\nDone!")
