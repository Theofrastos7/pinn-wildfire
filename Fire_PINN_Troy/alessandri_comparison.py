"""
Alessandri et al. (2021) Comparison — Causal XPINN

Train on all Troy fire frames EXCEPT the three Alessandri held-out frames
(15:21, 15:42, 15:55), then predict those three and compute the same
overlap metrics: Sorensen Similarity Index (SSI), Jaccard Similarity
Coefficient (JSC), and Kappa Statistics (KS).

Same XPINN architecture, causal training, and hyperparameters as
causal_xpinn_predict.py.
"""

import torch
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure, morphology
import math
import os
import time

np.random.seed(1234)
torch.manual_seed(1234)

# ══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (same as causal_xpinn_predict.py)
# ══════════════════════════════════════════════════════════════════════════
max_iter = 30000
w_d = 1.0
w_p = 5.0
w_i = 0.0
w_intf = 10.0
w_clamp = 1.0          # penalty for T exceeding physical bounds
T_CLAMP_MAX = 3.0      # max physically reasonable non-dim temperature
learning_rate = 0.4
data_points = 5000
collocation_points = 2000
interface_points = 500
dataset = 'Fire_PINN_troy/data/forward_dataset.csv'

epsilon = 10.0

T_COLD = 0.10
F_DEPLETED = 0.15
INTF_BAND_T = 0.05
INTF_BAND_F = 0.10

COL_FRAC_UNTOUCHED = 0.10
COL_FRAC_BURNING   = 0.70
COL_FRAC_BURNT     = 0.20

TOP_FRACTION = 330 / 830
OUTPUT_DIR = 'Fire_PINN_troy/graphs_alessandri_comparison'

layers_untouched = [3, 20, 20, 2]
layers_burning   = [3, 40, 40, 40, 40, 2]
layers_burnt     = [3, 20, 20, 2]

# Alessandri held-out frames (non-dim time values)
# 15:21 -> t=4020,  15:42 -> t=4440,  15:55 -> t=4700
HELDOUT_T = [4020.0, 4440.0, 4700.0]
HELDOUT_LABELS = ['15:21', '15:42', '15:55']

# Train only on frames up to and including 15:01
# Train on all frames except the three held-out ones

# Threshold for defining "burnt area" for overlap metrics
T_BURNT_THRESHOLD = 0.05  # slightly above ambient to filter noise

# Exclude the two very late frames (17:49, 18:30) — not in Alessandri
EXCLUDE_LATE_T = [6980.0, 7800.0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Causal epsilon: {epsilon}")
print(f"Held-out test frames: {HELDOUT_LABELS} (t={HELDOUT_T})")

REGION_NAMES = ['untouched', 'burning', 'burnt']
REGION_COLORS = {'untouched': '#2196F3', 'burning': '#FF5722', 'burnt': '#795548'}


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _to_t(arr, grad=False):
    return torch.tensor(arr, requires_grad=grad).float().to(device)


def _fmt(v, sig=3):
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10**e
    sup = str(e).translate(str.maketrans('0123456789-', '\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b'))
    return f"{m:.{sig-1}f}\u00d710{sup}"


def classify_points(T_vals, F_vals):
    burnt     = (F_vals < F_DEPLETED)
    untouched = (T_vals < T_COLD) & ~burnt
    burning   = ~untouched & ~burnt
    return untouched, burning, burnt


# ── Alessandri metrics ──────────────────────────────────────────────────

def compute_overlap_metrics(observed_mask, predicted_mask):
    """
    Compute SSI, JSC, KS between two binary masks (True = fire/burnt area).

    SSI  = 2|A∩B| / (|A| + |B|)                    Eq. (11)
    JSC  = |A∩B| / |A∪B|                            Eq. (12)
    KS   = (Pa - Pe) / (1 - Pe)                     Eq. (13)
    """
    A = observed_mask.astype(bool)
    B = predicted_mask.astype(bool)
    N = A.size  # total number of cells (|Omega|)

    intersection = np.sum(A & B)
    union = np.sum(A | B)
    size_A = np.sum(A)
    size_B = np.sum(B)

    # SSI
    if (size_A + size_B) > 0:
        SSI = 2.0 * intersection / (size_A + size_B)
    else:
        SSI = 1.0

    # JSC
    if union > 0:
        JSC = intersection / union
    else:
        JSC = 1.0

    # KS (Kappa)
    # Pa = |A∩B|/|Ω| + |Ω\(A∪B)|/|Ω|
    Pa = (intersection + (N - union)) / N
    # Pe = |A|*|B|/|Ω|^2 + |Ω\A|*|Ω\B|/|Ω|^2
    Pe = (size_A * size_B) / (N * N) + ((N - size_A) * (N - size_B)) / (N * N)

    if Pe < 1.0:
        KS = (Pa - Pe) / (1.0 - Pe)
    else:
        KS = 1.0

    return SSI, JSC, KS


# ══════════════════════════════════════════════════════════════════════════
#  DNN
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
#  CAUSAL XPINN  (identical to causal_xpinn_predict.py)
# ══════════════════════════════════════════════════════════════════════════

class CausalXPINN_Fire():
    def __init__(self, region_data, burning_col_groups, interface_data,
                 lb, ub, fire_params, future_col_groups=None):
        self.lb = _to_t(lb)
        self.ub = _to_t(ub)
        self.epsilon = epsilon

        layer_map = {
            'untouched': layers_untouched,
            'burning':   layers_burning,
            'burnt':     layers_burnt,
        }
        self.dnn = {r: DNN(layer_map[r]).to(device) for r in REGION_NAMES}

        self.has_data = {}
        self.has_col  = {}

        for r in REGION_NAMES:
            rd = region_data[r]
            if rd['X_data'].shape[0] > 0:
                self.has_data[r] = True
                setattr(self, f'x_d_{r}', _to_t(rd['X_data'][:, 0:1], True))
                setattr(self, f'y_d_{r}', _to_t(rd['X_data'][:, 1:2], True))
                setattr(self, f't_d_{r}', _to_t(rd['X_data'][:, 2:3], True))
                setattr(self, f'T_{r}',   _to_t(rd['TF_data'][:, 0:1]))
                setattr(self, f'F_{r}',   _to_t(rd['TF_data'][:, 1:2]))
            else:
                self.has_data[r] = False

            if r != 'burning' and rd['X_col'].shape[0] > 0:
                self.has_col[r] = True
                setattr(self, f'x_c_{r}', _to_t(rd['X_col'][:, 0:1], True))
                setattr(self, f'y_c_{r}', _to_t(rd['X_col'][:, 1:2], True))
                setattr(self, f't_c_{r}', _to_t(rd['X_col'][:, 2:3], True))
            elif r != 'burning':
                self.has_col[r] = False

        self.burning_col_groups = []
        for cg in burning_col_groups:
            x_c = _to_t(cg[:, 0:1], True)
            y_c = _to_t(cg[:, 1:2], True)
            t_c = _to_t(cg[:, 2:3], True)
            self.burning_col_groups.append((x_c, y_c, t_c))
        self.n_causal_groups = len(self.burning_col_groups)
        self.has_col['burning'] = self.n_causal_groups > 0

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

        # Future collocation groups (PDE-only, no data)
        self.future_col_groups = []
        if future_col_groups is not None:
            for cg in future_col_groups:
                x_c = _to_t(cg[:, 0:1], True)
                y_c = _to_t(cg[:, 1:2], True)
                t_c = _to_t(cg[:, 2:3], True)
                self.future_col_groups.append((x_c, y_c, t_c))
            print(f"  Future collocation groups: {len(self.future_col_groups)}")

        self.beta  = fire_params['beta']
        self.lamb  = fire_params['lamb']
        self.x_mid = fire_params['x_mid']
        self.y_mid = fire_params['y_mid']
        for q in ['q1', 'q2', 'q3', 'q4']:
            setattr(self, f'w1_{q}', fire_params[f'w1_{q}'])
            setattr(self, f'w2_{q}', fire_params[f'w2_{q}'])

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
        self.causal_weights_history = []

    def net_TF(self, x, y, t, dnn):
        x_n = 2.0 * (x - self.lb[0]) / (self.ub[0] - self.lb[0]) - 1.0
        y_n = 2.0 * (y - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1.0
        t_n = 2.0 * (t - self.lb[2]) / (self.ub[2] - self.lb[2]) - 1.0
        out = dnn(torch.cat([x_n, y_n, t_n], dim=1))
        return out[:, 0:1], out[:, 1:2]

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

    def _interface_loss(self, tag, ra, rb):
        x = getattr(self, f'x_i_{tag}')
        y = getattr(self, f'y_i_{tag}')
        t = getattr(self, f't_i_{tag}')

        T_a, F_a = self.net_TF(x, y, t, self.dnn[ra])
        T_b, F_b = self.net_TF(x, y, t, self.dnn[rb])

        loss_sol = torch.mean((T_a - T_b) ** 2) + torch.mean((F_a - F_b) ** 2)

        dTa_x = torch.autograd.grad(T_a, x, torch.ones_like(T_a), True, True)[0]
        dTa_y = torch.autograd.grad(T_a, y, torch.ones_like(T_a), True, True)[0]
        dTb_x = torch.autograd.grad(T_b, x, torch.ones_like(T_b), True, True)[0]
        dTb_y = torch.autograd.grad(T_b, y, torch.ones_like(T_b), True, True)[0]
        loss_grad = torch.mean((dTa_x - dTb_x) ** 2) + torch.mean((dTa_y - dTb_y) ** 2)

        pT_a, pF_a = self.net_pde(x, y, t, self.dnn[ra])
        pT_b, pF_b = self.net_pde(x, y, t, self.dnn[rb])
        loss_res = torch.mean((pT_a - pT_b) ** 2) + torch.mean((pF_a - pF_b) ** 2)

        return loss_sol + 0.1 * loss_grad + 0.01 * loss_res

    def loss_func(self):
        self.optimizer.zero_grad()

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

        loss_pde = torch.tensor(0.0, device=device)
        causal_weights = []

        if w_p > 0:
            for r in ['untouched', 'burnt']:
                if not self.has_col[r]:
                    continue
                x = getattr(self, f'x_c_{r}')
                y = getattr(self, f'y_c_{r}')
                t = getattr(self, f't_c_{r}')
                pT, pF = self.net_pde(x, y, t, self.dnn[r])
                loss_pde = loss_pde + torch.mean(pT ** 2) + torch.mean(pF ** 2)

            if self.has_col['burning']:
                cumulative = torch.tensor(0.0, device=device)
                burning_dnn = self.dnn['burning']
                for k in range(self.n_causal_groups):
                    x_c, y_c, t_c = self.burning_col_groups[k]
                    pT_k, pF_k = self.net_pde(x_c, y_c, t_c, burning_dnn)
                    r_k = torch.mean(pT_k ** 2) + torch.mean(pF_k ** 2)
                    w_k = torch.exp(-self.epsilon * cumulative)
                    loss_pde = loss_pde + w_k * r_k
                    cumulative = cumulative + r_k.detach()
                    causal_weights.append(w_k.detach().item())

        # Future collocation PDE loss (burning subnet, extrapolation window)
        loss_future = torch.tensor(0.0, device=device)
        if self.future_col_groups and w_p > 0:
            burning_dnn = self.dnn['burning']
            for x_c, y_c, t_c in self.future_col_groups:
                pT_f, pF_f = self.net_pde(x_c, y_c, t_c, burning_dnn)
                loss_future = loss_future + torch.mean(pT_f ** 2) + torch.mean(pF_f ** 2)
            loss_future = loss_future / len(self.future_col_groups)

        # Temperature clamping penalty (prevent blowup)
        loss_clamp = torch.tensor(0.0, device=device)
        if self.future_col_groups and w_clamp > 0:
            burning_dnn = self.dnn['burning']
            for x_c, y_c, t_c in self.future_col_groups:
                T_f, _ = self.net_TF(x_c, y_c, t_c, burning_dnn)
                excess = torch.relu(T_f - T_CLAMP_MAX)
                loss_clamp = loss_clamp + torch.mean(excess ** 2)
            loss_clamp = loss_clamp / len(self.future_col_groups)

        loss_intf = torch.tensor(0.0, device=device)
        for tag, ra, rb in self.interfaces:
            loss_intf = loss_intf + self._interface_loss(tag, ra, rb)

        loss = w_d * loss_data + w_p * loss_pde + w_p * loss_future \
             + w_intf * loss_intf + w_clamp * loss_clamp
        loss.backward()
        self.iter += 1

        if self.iter % 20 == 0:
            self.loss_history.append(loss.item())
            self.loss_data_history.append(loss_data.item())
            self.loss_pde_history.append(loss_pde.item())
            self.loss_intf_history.append(loss_intf.item())
            if causal_weights:
                self.causal_weights_history.append(causal_weights)

        if self.iter % 100 == 0:
            cw_str = ""
            if causal_weights:
                cw_str = "  CW: [" + " ".join(f"{w:.3f}" for w in causal_weights) + "]"
            print('Iter %d, Loss: %.5e, Data: %.5e, PDE: %.5e, Intf: %.5e%s'
                  % (self.iter, loss.item(), loss_data.item(),
                     loss_pde.item(), loss_intf.item(), cw_str))
        return loss

    def train(self):
        for r in REGION_NAMES:
            self.dnn[r].train()
        while self.iter < max_iter:
            self.optimizer.step(lambda: self.loss_func())
            if self.iter >= max_iter:
                break

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


# ══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Loading fire dataset...")
print("=" * 70)

data_df = pd.read_csv(dataset)

x_all = data_df['x'].values
y_all = data_df['y'].values
t_all = data_df['t'].values
T_all = data_df['T'].values
F_all = data_df['F'].values

# ── Time mapping ─────────────────────────────────────────────────────────
A_scale, B_scale = 100.0, 300.0
t_unique_full = np.unique(t_all)

print(f"\nAll {len(t_unique_full)} frames in dataset:")
for i, tv in enumerate(t_unique_full):
    t_sec = tv / (A_scale / B_scale)
    h = int(t_sec // 3600)
    m = int((t_sec % 3600) // 60)
    real_h = h + 12
    tag = ""
    if tv in HELDOUT_T:
        tag = "  <-- HELD OUT (test)"
    elif tv in EXCLUDE_LATE_T:
        tag = "  <-- EXCLUDED (late)"
    print(f"  [{i:>2d}]  t={tv:7.1f}  ->  {real_h}:{m:02d} UTC{tag}")

# ── Filter: exclude late frames, keep the 18 Alessandri-era frames ─────
keep_era = np.array([t not in EXCLUDE_LATE_T for t in t_all])
x_all = x_all[keep_era]; y_all = y_all[keep_era]; t_all = t_all[keep_era]
T_all = T_all[keep_era]; F_all = F_all[keep_era]

X_star  = np.column_stack([x_all, y_all, t_all])
TF_star = np.column_stack([T_all, F_all])

# ── Train / test split ─────────────────────────────────────────────────
heldout_set = set(HELDOUT_T)
train_mask = np.array([t not in heldout_set for t in t_all])

t_unique   = np.unique(t_all)
train_t_set = set(t_unique[np.array([t not in heldout_set for t in t_unique])])
train_t_sorted = sorted(train_t_set)
n_train_t  = len(train_t_sorted)

test_t_vals = sorted(heldout_set)
n_test_t    = len(test_t_vals)

lb = X_star.min(0)
ub = X_star.max(0)

print(f"\nDomain bounds:  lb = {lb}  |  ub = {ub}")
print(f"Training frames: {n_train_t}   Test frames: {n_test_t}")
print(f"Train t values: {train_t_sorted}")
print(f"Test  t values: {test_t_vals}")

# ── Fire parameters from CSV ───────────────────────────────────────────
beta  = data_df['beta'].iloc[0]
lamb  = data_df['lamb'].iloc[0]
Ta    = data_df['Ta'].iloc[0]
w1_q1 = data_df['w1_q1'].iloc[0]; w2_q1 = data_df['w2_q1'].iloc[0]
w1_q2 = data_df['w1_q2'].iloc[0]; w2_q2 = data_df['w2_q2'].iloc[0]
w1_q3 = data_df['w1_q3'].iloc[0]; w2_q3 = data_df['w2_q3'].iloc[0]
w1_q4 = data_df['w1_q4'].iloc[0]; w2_q4 = data_df['w2_q4'].iloc[0]

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

# ══════════════════════════════════════════════════════════════════════════
#  REGION CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Region classification")
print("=" * 70)

u_mask, b_mask, x_mask = classify_points(T_all, F_all)

region_labels = np.empty(len(T_all), dtype=object)
region_labels[u_mask] = 'untouched'
region_labels[b_mask] = 'burning'
region_labels[x_mask] = 'burnt'

# For each held-out frame, assign proxy region labels from the closest
# prior training frame
test_region_labels = {}
for test_t in test_t_vals:
    prior_train = [t for t in train_t_sorted if t < test_t]
    if prior_train:
        proxy_t = max(prior_train)
    else:
        proxy_t = train_t_sorted[0]

    proxy_mask = (t_all == proxy_t)
    proxy_T = T_all[proxy_mask]
    proxy_F = F_all[proxy_mask]
    u_p, b_p, x_p = classify_points(proxy_T, proxy_F)

    test_frame_mask = (t_all == test_t)
    n_pts = test_frame_mask.sum()
    rl = np.empty(n_pts, dtype=object)
    rl[u_p] = 'untouched'
    rl[b_p] = 'burning'
    rl[x_p] = 'burnt'
    test_region_labels[test_t] = rl

    t_sec = proxy_t / (A_scale / B_scale)
    h = int(t_sec // 3600) + 12
    m = int((t_sec % 3600) // 60)
    print(f"  Test t={test_t:.0f}: proxy from t={proxy_t:.0f} ({h}:{m:02d} UTC)")

for r in REGION_NAMES:
    n_train = np.sum((region_labels == r) & train_mask)
    print(f"  {r:>10s}:  train points = {n_train:>8,d}")

# ══════════════════════════════════════════════════════════════════════════
#  SAMPLE DATA, COLLOCATION, INTERFACES
# ══════════════════════════════════════════════════════════════════════════
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

    if r != 'burning':
        frac = {'untouched': COL_FRAC_UNTOUCHED, 'burnt': COL_FRAC_BURNT}[r]
        n_col_target = int(collocation_points * frac)
        r_train_indices = np.where(r_train_mask)[0]
        if len(r_train_indices) > 0:
            nc = min(n_col_target, len(r_train_indices))
            idx_col = np.random.choice(r_train_indices, nc, replace=False)
            X_c = X_star[idx_col]
        else:
            X_c = np.empty((0, 3))
    else:
        X_c = np.empty((0, 3))

    region_data[r] = {'X_data': X_d, 'TF_data': TF_d, 'X_col': X_c}
    print(f"  {r:>10s}:  data={X_d.shape[0]:>6,d}   col={X_c.shape[0]:>5,d}")

# ── Burning causal collocation groups (per timestep) ──────────────────
print("\nSampling burning causal collocation groups...")
n_col_burning_total = int(collocation_points * COL_FRAC_BURNING)
n_col_per_t = n_col_burning_total // n_train_t

burning_col_groups = []
burning_train_mask = (region_labels == 'burning') & train_mask

for t_val in train_t_sorted:
    t_burn_mask = burning_train_mask & (t_all == t_val)
    t_burn_idx = np.where(t_burn_mask)[0]
    if len(t_burn_idx) > 0:
        nc = min(n_col_per_t, len(t_burn_idx))
        idx = np.random.choice(t_burn_idx, nc, replace=False)
        burning_col_groups.append(X_star[idx])

print(f"  Burning causal groups: {len(burning_col_groups)} groups")
print(f"  Points per group: {[len(cg) for cg in burning_col_groups]}")
print(f"  Total burning collocation: {sum(len(cg) for cg in burning_col_groups)}")

# ── Interface sampling ─────────────────────────────────────────────────
print("\nSampling interface points...")
interface_data = {}

ub_band = train_mask & (np.abs(T_all - T_COLD) < INTF_BAND_T) & (F_all >= F_DEPLETED)
ub_idx  = np.where(ub_band)[0]
if len(ub_idx) > 0:
    ns = min(interface_points, len(ub_idx))
    interface_data['ub'] = X_star[np.random.choice(ub_idx, ns, replace=False)]
else:
    interface_data['ub'] = np.empty((0, 3))
print(f"  Untouched<->Burning:  {interface_data['ub'].shape[0]} points")

bb_band = train_mask & (np.abs(F_all - F_DEPLETED) < INTF_BAND_F) & (T_all >= T_COLD)
bb_idx  = np.where(bb_band)[0]
if len(bb_idx) > 0:
    ns = min(interface_points, len(bb_idx))
    interface_data['bb'] = X_star[np.random.choice(bb_idx, ns, replace=False)]
else:
    interface_data['bb'] = np.empty((0, 3))
print(f"  Burning<->Burnt:      {interface_data['bb'].shape[0]} points")

future_col_groups = None

# ══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"Starting Causal XPINN training (epsilon={epsilon})...")
print(f"  w_d={w_d}, w_p={w_p}, w_intf={w_intf}, max_iter={max_iter}")
print(f"  Training on {n_train_t} frames, testing on {n_test_t} held-out frames")
print("=" * 70)

model = CausalXPINN_Fire(region_data, burning_col_groups, interface_data,
                          lb, ub, fire_params, future_col_groups)

start_time = time.time()
model.train()
elapsed = time.time() - start_time

print('\n' + "=" * 70)
print(f'Training complete! Time: {elapsed:.2f}s  ({model.iter} iters)')
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
#  EVALUATION — L2 ERRORS + ALESSANDRI METRICS
# ══════════════════════════════════════════════════════════════════════════
print("\nEvaluating on held-out frames...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

x_unique = np.unique(x_all)
y_unique = np.unique(y_all)
nx = len(x_unique)
ny = len(y_unique)

print(f"Grid per frame: {nx} x {ny} = {nx*ny} points")

# ── Evaluate each test frame ──────────────────────────────────────────
results = []

for test_t, test_label in zip(test_t_vals, HELDOUT_LABELS):
    print(f"\n{'─'*60}")
    print(f"Test frame: {test_label} UTC  (t = {test_t:.1f})")
    print(f"{'─'*60}")

    # Extract observed data
    test_mask = (t_all == test_t)
    X_test = X_star[test_mask]
    T_obs  = T_all[test_mask]
    F_obs  = F_all[test_mask]

    # Get proxy region labels
    rl_test = test_region_labels[test_t]

    # Predict
    T_pred, F_pred = model.predict(X_test, rl_test)
    T_pred = T_pred.flatten()
    F_pred = F_pred.flatten()

    # L2 errors
    err_T = np.linalg.norm(T_obs - T_pred) / (np.linalg.norm(T_obs) + 1e-10)
    err_F = np.linalg.norm(F_obs - F_pred) / (np.linalg.norm(F_obs) + 1e-10)
    print(f"  Relative L2 — T: {_fmt(err_T)}   F: {_fmt(err_F)}")

    # Reshape to grid
    T_obs_grid  = T_obs.reshape(ny, nx)
    T_pred_grid = T_pred.reshape(ny, nx)
    F_obs_grid  = F_obs.reshape(ny, nx)
    F_pred_grid = F_pred.reshape(ny, nx)

    # ── Burnt area masks (for overlap metrics) ────────────────────────
    obs_burnt  = (T_obs_grid > T_BURNT_THRESHOLD)
    pred_burnt = (T_pred_grid > T_BURNT_THRESHOLD)

    SSI, JSC, KS = compute_overlap_metrics(obs_burnt, pred_burnt)
    print(f"  SSI  = {SSI:.4f}")
    print(f"  JSC  = {JSC:.4f}")
    print(f"  KS   = {KS:.4f}")
    print(f"  Observed fire pixels:  {obs_burnt.sum():,d} / {obs_burnt.size:,d}  "
          f"({100*obs_burnt.sum()/obs_burnt.size:.1f}%)")
    print(f"  Predicted fire pixels: {pred_burnt.sum():,d} / {pred_burnt.size:,d}  "
          f"({100*pred_burnt.sum()/pred_burnt.size:.1f}%)")

    # ── Persistence baseline (repeat last training frame) ─────────────
    prior_train = [t for t in train_t_sorted if t < test_t]
    persist_t = max(prior_train) if prior_train else train_t_sorted[-1]
    persist_mask = (t_all == persist_t)
    T_persist = T_all[persist_mask].reshape(ny, nx)
    persist_burnt = (T_persist > T_BURNT_THRESHOLD)
    SSI_p, JSC_p, KS_p = compute_overlap_metrics(obs_burnt, persist_burnt)

    err_T_persist = np.linalg.norm(T_obs - T_all[persist_mask]) / (np.linalg.norm(T_obs) + 1e-10)
    skill = 1.0 - err_T / (err_T_persist + 1e-10)

    print(f"  Persistence (t={persist_t:.0f}):  SSI={SSI_p:.4f}  JSC={JSC_p:.4f}  "
          f"KS={KS_p:.4f}  L2_T={_fmt(err_T_persist)}")
    print(f"  Skill score (1 - eps/eps_persist): {skill:.4f}")

    # ── Extract perimeters (morphological smoothing first) ─────────────
    disk = morphology.disk(5)
    obs_clean  = morphology.binary_closing(obs_burnt,  disk)
    obs_clean  = morphology.binary_opening(obs_clean,  disk)
    pred_clean = morphology.binary_closing(pred_burnt, disk)
    pred_clean = morphology.binary_opening(pred_clean, disk)
    obs_contours  = measure.find_contours(obs_clean.astype(float),  level=0.5)
    pred_contours = measure.find_contours(pred_clean.astype(float), level=0.5)

    results.append({
        'test_t': test_t,
        'label': test_label,
        'err_T': err_T, 'err_F': err_F,
        'SSI': SSI, 'JSC': JSC, 'KS': KS,
        'SSI_p': SSI_p, 'JSC_p': JSC_p, 'KS_p': KS_p,
        'err_T_persist': err_T_persist, 'skill': skill,
        'T_obs_grid': T_obs_grid, 'T_pred_grid': T_pred_grid,
        'F_obs_grid': F_obs_grid, 'F_pred_grid': F_pred_grid,
        'obs_burnt': obs_burnt, 'pred_burnt': pred_burnt,
        'obs_clean': obs_clean, 'pred_clean': pred_clean,
        'obs_contours': obs_contours, 'pred_contours': pred_contours,
    })

# ══════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COMPARISON WITH ALESSANDRI ET AL. (2021)")
print("=" * 70)

# Alessandri best results (Table 2, p.746 — with re-estimated p_hat_2)
alessandri = {
    '15:21': {'SSI': 0.9424, 'JSC': 0.9564, 'KS': 0.9703},
    '15:42': {'SSI': 0.8957, 'JSC': 0.9165, 'KS': 0.9450},
    '15:55': {'SSI': 0.8987, 'JSC': 0.9178, 'KS': 0.9466},
}

header = f"{'Time':>6s}  {'SSI':>7s}  {'JSC':>7s}  {'KS':>7s}  {'L2(T)':>8s}  " \
         f"{'SSI_A':>7s}  {'JSC_A':>7s}  {'KS_A':>7s}  {'Skill':>7s}"
print(header)
print("-" * len(header))

for res in results:
    lbl = res['label']
    a = alessandri.get(lbl, {})
    print(f"{lbl:>6s}  {res['SSI']:7.4f}  {res['JSC']:7.4f}  {res['KS']:7.4f}  "
          f"{res['err_T']:8.4f}  "
          f"{a.get('SSI', 0):7.4f}  {a.get('JSC', 0):7.4f}  {a.get('KS', 0):7.4f}  "
          f"{res['skill']:7.4f}")

print("-" * len(header))
print("Columns: Ours (SSI, JSC, KS, L2) | Alessandri (SSI_A, JSC_A, KS_A) | Skill")
print(f"Burnt area threshold: T > {T_BURNT_THRESHOLD}")

# ── Persistence baseline table ────────────────────────────────────────
print(f"\n{'Persistence Baseline':>30s}")
print(f"{'Time':>6s}  {'SSI_p':>7s}  {'JSC_p':>7s}  {'KS_p':>7s}  {'L2_p':>8s}")
for res in results:
    print(f"{res['label']:>6s}  {res['SSI_p']:7.4f}  {res['JSC_p']:7.4f}  "
          f"{res['KS_p']:7.4f}  {res['err_T_persist']:8.4f}")

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 1: Observed vs Predicted temperature + perimeter overlay
# ══════════════════════════════════════════════════════════════════════════
print("\nCreating figures...")

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle('Causal XPINN vs Observed — Alessandri Held-Out Frames',
             fontsize=15, y=0.98)

extent = [x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()]

for i, res in enumerate(results):
    # Col 0: Observed T
    ax = axes[i, 0]
    ax.set_facecolor('black')
    im = ax.imshow(res['T_obs_grid'], interpolation='nearest', cmap='hot',
                   vmin=0.0, vmax=2.0, origin='lower', extent=extent)
    ax.set_title(f'Observed — {res["label"]}', fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    divider = make_axes_locatable(ax)
    plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.08))

    # Col 1: Predicted T
    ax = axes[i, 1]
    ax.set_facecolor('black')
    im = ax.imshow(res['T_pred_grid'], interpolation='nearest', cmap='hot',
                   vmin=0.0, vmax=2.0, origin='lower', extent=extent)
    for c in res['pred_contours']:
        cy = y_unique[0] + c[:, 0] * (y_unique[-1] - y_unique[0]) / (ny - 1)
        cx = x_unique[0] + c[:, 1] * (x_unique[-1] - x_unique[0]) / (nx - 1)
        ax.plot(cx, cy, 'lime', lw=1.5, ls='--')
    ax.set_title(f'Causal XPINN — {res["label"]}', fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    divider = make_axes_locatable(ax)
    plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.08))

    # Col 2: Perimeter overlay
    ax = axes[i, 2]
    ax.set_facecolor('#f0f0f0')
    # Show observed burnt area as light red fill (cleaned)
    ax.imshow(res['obs_clean'].astype(float), interpolation='nearest',
              cmap='Reds', vmin=0, vmax=2, origin='lower', extent=extent, alpha=0.4)
    # Observed perimeter
    for c in res['obs_contours']:
        cy = y_unique[0] + c[:, 0] * (y_unique[-1] - y_unique[0]) / (ny - 1)
        cx = x_unique[0] + c[:, 1] * (x_unique[-1] - x_unique[0]) / (nx - 1)
        ax.plot(cx, cy, 'r-', lw=2, label='Observed' if c is res['obs_contours'][0] else None)
    # Predicted perimeter
    for c in res['pred_contours']:
        cy = y_unique[0] + c[:, 0] * (y_unique[-1] - y_unique[0]) / (ny - 1)
        cx = x_unique[0] + c[:, 1] * (x_unique[-1] - x_unique[0]) / (nx - 1)
        ax.plot(cx, cy, 'b--', lw=2, label='XPINN' if c is res['pred_contours'][0] else None)

    ax.set_title(f'Perimeter Overlay — {res["label"]}\n'
                 f'SSI={res["SSI"]:.3f}  JSC={res["JSC"]:.3f}  KS={res["KS"]:.3f}',
                 fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.legend(loc='upper right', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = f'{OUTPUT_DIR}/alessandri_comparison_overview.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f"  Saved: {fig_path}")

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 2: Bar chart comparison with Alessandri
# ══════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Overlap Metrics: Causal XPINN vs Alessandri et al. (2021)',
              fontsize=14, y=1.02)

metrics = ['SSI', 'JSC', 'KS']
metric_names = ['Sorensen (SSI)', 'Jaccard (JSC)', 'Kappa (KS)']
x_pos = np.arange(len(test_t_vals))
bar_w = 0.3

for j, (metric, mname) in enumerate(zip(metrics, metric_names)):
    ours = [res[metric] for res in results]
    aless = [alessandri[res['label']][metric] for res in results]
    persist = [res[f'{metric}_p'] for res in results]

    axes2[j].bar(x_pos - bar_w, aless,   bar_w, label='Alessandri (2021)', color='#2196F3', alpha=0.85)
    axes2[j].bar(x_pos,         ours,    bar_w, label='Causal XPINN (ours)', color='#FF5722', alpha=0.85)
    axes2[j].bar(x_pos + bar_w, persist, bar_w, label='Persistence', color='#9E9E9E', alpha=0.7)

    axes2[j].set_ylabel(mname, fontsize=12)
    axes2[j].set_xticks(x_pos)
    axes2[j].set_xticklabels(HELDOUT_LABELS)
    axes2[j].set_xlabel('Time')
    axes2[j].set_ylim(0.5, 1.0)
    axes2[j].legend(fontsize=9, loc='lower left')
    axes2[j].grid(axis='y', alpha=0.3)

    # Add value labels
    for k in range(len(x_pos)):
        axes2[j].text(x_pos[k] - bar_w, aless[k] + 0.005, f'{aless[k]:.3f}',
                      ha='center', va='bottom', fontsize=8)
        axes2[j].text(x_pos[k], ours[k] + 0.005, f'{ours[k]:.3f}',
                      ha='center', va='bottom', fontsize=8)

plt.tight_layout()
fig2_path = f'{OUTPUT_DIR}/alessandri_bar_comparison.png'
plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
print(f"  Saved: {fig2_path}")

# ══════════════════════════════════════════════════════════════════════════
#  FIGURE 3: Loss history
# ══════════════════════════════════════════════════════════════════════════
iterations = np.arange(20, len(model.loss_history) * 20 + 1, 20)

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

axes3[0, 0].plot(iterations, model.loss_history, 'b-', lw=2)
axes3[0, 0].set_xlabel('Iteration'); axes3[0, 0].set_ylabel('Total Loss')
axes3[0, 0].set_title('Total Loss'); axes3[0, 0].set_yscale('log'); axes3[0, 0].grid(True, alpha=0.3)

axes3[0, 1].plot(iterations, model.loss_data_history, 'r-', lw=2, label='Data')
axes3[0, 1].plot(iterations, model.loss_pde_history,  'g-', lw=2, label='PDE')
axes3[0, 1].plot(iterations, model.loss_intf_history,  'm-', lw=2, label='Interface')
axes3[0, 1].set_xlabel('Iteration'); axes3[0, 1].set_ylabel('Loss')
axes3[0, 1].set_title('Loss Components'); axes3[0, 1].set_yscale('log')
axes3[0, 1].legend(fontsize=11); axes3[0, 1].grid(True, alpha=0.3)

axes3[1, 0].plot(iterations, model.loss_data_history, 'r-', lw=2, label='Data')
axes3[1, 0].plot(iterations, model.loss_intf_history,  'm-', lw=2, label='Interface')
axes3[1, 0].set_xlabel('Iteration'); axes3[1, 0].set_ylabel('Loss')
axes3[1, 0].set_title('Data vs Interface'); axes3[1, 0].set_yscale('log')
axes3[1, 0].legend(fontsize=11); axes3[1, 0].grid(True, alpha=0.3)

if model.causal_weights_history:
    cw_arr = np.array(model.causal_weights_history)
    cw_iters = np.arange(20, len(cw_arr) * 20 + 1, 20)
    for k in range(cw_arr.shape[1]):
        axes3[1, 1].plot(cw_iters, cw_arr[:, k], lw=1.5,
                         label=f't={train_t_sorted[k]:.0f}', alpha=0.8)
    axes3[1, 1].set_xlabel('Iteration'); axes3[1, 1].set_ylabel('Causal Weight')
    axes3[1, 1].set_title('Burning Causal Weights'); axes3[1, 1].set_ylim(-0.05, 1.05)
    axes3[1, 1].legend(fontsize=7, ncol=3); axes3[1, 1].grid(True, alpha=0.3)
else:
    axes3[1, 1].text(0.5, 0.5, 'No causal weights', ha='center', va='center',
                     transform=axes3[1, 1].transAxes, fontsize=14)

plt.tight_layout()
fig3_path = f'{OUTPUT_DIR}/loss_history.png'
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {fig3_path}")

# ── Save models ──────────────────────────────────────────────────────────
for r in REGION_NAMES:
    torch.save(model.dnn[r].state_dict(), f'{OUTPUT_DIR}/causal_xpinn_{r}_model.pth')
print(f"\nModels saved to '{OUTPUT_DIR}/'")

# ── Also evaluate on training frames (spot check) ────────────────────────
print(f"\n{'─'*60}")
print("Training frame spot check (first, mid, last):")
spot_indices = [0, n_train_t // 2, n_train_t - 1]
for si in spot_indices:
    st = train_t_sorted[si]
    sm = (t_all == st)
    X_s = X_star[sm]
    rl_s = region_labels[sm]
    T_s_pred, _ = model.predict(X_s, rl_s)
    T_s_obs = T_all[sm]
    e = np.linalg.norm(T_s_obs - T_s_pred.flatten()) / (np.linalg.norm(T_s_obs) + 1e-10)

    t_sec = st / (A_scale / B_scale)
    h = int(t_sec // 3600) + 12
    m = int((t_sec % 3600) // 60)
    print(f"  t={st:.0f} ({h}:{m:02d} UTC)  L2(T) = {_fmt(e)}")

print(f"\n{'='*70}")
print("Done!")
print(f"{'='*70}")
