"""
Compute L2, MAE, Max MAE for PINN, XPINN, C-XPINN on unseen 15:08 frame.
"""
import torch
from collections import OrderedDict
import numpy as np
import pandas as pd

np.random.seed(1234)
torch.manual_seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
T_true = T_all[pred_mask]
F_true = F_all[pred_mask]


def predict_TF(dnn, x, y, t_val):
    t = np.full_like(x, t_val)
    x_t = torch.tensor(x.reshape(-1, 1), dtype=torch.float32).to(device)
    y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32).to(device)
    t_t = torch.tensor(t.reshape(-1, 1), dtype=torch.float32).to(device)
    x_n = 2.0 * (x_t - lb_t[0]) / (ub_t[0] - lb_t[0]) - 1.0
    y_n = 2.0 * (y_t - lb_t[1]) / (ub_t[1] - lb_t[1]) - 1.0
    t_n = 2.0 * (t_t - lb_t[2]) / (ub_t[2] - lb_t[2]) - 1.0
    with torch.no_grad():
        out = dnn(torch.cat([x_n, y_n, t_n], dim=1))
    T_out = out[:, 0].cpu().numpy()
    F_out = out[:, 1].cpu().numpy()
    return T_out, F_out


def compute_metrics(true, pred, label):
    l2 = np.linalg.norm(true - pred) / np.linalg.norm(true)
    mae = np.mean(np.abs(true - pred))
    max_ae = np.max(np.abs(true - pred))
    print(f"  {label:5s}  L2={l2:.4e}  MAE={mae:.4e}  MaxAE={max_ae:.4e}")
    return l2, mae, max_ae


# PINN
print("=== PINN ===")
pinn_dnn = DNN([3, 32, 32, 32, 32, 2]).to(device)
pinn_dnn.load_state_dict(torch.load(
    'Fire_PINN_troy/graphs_predict1/pinn_predict_model.pth',
    map_location=device, weights_only=True))
pinn_dnn.eval()
T_pinn, F_pinn = predict_TF(pinn_dnn, x_pred, y_pred, pred_t_val)
compute_metrics(T_true, T_pinn, 'T')
compute_metrics(F_true, F_pinn, 'F')


# XPINN / C-XPINN helper
def load_xpinn_predict(model_dir, prefix, label):
    print(f"\n=== {label} ===")
    dnns = {}
    for r in REGION_NAMES:
        dnn = DNN(layers_map[r]).to(device)
        dnn.load_state_dict(torch.load(
            f'{model_dir}/{prefix}_{r}_model.pth',
            map_location=device, weights_only=True))
        dnn.eval()
        dnns[r] = dnn

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
    F_out = np.zeros(pred_mask.sum())
    for r in REGION_NAMES:
        mask = (region_labels == r)
        if not mask.any():
            continue
        T_r, F_r = predict_TF(dnns[r], x_pred[mask], y_pred[mask], pred_t_val)
        T_out[mask] = T_r
        F_out[mask] = F_r

    compute_metrics(T_true, T_out, 'T')
    compute_metrics(F_true, F_out, 'F')


load_xpinn_predict('Fire_PINN_troy/graphs_xpinn_predict', 'xpinn', 'XPINN')
load_xpinn_predict('Fire_PINN_troy/graphs_causal_xpinn_predict', 'causal_xpinn', 'C-XPINN')
