"""
Fire PINN - Physics-Informed Neural Network for fire propagation (2D)

Based on PINN structure for Burgers equation, adapted for 2D fire dynamics.
Input: (x, y, t) - 2D space and time coordinates
Output: (T, F) - temperature and fuel predictions

Fire PDEs (2D non-dimensional form):
- Temperature: ∂T/∂t = ∂²T/∂x² + ∂²T/∂y² + F*exp(-1/T) - lamb*T
- Fuel: ∂F/∂t = -beta*F*exp(-1/T)

where:
- lamb = C*B (cooling parameter)
- beta = B*CS/A (fuel consumption parameter)
- Ta = 0 (ambient temperature in non-dimensional form)
"""

import torch
import torch.nn.functional as TF_func
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import time
import os

np.random.seed(1234)
torch.manual_seed(1234)


max_iter   = 20000  # total iterations
lbfgs_iter = 100    # LBFGS steps per outer loop call (controls how often we get feedback)
w_d = 1.0
w_p = 10.0  # physics loss weight
w_i = 0.0
learning_rate = 0.5

data_points = 100000
collocation_points = 10000  # More collocation points for physics
dataset = 'Fire_PINN_Troy/data/pde_dataset.csv'

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
        # Input shape: N x 3 (x, y, and t)
        # Output shape: N x 2 (T and F)
        self.depth = len(layers) - 1

        # Activation function Tanh
        self.activation = torch.nn.Tanh
        
        # Build layer list
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
        
        # Deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)
        return out

#---------------------- Physics-Informed Neural Network ---------------------    
class FirePINN():
    def __init__(self, X_data, TF_data, X_collocation, X_ic, TF_ic, layers, lb, ub, fire_params):
        """
        X_data: training data points (N_data x 3) [x, y, t]
        TF_data: known T and F values at X_data (N_data x 2) [T (non-dim), F]
        X_collocation: collocation points for PDE residual (N_collocation x 3)
        X_ic: initial condition points at t=0 (N_ic x 3) [x, y, 0]
        TF_ic: initial T and F values at t=0 (N_ic x 2) [T (non-dim), F]
        layers: network architecture
        lb, ub: domain bounds
        fire_params: dict with x_mid, y_mid (quadrant split coords)
        T is already non-dimensional in the CSV (T_celsius / T_ref from main.py).
        """
        # Domain bounds
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # Training points, where T and F are known
        self.x_data = torch.tensor(X_data[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_data = torch.tensor(X_data[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.t_data = torch.tensor(X_data[:, 2:3], dtype=torch.float32, requires_grad=True).to(device)
        self.T_true = torch.tensor(TF_data[:, 0:1], dtype=torch.float32).to(device)
        self.F_true = torch.tensor(TF_data[:, 1:2], dtype=torch.float32).to(device)

        # Collocation points, for PDE residual
        self.x_col = torch.tensor(X_collocation[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_col = torch.tensor(X_collocation[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.t_col = torch.tensor(X_collocation[:, 2:3], dtype=torch.float32, requires_grad=True).to(device)

        # Initial condition points
        self.x_ic = torch.tensor(X_ic[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.y_ic = torch.tensor(X_ic[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.t_ic = torch.tensor(X_ic[:, 2:3], dtype=torch.float32, requires_grad=True).to(device)
        self.T_ic = torch.tensor(TF_ic[:, 0:1], dtype=torch.float32).to(device)
        self.F_ic = torch.tensor(TF_ic[:, 1:2], dtype=torch.float32).to(device)

        # Quadrant midpoints for spatially-varying diffusion
        self.x_mid = fire_params['x_mid']
        self.y_mid = fire_params['y_mid']

        # Learnable parameters beta, lambda
        # All reparameterized via softplus to enforce positivity
        _init_beta = 1.0
        _init_lamb = 0.1
        self.raw_beta = torch.nn.Parameter(torch.tensor([math.log(math.exp(_init_beta) - 1.0)], requires_grad=True))
        self.betas = []
        self.raw_lamb = torch.nn.Parameter(torch.tensor([math.log(math.exp(_init_lamb) - 1.0)], requires_grad=True))
        self.lambs = []

        # Learnable per-quadrant diffusion weights (8 parameters)
        # Reparameterized via softplus to enforce positivity
        _init_w = 0.3  # initial value for all w params
        _raw_w = math.log(math.exp(_init_w) - 1.0)  # inverse softplus
        self.raw_w1_q1 = torch.nn.Parameter(torch.tensor([_raw_w], requires_grad=True))
        self.raw_w2_q1 = torch.nn.Parameter(torch.tensor([_raw_w], requires_grad=True))
        self.raw_w1_q2 = torch.nn.Parameter(torch.tensor([_raw_w], requires_grad=True))
        self.raw_w2_q2 = torch.nn.Parameter(torch.tensor([_raw_w], requires_grad=True))
        self.raw_w1_q3 = torch.nn.Parameter(torch.tensor([_raw_w], requires_grad=True))
        self.raw_w2_q3 = torch.nn.Parameter(torch.tensor([_raw_w], requires_grad=True))
        self.raw_w1_q4 = torch.nn.Parameter(torch.tensor([_raw_w], requires_grad=True))
        self.raw_w2_q4 = torch.nn.Parameter(torch.tensor([_raw_w], requires_grad=True))

        # History tracking for per-quadrant w values
        self.w1_q1s, self.w2_q1s = [], []
        self.w1_q2s, self.w2_q2s = [], []
        self.w1_q3s, self.w2_q3s = [], []
        self.w1_q4s, self.w2_q4s = [], []

        # Network
        self.layers = layers
        self.dnn = DNN(layers).to(device)

        # Optimizer LBFGS
        # Add the learnable parameters to the optimizer as network parameters
        all_params = list(self.dnn.parameters())
        if w_p > 0 or w_i > 0:
            all_params += [self.raw_beta, self.raw_lamb,
                           self.raw_w1_q1, self.raw_w2_q1, self.raw_w1_q2, self.raw_w2_q2,
                           self.raw_w1_q3, self.raw_w2_q3, self.raw_w1_q4, self.raw_w2_q4]
        self.optimizer = torch.optim.LBFGS(
            all_params,
            lr=learning_rate,
            max_iter=lbfgs_iter,
            max_eval=lbfgs_iter + lbfgs_iter // 4,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
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
        """
        Network prediction: concatenate [x, y, t]
        normalize to [-1, 1] for better training
        return [T, F]
        """

        # Normalize inputs to [-1, 1]
        x_n = 2.0 * (x - self.lb[0]) / (self.ub[0] - self.lb[0]) - 1.0
        y_n = 2.0 * (y - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1.0
        t_n = 2.0 * (t - self.lb[2]) / (self.ub[2] - self.lb[2]) - 1.0
        output = self.dnn(torch.cat([x_n, y_n, t_n], dim=1))

        # output shape: N x 2, T and F
        T = output[:, 0:1]
        F = output[:, 1:2]
        return T, F

    def net_pde(self, x, y, t):
        """
        Compute PDE residuals in collocation points using autograd
        """
        T, F = self.net_TF(x, y, t)
        
        # First derivatives

        # T_t = ∂T/∂t
        T_t = torch.autograd.grad(
            T, t,
            grad_outputs=torch.ones_like(T),
            retain_graph=True,
            create_graph=True
        )[0]

        # T_x = ∂T/∂x
        T_x = torch.autograd.grad(
            T, x,
            grad_outputs=torch.ones_like(T),
            retain_graph=True,
            create_graph=True
        )[0]

        # T_y = ∂T/∂y
        T_y = torch.autograd.grad(
            T, y,
            grad_outputs=torch.ones_like(T),
            retain_graph=True,
            create_graph=True
        )[0]

        # F_t = ∂F/∂t
        F_t = torch.autograd.grad(
            F, t,
            grad_outputs=torch.ones_like(F),
            retain_graph=True,
            create_graph=True
        )[0]

        # Second derivatives

        # T_xx = ∂²T/∂x²
        T_xx = torch.autograd.grad(
            T_x, x,
            grad_outputs=torch.ones_like(T_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # T_yy = ∂²T/∂y²
        T_yy = torch.autograd.grad(
            T_y, y,
            grad_outputs=torch.ones_like(T_y),
            retain_graph=True,
            create_graph=True
        )[0]

        # Fire physics
        beta = TF_func.softplus(self.raw_beta)
        lamb = TF_func.softplus(self.raw_lamb)
        T_safe = torch.clamp(T, min=0.1) # Avoid small T for exp(-1/T)
        reaction_term = F * torch.exp(-1.0 / T_safe) # reaction term = F*exp(-1/T)
        cooling_term = lamb * T  # cooling term lamb*T
        fuel_consumption = beta * reaction_term # fuel consumption term = beta*F*exp(-1/T)

        # Per-quadrant diffusion weights (softplus ensures positivity)
        is_x_high = x >= self.x_mid
        is_y_high = y >= self.y_mid

        w1_q1 = TF_func.softplus(self.raw_w1_q1)
        w1_q2 = TF_func.softplus(self.raw_w1_q2)
        w1_q3 = TF_func.softplus(self.raw_w1_q3)
        w1_q4 = TF_func.softplus(self.raw_w1_q4)
        w2_q1 = TF_func.softplus(self.raw_w2_q1)
        w2_q2 = TF_func.softplus(self.raw_w2_q2)
        w2_q3 = TF_func.softplus(self.raw_w2_q3)
        w2_q4 = TF_func.softplus(self.raw_w2_q4)

        w1_spatial = torch.where(
            is_x_high,
            torch.where(is_y_high, w1_q1, w1_q4),
            torch.where(is_y_high, w1_q2, w1_q3)
        )
        w2_spatial = torch.where(
            is_x_high,
            torch.where(is_y_high, w2_q1, w2_q4),
            torch.where(is_y_high, w2_q2, w2_q3)
        )

        # PDE residuals
        # Temperature: ∂T/∂t = w1*∂²T/∂x² + w2*∂²T/∂y² + F*exp(-1/T) - lamb*T
        # Fuel: ∂F/∂t = -beta*F*exp(-1/T)
        pde_T = T_t - w1_spatial * T_xx - w2_spatial * T_yy - reaction_term + cooling_term
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
        else:
            loss_T_pde = torch.tensor(0.0).to(device)
            loss_F_pde = torch.tensor(0.0).to(device)
        loss_pde = loss_T_pde + loss_F_pde

        # Initial condition loss (at t=0)
        # only calculate if the weight of initial conditions is greater than 0
        if w_i > 0:
            T_ic_pred, F_ic_pred = self.net_TF(self.x_ic, self.y_ic, self.t_ic)
            loss_T_ic = torch.mean((self.T_ic - T_ic_pred) ** 2)
            loss_F_ic = torch.mean((self.F_ic - F_ic_pred) ** 2)
        else:
            loss_T_ic = torch.tensor(0.0).to(device)
            loss_F_ic = torch.tensor(0.0).to(device)
        loss_ic = loss_T_ic + loss_F_ic
        
        w_data = w_d
        w_pde = w_p
        w_ic = w_i

        # Total loss = data loss + PDE loss + initial condition loss
        loss = w_data * loss_data + w_pde * loss_pde + w_ic * loss_ic
        loss.backward()

        # Track learnable parameter values
        self.betas.append(TF_func.softplus(self.raw_beta).item())
        self.lambs.append(TF_func.softplus(self.raw_lamb).item())

        self.w1_q1s.append(TF_func.softplus(self.raw_w1_q1).item())
        self.w2_q1s.append(TF_func.softplus(self.raw_w2_q1).item())
        self.w1_q2s.append(TF_func.softplus(self.raw_w1_q2).item())
        self.w2_q2s.append(TF_func.softplus(self.raw_w2_q2).item())
        self.w1_q3s.append(TF_func.softplus(self.raw_w1_q3).item())
        self.w2_q3s.append(TF_func.softplus(self.raw_w2_q3).item())
        self.w1_q4s.append(TF_func.softplus(self.raw_w1_q4).item())
        self.w2_q4s.append(TF_func.softplus(self.raw_w2_q4).item())

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
            self.loss_T_ic_history.append(loss_T_ic.item())
            self.loss_F_ic_history.append(loss_F_ic.item())
        
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Data: %.5e (T:%.5e, F:%.5e), PDE: %.5e (T:%.5e, F:%.5e), IC: %.5e (T:%.5e, F:%.5e), beta: %.6f, lambda: %.6f'
                % (self.iter, loss.item(), loss_data.item(), loss_T_data.item(), loss_F_data.item(),
                   loss_pde.item(), loss_T_pde.item(), loss_F_pde.item(),
                   loss_ic.item(), loss_T_ic.item(), loss_F_ic.item(),
                   TF_func.softplus(self.raw_beta).item(), TF_func.softplus(self.raw_lamb).item())
            )
            print(
                '  Q1: w1=%.4f, w2=%.4f, Q2: w1=%.4f, w2=%.4f, Q3: w1=%.4f, w2=%.4f, Q4: w1=%.4f, w2=%.4f'
                % (TF_func.softplus(self.raw_w1_q1).item(), TF_func.softplus(self.raw_w2_q1).item(),
                   TF_func.softplus(self.raw_w1_q2).item(), TF_func.softplus(self.raw_w2_q2).item(),
                   TF_func.softplus(self.raw_w1_q3).item(), TF_func.softplus(self.raw_w2_q3).item(),
                   TF_func.softplus(self.raw_w1_q4).item(), TF_func.softplus(self.raw_w2_q4).item())
            )
        
        return loss
    
    def train(self, max_iterations=max_iter):
        """
        Train the network for a max_iter number of iterations
        """
        self.dnn.train()

        def closure():
            return self.loss_func()

        #ensure LBFGS runs for max_iterations
        while self.iter < max_iterations:
            self.optimizer.step(closure)

            # If LBFGS stops early, reinitialize it
            if self.iter < max_iterations:
                # Continue training if it hasn't reached max_iterations yet
                continue
            else:
                break

    def predict(self, X, batch_size=50000, compute_pde=False):
        """
        Predict T and F at given points X
        Using batches to avoid memory overflowing issues
        """
        self.dnn.eval()
        
        N = X.shape[0]
        T_all = []
        F_all = []
        pde_T_all = []
        pde_F_all = []
        
        # Process in batches
        num_batches = (N + batch_size - 1) // batch_size
        print(f"Processing {N} points in {num_batches} batches...")
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, N)
            X_batch = X[start_idx:end_idx, :]
            
            x = torch.tensor(X_batch[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
            y = torch.tensor(X_batch[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
            t = torch.tensor(X_batch[:, 2:3], dtype=torch.float32, requires_grad=True).to(device)
            
            # Forward pass
            T_batch, F_batch = self.net_TF(x, y, t)
            
            T_all.append(T_batch.detach().cpu().numpy())
            F_all.append(F_batch.detach().cpu().numpy())
            
            # Only compute PDE if requested to save memory
            if compute_pde:
                pde_T_batch, pde_F_batch = self.net_pde(x, y, t)
                pde_T_all.append(pde_T_batch.detach().cpu().numpy())
                pde_F_all.append(pde_F_batch.detach().cpu().numpy())
            
            if (i + 1) % 100 == 0 or (i + 1) == num_batches:
                print(f"  Processed batch {i+1}/{num_batches}")
        
        # Concatenate all batches
        T = np.vstack(T_all)
        F = np.vstack(F_all)
        
        if compute_pde and pde_T_all:
            pde_T = np.vstack(pde_T_all)
            pde_F = np.vstack(pde_F_all)
        else:
            pde_T = None
            pde_F = None
        
        return T, F, pde_T, pde_F
    

    def plot_beta_lambda_convergence(self):
        """
        Plot the convergence of beta and lambda parameters during training
        """
        final_beta = self.betas[-1] if self.betas else float('nan')
        final_lambda = self.lambs[-1] if self.lambs else float('nan')

        plt.figure(figsize=(10, 6))
        plt.plot(self.betas, 'b-', linewidth=2, label='Estimated beta')
        plt.xlabel('Iteration')
        plt.ylabel('Beta Parameter')
        plt.title('Beta Parameter Convergence During Inverse PINN Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        print(f"\nBeta Estimation Results:")
        print(f"Estimated: {final_beta:.4f}")

        plt.savefig('Fire_PINN_Troy/graphs_inverse/beta_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.lambs, 'b-', linewidth=2, label='Estimated lambda')
        plt.xlabel('Iteration')
        plt.ylabel('Lambda Parameter')
        plt.title('Lambda Parameter Convergence During Inverse PINN Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        print(f"\nLambda Estimation Results:")
        print(f"Estimated: {final_lambda:.4f}")

        plt.savefig('Fire_PINN_Troy/graphs_inverse/lambda_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_w_convergence(self):
        """
        Plot the convergence of per-quadrant diffusion weights during training
        """
        real_w1_q1 = fire_params.get('w1_q1', None)
        real_w2_q1 = fire_params.get('w2_q1', None)
        real_w1_q2 = fire_params.get('w1_q2', None)
        real_w2_q2 = fire_params.get('w2_q2', None)
        real_w1_q3 = fire_params.get('w1_q3', None)
        real_w2_q3 = fire_params.get('w2_q3', None)
        real_w1_q4 = fire_params.get('w1_q4', None)
        real_w2_q4 = fire_params.get('w2_q4', None)

        print(f"\nW Parameter Estimation Results:")
        for qname, hist, true_val in [
            ('Q1-w1', self.w1_q1s, real_w1_q1), ('Q1-w2', self.w2_q1s, real_w2_q1),
            ('Q2-w1', self.w1_q2s, real_w1_q2), ('Q2-w2', self.w2_q2s, real_w2_q2),
            ('Q3-w1', self.w1_q3s, real_w1_q3), ('Q3-w2', self.w2_q3s, real_w2_q3),
            ('Q4-w1', self.w1_q4s, real_w1_q4), ('Q4-w2', self.w2_q4s, real_w2_q4),
        ]:
            est = hist[-1] if hist else 0.0
            if true_val is not None and true_val != 0:
                err = abs(est - true_val) / abs(true_val) * 100
                print(f"  {qname}: True={true_val:.4f}, Est={est:.4f}, Error={err:.2f}%")
            else:
                print(f"  {qname}: Est={est:.4f}")

        _, axes = plt.subplots(2, 1, figsize=(12, 10))

        # w1 all quadrants
        for hist, color, label_q, true_val in [
            (self.w1_q1s, 'b', 'Q1', real_w1_q1), (self.w1_q2s, 'r', 'Q2', real_w1_q2),
            (self.w1_q3s, 'g', 'Q3', real_w1_q3), (self.w1_q4s, 'm', 'Q4', real_w1_q4),
        ]:
            final = hist[-1] if hist else 0.0
            axes[0].plot(hist, f'{color}-', linewidth=2, label=f'Est. w1_{label_q} = {final:.4f}')
            if true_val is not None:
                axes[0].axhline(y=true_val, color=color, linestyle='--', linewidth=2, alpha=0.7,
                                label=f'True w1_{label_q} = {true_val:.4f}')
        axes[0].set_ylabel('w1 (x-diffusion)')
        axes[0].set_title('W1 Parameters Convergence (All Quadrants)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # w2 all quadrants
        for hist, color, label_q, true_val in [
            (self.w2_q1s, 'b', 'Q1', real_w2_q1), (self.w2_q2s, 'r', 'Q2', real_w2_q2),
            (self.w2_q3s, 'g', 'Q3', real_w2_q3), (self.w2_q4s, 'm', 'Q4', real_w2_q4),
        ]:
            final = hist[-1] if hist else 0.0
            axes[1].plot(hist, f'{color}-', linewidth=2, label=f'Est. w2_{label_q} = {final:.4f}')
            if true_val is not None:
                axes[1].axhline(y=true_val, color=color, linestyle='--', linewidth=2, alpha=0.7,
                                label=f'True w2_{label_q} = {true_val:.4f}')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('w2 (y-diffusion)')
        axes[1].set_title('W2 Parameters Convergence (All Quadrants)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('Fire_PINN_Troy/graphs_inverse/w1_w2_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()

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

# Build coordinate and solution arrays
X_star = np.column_stack([x_all, y_all, t_all])  # All points (N x 3)
TF_star = np.column_stack([T_all, F_all])  # All solutions (N x 2)

print("Size of dataset: ", X_star.shape)

# Domain bounds
lb = X_star.min(0)
ub = X_star.max(0)

print("Size of lb and ub: ", lb.shape, ub.shape)

print(f"\nDomain bounds:")
print(f"  lb = {lb}") 
print(f"  ub = {ub}")

#---------------------- Configuration -----------------------------
# beta, lamb, w1_q1-w2_q4 are unknown — the PINN will estimate them (inverse problem)
# T is already non-dimensional in the CSV (T_celsius / T_ref=300 from main.py)

# Read per-quadrant w values from CSV (for reference / convergence plots)
w1_q1 = data_df['w1_q1'].iloc[0]
w2_q1 = data_df['w2_q1'].iloc[0]
w1_q2 = data_df['w1_q2'].iloc[0]
w2_q2 = data_df['w2_q2'].iloc[0]
w1_q3 = data_df['w1_q3'].iloc[0]
w2_q3 = data_df['w2_q3'].iloc[0]
w1_q4 = data_df['w1_q4'].iloc[0]
w2_q4 = data_df['w2_q4'].iloc[0]
beta_true = data_df['beta'].iloc[0]
lamb_true = data_df['lamb'].iloc[0]

fire_params = {
    'w1_q1': w1_q1, 'w2_q1': w2_q1,
    'w1_q2': w1_q2, 'w2_q2': w2_q2,
    'w1_q3': w1_q3, 'w2_q3': w2_q3,
    'w1_q4': w1_q4, 'w2_q4': w2_q4,
    'beta': beta_true, 'lamb': lamb_true,
    'x_mid': (ub[0] + lb[0]) / 2,
    'y_mid': lb[1] + (ub[1] - lb[1]) * (1 - 330 / 830),  # TOP_FRACTION=330/830, matches forward sim
}

print(f"\nFire parameters (true values from CSV):")
print(f"  Q1: w1={w1_q1}, w2={w2_q1}")
print(f"  Q2: w1={w1_q2}, w2={w2_q2}")
print(f"  Q3: w1={w1_q3}, w2={w2_q3}")
print(f"  Q4: w1={w1_q4}, w2={w2_q4}")
print(f"  beta={beta_true}, lamb={lamb_true}")
print(f"  x_mid={fire_params['x_mid']:.4f}, y_mid={fire_params['y_mid']:.4f}")

# Training configuration
N_data = data_points      # Number of data points to use for training
N_collocation = collocation_points  # Number of collocation points for PDE residual

layers = [3, 20, 20, 20, 20, 2]  # Smaller network forces reliance on physics terms

print(f"\nTraining configuration:")
print(f"  N_data = {N_data}")
print(f"  N_collocation = {N_collocation}")
print(f"  Network layers = {layers}")

# Randomly sample training data points with same
N_data = min(N_data, X_star.shape[0])
idx_data = np.random.choice(X_star.shape[0], N_data, replace=False)
X_data = X_star[idx_data, :]
TF_data = TF_star[idx_data, :]

# Randomly sample collocation points
X_collocation = np.column_stack([
    np.random.uniform(lb[0], ub[0], N_collocation),
    np.random.uniform(lb[1], ub[1], N_collocation),
    np.random.uniform(lb[2], ub[2], N_collocation),
])

# Extract initial condition points (t=0)
ic_mask = (t_all == t_all.min())
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
print("="*60)

# ── Estimated inverse parameters ──────────────────────────────────────────────
final_beta = TF_func.softplus(model.raw_beta).item()
final_lamb = TF_func.softplus(model.raw_lamb).item()
print(f'\n>>> Estimated parameters (inverse PINN):')
print(f'    beta        = {final_beta:.6f}')
print(f'    lambda      = {final_lamb:.6f}')
print(f'    Q1: w1={TF_func.softplus(model.raw_w1_q1).item():.4f}, w2={TF_func.softplus(model.raw_w2_q1).item():.4f}')
print(f'    Q2: w1={TF_func.softplus(model.raw_w1_q2).item():.4f}, w2={TF_func.softplus(model.raw_w2_q2).item():.4f}')
print(f'    Q3: w1={TF_func.softplus(model.raw_w1_q3).item():.4f}, w2={TF_func.softplus(model.raw_w2_q3).item():.4f}')
print(f'    Q4: w1={TF_func.softplus(model.raw_w1_q4).item():.4f}, w2={TF_func.softplus(model.raw_w2_q4).item():.4f}')

#------------------------ Evaluation -----------------------------
print("\nEvaluating on full dataset...")

T_pred, F_pred, pde_T, pde_F = model.predict(X_star)

# Compute errors
error_T = np.linalg.norm(T_all.reshape(-1, 1) - T_pred, 2) / np.linalg.norm(T_all.reshape(-1, 1), 2)
error_F = np.linalg.norm(F_all.reshape(-1, 1) - F_pred, 2) / np.linalg.norm(F_all.reshape(-1, 1), 2)

print(f'\nRelative L2 errors:')
print(f'  Temperature (non-dim): {error_T:.6e}')
print(f'  Fuel: {error_F:.6e}')

# Absolute errors
error_T_abs = np.abs(T_all.reshape(-1, 1) - T_pred)
error_F_abs = np.abs(F_all.reshape(-1, 1) - F_pred)

print(f'\nMean absolute errors:')
print(f'  Temperature: {np.mean(error_T_abs):.6e}')
print(f'  Fuel: {np.mean(error_F_abs):.6e}')

print(f'\nMax absolute errors:')
print(f'  Temperature: {np.max(error_T_abs):.6e}')
print(f'  Fuel: {np.max(error_F_abs):.6e}')

#------------------------ Visualization -----------------------------
print("\nCreating visualizations...")
os.makedirs("Fire_PINN_Troy/graphs_inverse", exist_ok=True)

# Get unique x, y, and t for reshaping
x_unique = np.unique(x_all)
y_unique = np.unique(y_all)
t_unique = np.unique(t_all)
nx = len(x_unique)
ny = len(y_unique)
nt = len(t_unique)

print(f"Grid: {nt} time steps x {nx} x {ny} space points")
print(f"Total unique grid points: {nt * nx * ny}, actual data points: {len(x_all)}")

# Map coordinates to grid indices and fill grids (handles non-contiguous or duplicate rows)
x_idx = np.searchsorted(x_unique, x_all)
y_idx = np.searchsorted(y_unique, y_all)
t_idx = np.searchsorted(t_unique, t_all)

T_true_grid = np.full((nt, ny, nx), np.nan)
F_true_grid = np.full((nt, ny, nx), np.nan)
T_pred_grid = np.full((nt, ny, nx), np.nan)
F_pred_grid = np.full((nt, ny, nx), np.nan)

T_true_grid[t_idx, y_idx, x_idx] = T_all
F_true_grid[t_idx, y_idx, x_idx] = F_all
T_pred_grid[t_idx, y_idx, x_idx] = T_pred.ravel()
F_pred_grid[t_idx, y_idx, x_idx] = F_pred.ravel()

# Select a few time snapshots to visualize
time_indices = [0, nt//4, nt//2, 3*nt//4, nt-1]
time_labels = [f't={t_unique[i]:.2f}' for i in time_indices]

# Create figure with snapshots at different times
fig, axes = plt.subplots(len(time_indices), 4, figsize=(20, 4*len(time_indices)))

for i, (tidx, tlabel) in enumerate(zip(time_indices, time_labels)):
    # Temperature - True
    im1 = axes[i, 0].imshow(T_true_grid[tidx], interpolation='nearest', cmap='hot',
                            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                            origin='lower', aspect='auto',
                            vmin=T_all.min(), vmax=T_all.max())
    axes[i, 0].set_title(f'Temperature True ({tlabel})', fontsize=12)
    axes[i, 0].set_xlabel('X', fontsize=10)
    axes[i, 0].set_ylabel('Y', fontsize=10)
    divider = make_axes_locatable(axes[i, 0])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im1, cax=cax)

    # Temperature - Predicted
    im2 = axes[i, 1].imshow(T_pred_grid[tidx], interpolation='nearest', cmap='hot',
                            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                            origin='lower', aspect='auto',
                            vmin=T_all.min(), vmax=T_all.max())
    axes[i, 1].set_title(f'Temperature PINN ({tlabel})', fontsize=12)
    axes[i, 1].set_xlabel('X', fontsize=10)
    axes[i, 1].set_ylabel('Y', fontsize=10)
    divider = make_axes_locatable(axes[i, 1])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im2, cax=cax)

    # Fuel - True
    im3 = axes[i, 2].imshow(F_true_grid[tidx], interpolation='nearest', cmap='viridis',
                            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                            origin='lower', aspect='auto',
                            vmin=F_all.min(), vmax=F_all.max())
    axes[i, 2].set_title(f'Fuel True ({tlabel})', fontsize=12)
    axes[i, 2].set_xlabel('X', fontsize=10)
    axes[i, 2].set_ylabel('Y', fontsize=10)
    divider = make_axes_locatable(axes[i, 2])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im3, cax=cax)

    # Fuel - Predicted
    im4 = axes[i, 3].imshow(F_pred_grid[tidx], interpolation='nearest', cmap='viridis',
                            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                            origin='lower', aspect='auto',
                            vmin=F_all.min(), vmax=F_all.max())
    axes[i, 3].set_title(f'Fuel PINN ({tlabel})', fontsize=12)
    axes[i, 3].set_xlabel('X', fontsize=10)
    axes[i, 3].set_ylabel('Y', fontsize=10)
    divider = make_axes_locatable(axes[i, 3])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im4, cax=cax)

plt.tight_layout()
plt.savefig('Fire_PINN_Troy/graphs_inverse/fire_pinn_2d_snapshots.png', dpi=150, bbox_inches='tight')
print("Snapshots saved to 'Fire_PINN_Troy/graphs_inverse/fire_pinn_2d_snapshots.png'")

# Create error visualization snapshots at different times
fig2, axes2 = plt.subplots(len(time_indices), 2, figsize=(14, 4*len(time_indices)))

for i, (tidx, tlabel) in enumerate(zip(time_indices, time_labels)):
    error_T_grid = T_true_grid[tidx] - T_pred_grid[tidx]
    error_F_grid = F_true_grid[tidx] - F_pred_grid[tidx]

    mse_T = np.mean(error_T_grid**2)
    mse_F = np.mean(error_F_grid**2)
    rel_T = np.linalg.norm(error_T_grid) / (np.linalg.norm(T_true_grid[tidx]) + 1e-10)
    rel_F = np.linalg.norm(error_F_grid) / (np.linalg.norm(F_true_grid[tidx]) + 1e-10)

    # Temperature Error
    im5 = axes2[i, 0].imshow(np.abs(error_T_grid), interpolation='nearest', cmap='Reds',
                             extent=[x_unique.min(), x_unique.max(),
                                     y_unique.min(), y_unique.max()],
                             origin='lower', aspect='auto')
    axes2[i, 0].set_title(f'Temperature Error ({tlabel})', fontsize=12)
    axes2[i, 0].set_xlabel('X', fontsize=10)
    axes2[i, 0].set_ylabel('Y', fontsize=10)
    axes2[i, 0].legend(handles=[
        Patch(fc='none', ec='none', label=f'Min:  {np.abs(error_T_grid).min():.4e}'),
        Patch(fc='none', ec='none', label=f'Max:  {np.abs(error_T_grid).max():.4e}'),
        Patch(fc='none', ec='none', label=f'Rel:  {rel_T:.4e}'),
        Patch(fc='none', ec='none', label=f'MSE: {mse_T:.4e}'),
    ], loc='upper right', fontsize=8, framealpha=0.8)
    divider = make_axes_locatable(axes2[i, 0])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im5, cax=cax)

    # Fuel Error
    im6 = axes2[i, 1].imshow(np.abs(error_F_grid), interpolation='nearest', cmap='Reds',
                             extent=[x_unique.min(), x_unique.max(),
                                     y_unique.min(), y_unique.max()],
                             origin='lower', aspect='auto')
    axes2[i, 1].set_title(f'Fuel Error ({tlabel})', fontsize=12)
    axes2[i, 1].set_xlabel('X', fontsize=10)
    axes2[i, 1].set_ylabel('Y', fontsize=10)
    axes2[i, 1].legend(handles=[
        Patch(fc='none', ec='none', label=f'Min:  {np.abs(error_F_grid).min():.4e}'),
        Patch(fc='none', ec='none', label=f'Max:  {np.abs(error_F_grid).max():.4e}'),
        Patch(fc='none', ec='none', label=f'Rel:  {rel_F:.4e}'),
        Patch(fc='none', ec='none', label=f'MSE: {mse_F:.4e}'),
    ], loc='upper right', fontsize=8, framealpha=0.8)
    divider = make_axes_locatable(axes2[i, 1])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im6, cax=cax)

plt.tight_layout()
plt.savefig('Fire_PINN_Troy/graphs_inverse/fire_pinn_2d_error_snapshots.png', dpi=150, bbox_inches='tight')
print("Error snapshots saved to 'Fire_PINN_Troy/graphs_inverse/fire_pinn_2d_error_snapshots.png'")

#------------------------ Loss History Plots -----------------------------
print("\nCreating loss history plots...")

# Create iteration array (losses are recorded every 20 iterations)
iterations = np.arange(20, len(model.loss_history) * 20 + 1, 20)

# Create loss plots
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

# Total Loss
axes4[0, 0].plot(iterations, model.loss_history, 'b-', linewidth=2)
axes4[0, 0].set_xlabel('Iteration', fontsize=12)
axes4[0, 0].set_ylabel('Total Loss', fontsize=12)
axes4[0, 0].set_title('Total Loss History', fontsize=14)
axes4[0, 0].set_yscale('log')
axes4[0, 0].grid(True, alpha=0.3)

# Data Loss vs PDE Loss
axes4[0, 1].plot(iterations, model.loss_data_history, 'r-', linewidth=2, label='Data Loss')
axes4[0, 1].plot(iterations, model.loss_pde_history, 'g-', linewidth=2, label='PDE Loss')
axes4[0, 1].set_xlabel('Iteration', fontsize=12)
axes4[0, 1].set_ylabel('Loss', fontsize=12)
axes4[0, 1].set_title('Data Loss vs PDE Loss', fontsize=14)
axes4[0, 1].set_yscale('log')
axes4[0, 1].legend(fontsize=11)
axes4[0, 1].grid(True, alpha=0.3)

# Temperature Losses
axes4[1, 0].plot(iterations, model.loss_T_data_history, 'r-', linewidth=2, label='T Data Loss')
axes4[1, 0].plot(iterations, model.loss_T_pde_history, 'g-', linewidth=2, label='T PDE Loss')
axes4[1, 0].set_xlabel('Iteration', fontsize=12)
axes4[1, 0].set_ylabel('Loss', fontsize=12)
axes4[1, 0].set_title('Temperature Losses', fontsize=14)
axes4[1, 0].set_yscale('log')
axes4[1, 0].legend(fontsize=11)
axes4[1, 0].grid(True, alpha=0.3)

# Fuel Losses
axes4[1, 1].plot(iterations, model.loss_F_data_history, 'r-', linewidth=2, label='F Data Loss')
axes4[1, 1].plot(iterations, model.loss_F_pde_history, 'g-', linewidth=2, label='F PDE Loss')
axes4[1, 1].set_xlabel('Iteration', fontsize=12)
axes4[1, 1].set_ylabel('Loss', fontsize=12)
axes4[1, 1].set_title('Fuel Losses', fontsize=14)
axes4[1, 1].set_yscale('log')
axes4[1, 1].legend(fontsize=11)
axes4[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Fire_PINN_Troy/graphs_inverse/fire_pinn_lbfgs_loss_history.png', dpi=150, bbox_inches='tight')
print("Loss history saved to 'Fire_PINN_Troy/graphs_inverse/fire_pinn_lbfgs_loss_history.png'")

# Save model
torch.save(model.dnn.state_dict(), 'Fire_PINN_Troy/data/fire_pinn_inverse_model.pth')
print("Model saved to 'Fire_PINN_Troy/data/fire_pinn_inverse_model.pth'")

plt.show()

#------------------------- Inverse Value -----------------------------

model.plot_beta_lambda_convergence()

model.plot_w_convergence()

print("\nDone!")


