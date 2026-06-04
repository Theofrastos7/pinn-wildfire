"""
Fire PINN - Physics-Informed Neural Network for fire propagation (2D)

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
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

np.random.seed(1234)
torch.manual_seed(1234)


max_iter = 15000
w_d = 1.0  # Data loss
w_p = 1.0  # Physics loss
w_i = 0.0  # Initial conditions. Enable for great number of data points
learning_rate = 0.6 # learning rate for LBFGS
data_points = 10000  # Data points for data
collocation_points = 300  # Collocation points for physics
dataset = 'Fire_PINN_Multi_variable/data/fire_dataset_columns.csv'

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
        TF_data: known T and F values at X_data (N_data x 2) [T, F]
        X_collocation: collocation points for PDE residual (N_collocation x 3)
        X_ic: initial condition points at t=0 (N_ic x 3) [x, y, 0]
        TF_ic: initial T and F values at t=0 (N_ic x 2) [T, F]
        layers: network architecture
        lb, ub: domain bounds
        fire_params: dict with Ta, w1, w2, lamb1, beta1, lamb2, beta2, lamb3, beta3, lamb4, beta4
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

        # Learnable parameters beta1-4, lambda1-4, w1, w2
        self.lamb1 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.beta1 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.lamb2 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.beta2 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.lamb3 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.beta3 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.lamb4 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.beta4 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))

        self.w1 = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self.w2 = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))

        # History tracking for each learnable parameter
        self.lamb1_history = []
        self.beta1_history = []
        self.lamb2_history = []
        self.beta2_history = []
        self.lamb3_history = []
        self.beta3_history = []
        self.lamb4_history = []
        self.beta4_history = []
        self.w1s = []
        self.w2s = []

        self.Ta = fire_params['Ta']
        self.x_mid = fire_params['x_mid']
        self.y_mid = fire_params['y_mid']
        
        # Network
        self.layers = layers
        self.dnn = DNN(layers).to(device)
        
        # Optimizer LBFGS
        # Add the learnable parameters to the optimizer as network parameters
        all_params = list(self.dnn.parameters()) + [
            self.lamb1, self.beta1, self.lamb2, self.beta2,
            self.lamb3, self.beta3, self.lamb4, self.beta4,
            self.w1, self.w2
        ]
        self.optimizer = torch.optim.LBFGS(
            all_params, 
            lr= learning_rate, 
            max_iter= max_iter, 
            max_eval= max_iter,  
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
            create_graph=True        )[0]
        
        # T_yy = ∂²T/∂y²
        T_yy = torch.autograd.grad(
            T_y, y, 
            grad_outputs=torch.ones_like(T_y),
            retain_graph=True,
            create_graph=True
        )[0]          
        
        # Fire physics terms
        T_safe = torch.clamp(T, min=0.1) # Avoid small T for exp(-1/T)
        reaction_term = F * torch.exp(-1.0 / T_safe) # reaction term = F*exp(-1/T)
        
        # Determine position for each collocation point
        is_x_high = x >= self.x_mid  # x >= mid
        is_y_high = y >= self.y_mid  # y >= mid

        # Select correct lamb and beta based on quadrant of collocation point
        lamb_spatial = torch.where(
            is_x_high,
            torch.where(is_y_high, self.lamb1, self.lamb4),  
            torch.where(is_y_high, self.lamb2, self.lamb3)   
        )
        # Select correct beta
        beta_spatial = torch.where(
            is_x_high,
            torch.where(is_y_high, self.beta1, self.beta4),  
            torch.where(is_y_high, self.beta2, self.beta3)   
        )
        
        cooling_term = lamb_spatial * T # cooling term lamb*T
        fuel_consumption = beta_spatial * reaction_term # fuel consumption term = beta*F*exp(-1/T)
        
        # PDE residuals 
        # Temperature: ∂T/∂t = ∂²T/∂x² + ∂²T/∂y² + F*exp(-1/T) - lamb*T
        # Fuel: ∂F/∂t = -beta*F*exp(-1/T)
        pde_T = T_t - self.w1 * T_xx - self.w2 * T_yy - reaction_term + cooling_term
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
            loss_T_ic = torch.mean((self.T_ic - T_ic_pred) ** 2)
            loss_F_ic = torch.mean((self.F_ic - F_ic_pred) ** 2)
            loss_ic = loss_T_ic + loss_F_ic
        else:
            loss_T_ic = torch.tensor(0.0).to(device)
            loss_F_ic = torch.tensor(0.0).to(device)
            loss_ic = torch.tensor(0.0).to(device)
        
        w_data = w_d
        w_pde = w_p
        w_ic = w_i 

        # Total loss = data loss + PDE loss + initial condition loss
        loss = w_data * loss_data + w_pde * loss_pde + w_ic * loss_ic
        loss.backward()


        # Track all lambs and betas in history
        self.lamb1_history.append(self.lamb1.item())
        self.beta1_history.append(self.beta1.item())
        self.lamb2_history.append(self.lamb2.item())
        self.beta2_history.append(self.beta2.item())
        self.lamb3_history.append(self.lamb3.item())
        self.beta3_history.append(self.beta3.item())
        self.lamb4_history.append(self.lamb4.item())
        self.beta4_history.append(self.beta4.item())

        self.w1s.append(self.w1.item())
        self.w2s.append(self.w2.item())
        
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
                'Iter %d, Loss: %.5e, Data: %.5e (T:%.5e, F:%.5e), PDE: %.5e (T:%.5e, F:%.5e), IC: %.5e (T:%.5e, F:%.5e)' 
                % (self.iter, loss.item(), loss_data.item(), loss_T_data.item(), loss_F_data.item(),
                   loss_pde.item(), loss_T_pde.item(), loss_F_pde.item(),
                   loss_ic.item(), loss_T_ic.item(), loss_F_ic.item())
            )
            print(
                '  λ1=%.4f, β1=%.4f, λ2=%.4f, β2=%.4f, λ3=%.4f, β3=%.4f, λ4=%.4f, β4=%.4f'
                % (self.lamb1.item(), self.beta1.item(), self.lamb2.item(), self.beta2.item(),
                   self.lamb3.item(), self.beta3.item(), self.lamb4.item(), self.beta4.item())
            )
        
        return loss
    
    def train(self, max_iterations = max_iter):
        """
        Train the network for a max_iter number of iterations
        """
        self.dnn.train()
        
        #ensure LBFGS runs for max_iterations
        while self.iter < max_iterations:
            def closure():
                return self.loss_func()
            
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
            
            x = torch.tensor(X_batch[:, 0:1], requires_grad=True).float().to(device)
            y = torch.tensor(X_batch[:, 1:2], requires_grad=True).float().to(device)
            t = torch.tensor(X_batch[:, 2:3], requires_grad=True).float().to(device)
            
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

    def plot_parameter_convergence(self):
        """
        "Plot the convergence of all 8 parameters during training"
        """
        
        # Get true values from fire_params
        true_lamb1 = fire_params['lamb1']
        true_beta1 = fire_params['beta1']
        true_lamb2 = fire_params['lamb2']
        true_beta2 = fire_params['beta2']
        true_lamb3 = fire_params['lamb3']
        true_beta3 = fire_params['beta3']
        true_lamb4 = fire_params['lamb4']
        true_beta4 = fire_params['beta4']

        quadrant_names = ['Q1 top-right', 'Q2 top-left', 'Q3 bottom-left', 'Q4 bottom-right']
        lamb_histories = [self.lamb1_history, self.lamb2_history, self.lamb3_history, self.lamb4_history]
        beta_histories = [self.beta1_history, self.beta2_history, self.beta3_history, self.beta4_history]
        true_lambs = [true_lamb1, true_lamb2, true_lamb3, true_lamb4]
        true_betas = [true_beta1, true_beta2, true_beta3, true_beta4]


        # Q1:top-right, Q2:top-left, Q3:bottom-left, Q4:bottom-right
        positions = [(0, 1), (0, 0), (1, 0), (1, 1)]

        # Lambda convergence
        fig_lamb, axes_lamb = plt.subplots(2, 2, figsize=(14, 10))
        for idx, (name, pos) in enumerate(zip(quadrant_names, positions)):
            ax = axes_lamb[pos[0], pos[1]]
            final_lamb = lamb_histories[idx][-1] if lamb_histories[idx] else 0.0
            lamb_err = abs(final_lamb - true_lambs[idx]) / true_lambs[idx] * 100 if true_lambs[idx] != 0 else 0
            ax.plot(lamb_histories[idx], 'b-', linewidth=2, label=f'Est. λ{idx+1} = {final_lamb:.4f}')
            ax.axhline(y=true_lambs[idx], color='b', linestyle='--', linewidth=1.5,
                       label=f'True λ{idx+1}={true_lambs[idx]:.4f} | Rel. error: {lamb_err:.2f}%')
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Lambda Value', fontsize=11)
            ax.set_title(f'λ Convergence - {name}', fontsize=12)
            ax.set_ylim(0, 0.1)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        fig_lamb.tight_layout()
        fig_lamb.savefig('Fire_PINN_multi_variable/graphs/lambda_convergence_4quadrants.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Beta convergence
        fig_beta, axes_beta = plt.subplots(2, 2, figsize=(14, 10))
        for idx, (name, pos) in enumerate(zip(quadrant_names, positions)):
            ax = axes_beta[pos[0], pos[1]]
            final_beta = beta_histories[idx][-1] if beta_histories[idx] else 0.0
            beta_err = abs(final_beta - true_betas[idx]) / true_betas[idx] * 100 if true_betas[idx] != 0 else 0
            ax.plot(beta_histories[idx], 'r-', linewidth=2, label=f'Est. β{idx+1} = {final_beta:.4f}')
            ax.axhline(y=true_betas[idx], color='r', linestyle='--', linewidth=1.5,
                       label=f'True β{idx+1}={true_betas[idx]:.4f} | Rel. error: {beta_err:.2f}%')
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Beta Value', fontsize=11)
            ax.set_title(f'β Convergence - {name}', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        fig_beta.tight_layout()
        fig_beta.savefig('Fire_PINN_multi_variable/graphs/beta_convergence_4quadrants.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n" + "="*60)
        print("PARAMETER ESTIMATION RESULTS 4 Quadrants")
        print("="*60)
        
        for idx, name in enumerate(quadrant_names):
            final_lamb = lamb_histories[idx][-1] if lamb_histories[idx] else 0.0
            final_beta = beta_histories[idx][-1] if beta_histories[idx] else 0.0
            lamb_error = abs(final_lamb - true_lambs[idx]) / true_lambs[idx] * 100 if true_lambs[idx] != 0 else 0
            beta_error = abs(final_beta - true_betas[idx]) / true_betas[idx] * 100 if true_betas[idx] != 0 else 0
            
            print(f"\n{name}:")
            print(f"  λ{idx+1}: True={true_lambs[idx]:.6f}, Est={final_lamb:.6f}, Error={lamb_error:.2f}%")
            print(f"  β{idx+1}: True={true_betas[idx]:.6f}, Est={final_beta:.6f}, Error={beta_error:.2f}%")

    
    def plot_w_convergence(self):
        """
        Plot the convergence of diffusion weights w1 and w2 during training
        """

        real_w1 = fire_params.get('w1', 1.0)
        real_w2 = fire_params.get('w2', 1.0)

        final_w1 = self.w1s[-1] if self.w1s else 0.0
        error_percent_w1 = abs(final_w1 - real_w1) / real_w1 * 100
        print(f"\nw1 Estimation Results:")
        print(f"True value: {real_w1:.4f}")
        print(f"Estimated: {final_w1:.4f}")
        print(f"Error: {error_percent_w1:.2f}%")

        final_w2 = self.w2s[-1] if self.w2s else 0.0
        error_percent_w2 = abs(final_w2 - real_w2) / real_w2 * 100
        print(f"\nw2 Estimation Results:")
        print(f"True value: {real_w2:.4f}")
        print(f"Estimated: {final_w2:.4f}")
        print(f"Error: {error_percent_w2:.2f}%")

        _, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(self.w1s, 'b-', linewidth=2, label=f'Estimated w1 = {final_w1:.4f}')
        axes[0].axhline(y=real_w1, color='r', linestyle='--', linewidth=2,
                        label=f'True w1 = {real_w1:.4f} | Rel. error: {error_percent_w1:.2f}%')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('w1 (x-diffusion)')
        axes[0].set_title('Diffusion Weight Parameters Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.w2s, 'g-', linewidth=2, label=f'Estimated w2 = {final_w2:.4f}')
        axes[1].axhline(y=real_w2, color='r', linestyle='--', linewidth=2,
                        label=f'True w2 = {real_w2:.4f} | Rel. error: {error_percent_w2:.2f}%')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('w2 (y-diffusion)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('Fire_PINN_Multi_variable/graphs/w1_w2_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()


def _fmt(v, sig=3):
    """Format a float as  m.mm × 10ⁿ  (3 sig. figs by default)."""
    if v == 0:
        return "0"
    import math
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
# Fire parameters for 4 quadrants from dataset

beta1 = data_df['beta1'].iloc[0]
lamb1 = data_df['lamb1'].iloc[0]
beta2 = data_df['beta2'].iloc[0]
lamb2 = data_df['lamb2'].iloc[0]
beta3 = data_df['beta3'].iloc[0]
lamb3 = data_df['lamb3'].iloc[0]
beta4 = data_df['beta4'].iloc[0]
lamb4 = data_df['lamb4'].iloc[0]
Ta = data_df['Ta'].iloc[0]
w1 = data_df['w1'].iloc[0]
w2 = data_df['w2'].iloc[0]

fire_params = {
    'lamb1': lamb1, 'beta1': beta1,
    'lamb2': lamb2, 'beta2': beta2,
    'lamb3': lamb3, 'beta3': beta3,
    'lamb4': lamb4, 'beta4': beta4,
    'w1': w1,
    'w2': w2,
    'Ta': Ta,
    'x_mid': (ub[0] + lb[0]) / 2,  # midpoint for left/right split
    'y_mid': (ub[1] + lb[1]) / 2   # midpoint for top/bottom split
}

print(f"\nFire parameters:")
print(f"Q1: lamb1={lamb1:.6f}, beta1={beta1:.6f}")
print(f"Q2: lamb2={lamb2:.6f}, beta2={beta2:.6f}")
print(f"Q3: lamb3={lamb3:.6f}, beta3={beta3:.6f}")
print(f"Q4: lamb4={lamb4:.6f}, beta4={beta4:.6f}")
print(f"w1={w1}, w2={w2}")
print(f"Ta={Ta}")

# Training configuration
N_data = data_points      # Number of data points to use for training
N_collocation = collocation_points  # Number of collocation points for PDE residual

# Network architecture
# input=3 (x,y,t), output=2 (T,F), 4 hidden layers of 20 neurons

layers = [3, 20, 20, 20, 20, 2] 

print(f"\nTraining configuration:")
print(f"  N_data = {N_data}")
print(f"  N_collocation = {N_collocation}")
print(f"  Network layers = {layers}")

# Randomly sample training data points with same seed
idx_data = np.random.choice(X_star.shape[0], N_data, replace=False)
X_data = X_star[idx_data, :]
TF_data = TF_star[idx_data, :]

# Randomly sample collocation points with same seed
idx_col = np.random.choice(X_star.shape[0], N_collocation, replace=False)
X_collocation = X_star[idx_col, :]

# Extract initial condition points (t=0)
ic_mask = (t_all == 0.0)
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

#------------------------ Evaluation -----------------------------
print("\nEvaluating on full dataset...")

T_pred, F_pred, pde_T, pde_F = model.predict(X_star)

# Compute errors
error_T = np.linalg.norm(T_all.reshape(-1, 1) - T_pred, 2) / np.linalg.norm(T_all.reshape(-1, 1), 2)
error_F = np.linalg.norm(F_all.reshape(-1, 1) - F_pred, 2) / np.linalg.norm(F_all.reshape(-1, 1), 2)

print(f'\nRelative L2 errors:')
print(f'  Temperature: {error_T:.6e}')
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

# Get unique x, y, and t for reshaping
x_unique = np.unique(x_all)
y_unique = np.unique(y_all)
t_unique = np.unique(t_all)
nx = len(x_unique)
ny = len(y_unique)
nt = len(t_unique)

print(f"Grid: {nt} time steps x {nx} x {ny} space points")

# Reshape for plotting (nt, nx, ny)
T_true_grid = T_all.reshape(nt, nx, ny)
F_true_grid = F_all.reshape(nt, nx, ny)
T_pred_grid = T_pred.reshape(nt, nx, ny)
F_pred_grid = F_pred.reshape(nt, nx, ny)

# Select time snapshots to visualize
time_indices = [0, nt//4, nt//2, 3*nt//4, nt-1]
time_labels = [f't={t_unique[i]:.2f}' for i in time_indices]

# Create figure with snapshots
fig, axes = plt.subplots(len(time_indices), 4, figsize=(20, 4*len(time_indices)))

for i, (tidx, tlabel) in enumerate(zip(time_indices, time_labels)):
    # True Temperature
    im1 = axes[i, 0].imshow(T_true_grid[tidx, :, :].T, interpolation='nearest', cmap='hot',
                            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                            origin='lower', aspect='auto',
                            vmin=0, vmax=1.75)
    axes[i, 0].set_title(f'Temperature True ({tlabel})', fontsize=12)
    axes[i, 0].set_xlabel('X', fontsize=10)
    axes[i, 0].set_ylabel('Y', fontsize=10)
    divider = make_axes_locatable(axes[i, 0])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im1, cax=cax)

    # Predicted Temperature
    im2 = axes[i, 1].imshow(T_pred_grid[tidx, :, :].T, interpolation='nearest', cmap='hot',
                            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                            origin='lower', aspect='auto',
                            vmin=0, vmax=1.75)
    axes[i, 1].set_title(f'Temperature PINN ({tlabel})', fontsize=12)
    axes[i, 1].set_xlabel('X', fontsize=10)
    axes[i, 1].set_ylabel('Y', fontsize=10)
    divider = make_axes_locatable(axes[i, 1])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im2, cax=cax)

    # True Fuel
    im3 = axes[i, 2].imshow(F_true_grid[tidx, :, :].T, interpolation='nearest', cmap='viridis',
                            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                            origin='lower', aspect='auto',
                            vmin=0, vmax=1.0)
    axes[i, 2].set_title(f'Fuel True ({tlabel})', fontsize=12)
    axes[i, 2].set_xlabel('X', fontsize=10)
    axes[i, 2].set_ylabel('Y', fontsize=10)
    divider = make_axes_locatable(axes[i, 2])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im3, cax=cax)

    # Predicted Fuel
    im4 = axes[i, 3].imshow(F_pred_grid[tidx, :, :].T, interpolation='nearest', cmap='viridis',
                            extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                            origin='lower', aspect='auto',
                            vmin=0, vmax=1.0)
    axes[i, 3].set_title(f'Fuel PINN ({tlabel})', fontsize=12)
    axes[i, 3].set_xlabel('X', fontsize=10)
    axes[i, 3].set_ylabel('Y', fontsize=10)
    divider = make_axes_locatable(axes[i, 3])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im4, cax=cax)

plt.tight_layout()
plt.savefig('Fire_PINN_Multi_variable/graphs/inverse_fire_pinn_2d_snapshots.png', dpi=150, bbox_inches='tight')
print("Snapshots saved to 'inverse_fire_pinn_2d_snapshots.png'")

# Create error visualization snapshots 
fig2, axes2 = plt.subplots(len(time_indices), 2, figsize=(14, 4*len(time_indices)))

for i, (tidx, tlabel) in enumerate(zip(time_indices, time_labels)):
    error_T_grid = T_true_grid[tidx, :, :] - T_pred_grid[tidx, :, :]
    error_F_grid = F_true_grid[tidx, :, :] - F_pred_grid[tidx, :, :]

    mse_T = np.mean(error_T_grid**2)
    mse_F = np.mean(error_F_grid**2)
    rel_T = np.linalg.norm(error_T_grid) / (np.linalg.norm(T_true_grid[tidx, :, :]) + 1e-10)
    rel_F = np.linalg.norm(error_F_grid) / (np.linalg.norm(F_true_grid[tidx, :, :]) + 1e-10)

    # Temperature Error
    im5 = axes2[i, 0].imshow(np.abs(error_T_grid).T, interpolation='nearest', cmap='Reds',
                             extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                             origin='lower', aspect='auto')
    axes2[i, 0].set_title(f'Temperature Error ({tlabel})', fontsize=12)
    axes2[i, 0].set_xlabel('X', fontsize=10)
    axes2[i, 0].set_ylabel('Y', fontsize=10)
    axes2[i, 0].legend(handles=[
        Patch(fc='none', ec='none', label=f'Min:  {_fmt(np.abs(error_T_grid).min())}'),
        Patch(fc='none', ec='none', label=f'Max:  {_fmt(np.abs(error_T_grid).max())}'),
        Patch(fc='none', ec='none', label=f'Rel:  {rel_T*100:.2f}%'),
        Patch(fc='none', ec='none', label=f'MSE: {_fmt(mse_T)}'),
    ], loc='upper right', fontsize=8, framealpha=0.8)
    divider = make_axes_locatable(axes2[i, 0])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im5, cax=cax)

    # Fuel Error
    im6 = axes2[i, 1].imshow(np.abs(error_F_grid).T, interpolation='nearest', cmap='Reds',
                             extent=[x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()],
                             origin='lower', aspect='auto')
    axes2[i, 1].set_title(f'Fuel Error ({tlabel})', fontsize=12)
    axes2[i, 1].set_xlabel('X', fontsize=10)
    axes2[i, 1].set_ylabel('Y', fontsize=10)
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
plt.savefig('Fire_PINN_Multi_variable/graphs/inverse_fire_pinn_2d_error_snapshots.png', dpi=150, bbox_inches='tight')
print("Error snapshots saved to 'inverse_fire_pinn_2d_error_snapshots.png'")

#------------------------ Loss History Plots -----------------------------
print("\nCreating loss history plots...")


# Create iteration array , losses are recorded every 20 iterations
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
plt.savefig('Fire_PINN_Multi_variable/graphs/inverse_fire_pinn_lbfgs_loss_history.png', dpi=150, bbox_inches='tight')
print("Loss history saved to 'inverse_fire_pinn_lbfgs_loss_history.png'")

# Save model
torch.save(model.dnn.state_dict(), 'Fire_PINN_Multi_variable/data/inverse_fire_pinn_lbfgs_model.pth')
print("Model saved to 'inverse_fire_pinn_lbfgs_model.pth'")

plt.show()

#------------------------- Inverse Value -----------------------------

model.plot_parameter_convergence()

model.plot_w_convergence()

print("\nDone!")
