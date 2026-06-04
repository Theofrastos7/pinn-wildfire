# Physics-Informed Neural Networks for Wildfire Spread Modeling

Diploma thesis — Computer Engineering & Informatics Department (CEID), University of Patras.

Applies Physics-Informed Neural Networks (PINNs) to the non-dimensional wildfire model of Mandel et al. (2008), solving both the **forward problem** (predicting temperature and fuel fields) and the **inverse problem** (estimating PDE parameters from data).

---

## Repository Structure

```
.
├── Dimensinal_2D_model/          # Numerical solver (operator splitting, finite differences)
│   ├── main.py                   # Run the 2D fire simulator
│   └── functions.py              # Solver implementation
│
├── Fire_PINN_2D/                 # Base PINN (uniform λ, β, w1, w2)
│   ├── create_fire_dataset_2D.py # Generate synthetic dataset from the solver
│   ├── pinn_2D.py                # Forward PINN training & evaluation
│   ├── pinn_inverse_2D.py        # Inverse PINN (estimates λ, β, w1, w2)
│   └── main_2D.py                # 3D fire simulation results
│
├── Fire_PINN_Multi_variable/     # Multi-Variable PINN (per-quadrant λ_i, β_i)
│   ├── create_fire_dataset_multi_variable.py
│   ├── pinn_multi_variable.py
│   └── pinn_inverse_multi_variable.py
│
├── Fire_PINN_Multi_diffusion/    # Multi-Diffusion PINN (per-quadrant w1_i, w2_i)
│   ├── create_fire_dataset_multi_diffusion.py
│   ├── pinn_multi_diffusion.py
│   └── pinn_inverse_multi_diffusion.py
│
└── Fire_PINN_Multi_variable_diffusion/   # Combined (per-quadrant λ_i, β_i, w1_i, w2_i)
    ├── create_fire_dataset_multi_var_dif.py
    ├── pinn__multi_var_dif.py
    └── pinn_inverse_multi_var_dif.py
```

---

## The Model

The PDE system derived from the non-dimensional form of Mandel et al. 2008:

$$\frac{\partial \tilde{T}}{\partial \tilde{t}} = w_1\frac{\partial^2 \tilde{T}}{\partial \tilde{x}^2} + w_2\frac{\partial^2 \tilde{T}}{\partial \tilde{y}^2} + \tilde{S}\,e^{-1/\tilde{T}} - \lambda\,\tilde{T}$$

$$\frac{\partial \tilde{S}}{\partial \tilde{t}} = -\beta\,\tilde{S}\,e^{-1/\tilde{T}}$$

where:
- `T̃` — non-dimensional temperature
- `S̃` — non-dimensional fuel fraction
- `λ` — heat loss coefficient
- `β` — fuel consumption rate
- `w1, w2` — anisotropic diffusion weights

The PINN receives `(x, y, t)` as inputs and outputs `(T̃, S̃)`. Derivatives are computed via automatic differentiation (PyTorch autograd) and the PDE residuals are included in the loss function.

---

## Requirements

| Package      | Version used |
|-------------|-------------|
| Python      | 3.12.8      |
| PyTorch     | 2.11.0+cpu  |
| NumPy       | 2.4.4       |
| Matplotlib  | 3.10.9      |
| Pandas      | 3.0.2       |

Install with:

```bash
pip install torch numpy matplotlib pandas
```

> GPU is not required. All experiments were run on CPU using the L-BFGS optimizer.

---

## Usage

### 1. Generate the synthetic dataset

```bash
cd Fire_PINN_2D
python create_fire_dataset_2D.py
```

This runs the finite-difference solver and saves the dataset to `data/`.

### 2. Train the Forward PINN

```bash
python pinn_2D.py
```

### 3. Train the Inverse PINN

```bash
python pinn_inverse_2D.py
```

The same pattern applies for the other three variants (`Multi_variable`, `Multi_diffusion`, `Multi_variable_diffusion`).

---

## Results

All training graphs (loss history, field snapshots, parameter convergence) are saved automatically in each module's `graphs/` folder.

| Variant                  | Temp L2 error | Fuel L2 error |
|--------------------------|--------------|--------------|
| Base                     | ~2.90%       | ~0.85%       |
| Multi-Variable           | ~2.45%       | ~0.74%       |
| Multi-Diffusion          | ~3.8%        | ~0.9%        |
| Multi-Variable+Diffusion | ~3.2%        | ~0.8%        |

Inverse problem: `λ` and `β` recovered with ~1–6% relative error across all variants.

---

## Reference

> Mandel, J., Beezley, J.D., Kochanski, A.K. (2008). *A wildland fire model with data assimilation*. Mathematics and Computers in Simulation.

---

## Author

Theofrastos Paximadis — University of Patras, CEID, 2026  
Supervisors: K. Tsichlas, C. Konstantopoulos, A. Kalogeropulos
