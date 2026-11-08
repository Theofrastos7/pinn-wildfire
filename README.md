# Physics-Informed Neural Networks for Wildfire Spread Modeling

Diploma thesis — Computer Engineering & Informatics Department (CEID), University of Patras.

Applies Physics-Informed Neural Networks (PINNs) to the non-dimensional wildfire model of Mandel et al. (2008), solving both the **forward problem** (predicting temperature and fuel fields) and the **inverse problem** (estimating PDE parameters from data). The framework is validated on synthetic benchmarks with spatially heterogeneous coefficients and applied to real thermal imagery from the 2003 Troy Fire.

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
│   └── main_2D.py                # Entry point
│
├── Fire_PINN_Multi_variable/     # Multi-Variable PINN (per-region λ_i, β_i)
│   ├── create_fire_dataset_multi_variable.py
│   ├── pinn_multi_variable.py
│   └── pinn_inverse_multi_variable.py
│
├── Fire_PINN_Multi_diffusion/    # Multi-Diffusion PINN (per-region w1_i, w2_i)
│   ├── create_fire_dataset_multi_diffusion.py
│   ├── pinn_multi_diffusion.py
│   └── pinn_inverse_multi_diffusion.py
│
├── Fire_PINN_Multi_variable_diffusion/   # Combined (per-region λ_i, β_i, w1_i, w2_i)
│   ├── create_fire_dataset_multi_var_dif.py
│   ├── pinn__multi_var_dif.py
│   └── pinn_inverse_multi_var_dif.py
│
├── Fire_PINN_Troy/               # Troy Fire real-data experiments
│   ├── pinn_predict.py           # Classic NN vs PINN prediction
│   ├── xpinn_predict.py          # XPINN with domain decomposition
│   ├── causal_xpinn_predict.py   # Causal XPINN (C-XPINN)
│   ├── alessandri_comparison.py  # Comparison with level-set method
│   ├── make_prediction_figure.py # Generate prediction comparison figures
│   └── make_xpinn_comparison_figure.py
│
├── extras/                       # Noise robustness study and seed experiments
│   ├── run_noise_study_avg.py    # 5-seed noise robustness experiments
│   ├── replot_noise.py           # Regenerate noise robustness plot
│   └── run_summary_table.py      # Multi-seed summary table
│
├── Latex/                        # Paper source (LaTeX)
│   └── main.tex
│
└── troy_fire_area_data/          # Troy Fire geospatial and wind data
```

---

## The Model

The PDE system (non-dimensional form, Mandel et al. 2008):

$$\frac{\partial \tilde{T}}{\partial \tilde{t}} = w_1\frac{\partial^2 \tilde{T}}{\partial \tilde{x}^2} + w_2\frac{\partial^2 \tilde{T}}{\partial \tilde{y}^2} + \tilde{S}\,e^{-1/\tilde{T}} - \lambda\,\tilde{T}$$

$$\frac{\partial \tilde{S}}{\partial \tilde{t}} = -\beta\,\tilde{S}\,e^{-1/\tilde{T}}$$

where:
- `T̃` — non-dimensional temperature
- `S̃` — non-dimensional fuel fraction
- `λ` — heat loss coefficient
- `β` — fuel consumption rate
- `w1, w2` — anisotropic diffusion weights

Three heterogeneous formulations are studied:
- **MV (Multi-Variable)** — per-region reaction coefficients (λ_i, β_i), global diffusion
- **MD (Multi-Diffusion)** — global reaction, per-region diffusion weights (w1_i, w2_i)
- **MVD (Multi-Variable-Diffusion)** — all coefficients vary per region (16 unknowns)

For the Troy Fire, the framework extends to:
- **XPINN** — domain decomposition into untouched/burning/burnt regions with separate sub-networks
- **C-XPINN** — causal training that enforces temporal ordering in the burning sub-PINN

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

### 4. Troy Fire experiments

```bash
cd Fire_PINN_Troy
python pinn_predict.py            # Classic NN vs PINN
python xpinn_predict.py           # XPINN
python causal_xpinn_predict.py    # C-XPINN
python alessandri_comparison.py   # Level-set comparison
```

---

## Results

All training graphs (loss history, field snapshots, parameter convergence) are saved automatically in each module's `graphs/` folder.

### Synthetic benchmarks (inverse problem)

| Variant | Reaction error (λ, β) | Diffusion error (w) |
|---------|----------------------|---------------------|
| MV      | 0.1% – 5.2%         | 5.5% – 10.0%       |
| MD      | 1.5% – 3.4%         | 1.9% – 26.9%       |
| MVD     | 0.3% – 4.2%         | 2.1% – 21.1%       |

### Troy Fire (prediction on unseen frame at 15:08)

| Model   | Temp L2 error | Fuel L2 error |
|---------|--------------|--------------|
| Classic NN | 1.05        | 0.26         |
| PINN    | 7.63e-1      | 2.23e-1      |
| XPINN   | 6.23e-1      | 1.46e-1      |
| C-XPINN | 5.37e-1      | 1.54e-1      |

---

## Reference

> Mandel, J., Beezley, J.D., Kochanski, A.K. (2008). *A wildland fire model with data assimilation*. Mathematics and Computers in Simulation.

---

## Author

Theofrastos Paximadis — University of Patras, CEID, 2025
Supervisors: K. Tsichlas, C. Konstantopoulos, A. Kalogeropulos
