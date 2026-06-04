import os
import pandas as pd
import numpy as np
from functions_multi_diffusion import Diffusion_Reaction_Windless_Operator_Splitting

def create_fire_dataset():
    """
    Generate fire propagation dataset
    - x: spatial array
    - y: spatial array
    - t: time array
    - T_sol: temperature solution array (space x time)
    - F_sol: fuel solution array (space x time)
    - params: dictionary of parameters used in the simulation
    (Ta, lamb, beta, w1_q1, w2_q1, w1_q2, w2_q2, w1_q3, w2_q3, w1_q4, w2_q4)
    """

    #Parameters changed to fit a non-dimensional model

    # fixed constants
    k, A, C = 0.28, 187, 0.000056
    B = 558.49
    CS = 0.1625
    Ta = 0

    # Initial area of ignition
    start_x= 50
    start_y=50
    end_x= 25
    end_y= 25

    # Total time and time step
    dt=0.5
    T_total=300
    original_dt = dt

    # Spatial discretization
    dx=2
    dy=2
    Nx = 125
    Ny = 125

    # lamb is the cooling parameter, beta is the fuel consumption rate
    lamb=C*B
    beta=B*CS/A

    # Define 4 quadrants with different dispersion weights
    # Q1 top-right:    x >= mid, y >= mid = w1_q1, w2_q1
    # Q2 top-left:     x < mid,  y >= mid = w1_q2, w2_q2
    # Q3 bottom-left:  x < mid,  y < mid  = w1_q3, w2_q3
    # Q4 bottom-right: x >= mid, y < mid  = w1_q4, w2_q4

    w1_q1 = 1
    w2_q1 = 1

    w1_q2 = 2
    w2_q2 = 2

    w1_q3 = 3
    w2_q3 = 3

    w1_q4 = 4
    w2_q4 = 4

    # Fire at ignition
    temp1= (1160 - Ta) / B
    temp2= (1260 - Ta) / B


    # Fuel at ignition
    fuel1 = 0.45
    fuel2 = 0.55

    # Time for non-dimensional model
    dt=(A/B)*dt
    T_total=(A/B)*(T_total)

    # Space for non-dimensional model
    dx= np.sqrt(A)*dx/np.sqrt(B*k)
    Nx= np.sqrt(A)*Nx/np.sqrt(B*k)
    start_x= np.sqrt(A)*start_x/np.sqrt(B*k)
    end_x= np.sqrt(A)*end_x/np.sqrt(B*k)
    dy= np.sqrt(A)*dy/np.sqrt(B*k)
    Ny= np.sqrt(A)*Ny/np.sqrt(B*k)
    start_y= np.sqrt(A)*start_y/np.sqrt(B*k)
    end_y = np.sqrt(A)*end_y/np.sqrt(B*k)


    # Convert to integers for array indexing
    start_x_int = int(start_x)
    end_x_int = int(end_x)
    Nx_int = int(Nx)
    start_y_int = int(start_y)
    end_y_int = int(end_y)
    Ny_int = int(Ny)


    print(f"start_x_int: {start_x_int}, end_x_int: {end_x_int}, Nx_int: {Nx_int}")
    print(f"start_y_int: {start_y_int}, end_y_int: {end_y_int}, Ny_int: {Ny_int}")

    # Ensure start is less than end for both x and y to avoid negative indexing
    area_start_x = min(start_x_int, end_x_int)
    area_size_x = abs(end_x_int - start_x_int)

    area_start_y = min(start_y_int, end_y_int)
    area_size_y = abs(end_y_int - start_y_int)

    # Make sure the area size is reasonable
    if area_size_x == 0:
        area_size_x = 20  # Default size
    if area_size_y == 0:
        area_size_y = 20  # Default size

    print(f"area_start_x: {area_start_x}, area_size_x: {area_size_x}")
    print(f"area_start_y: {area_start_y}, area_size_y: {area_size_y}")

    print(f"For time: {A/B:.2f}")
    print(f"For space: {np.sqrt(A/(B*k)):.2f}")
    print(f"For fuel: {B*CS/A:.5f}")

    print("Generating fire propagation dataset...")
    print(f"Spatial domain: {Nx_int} points, dx = {dx}")
    print(f"Spatial domain: {Ny_int} points, dy = {dy}")
    print(f"Time domain: T = {T_total}, dt = {dt}")

    print("Lambda: ", lamb)
    print("Beta: ", beta)


    print("Q1 (top-right):    w1:", w1_q1, "w2:", w2_q1)
    print("Q2 (top-left):     w1:", w1_q2, "w2:", w2_q2)
    print("Q3 (bottom-left):  w1:", w1_q3, "w2:", w2_q3)
    print("Q4 (bottom-right): w1:", w1_q4, "w2:", w2_q4)



    T_sol,F_sol=Diffusion_Reaction_Windless_Operator_Splitting(1, w1_q1, w2_q1, w1_q2, w2_q2, w1_q3, w2_q3, w1_q4, w2_q4, 1, 1, lamb, beta, 0, Nx_int, Ny_int, dx, dy, dt, T_total, start_x_int, start_y_int, end_x_int, end_y_int, temp1, temp2, fuel1, fuel2)

    # Create coordinate arrays
    x = np.linspace(0, (Nx_int - 1) * dx, Nx_int)
    y = np.linspace(0, (Ny_int - 1) * dy, Ny_int)
    Nt = T_sol.shape[0]
    t = np.linspace(0, T_total, Nt)
    print(f"Solution generated.")
    print(f"Time steps: {Nt}")
    print(f"Temperature range: [{T_sol.min():.4f}, {T_sol.max():.4f}]")
    print(f"Fuel range: [{F_sol.min():.4f}, {F_sol.max():.4f}]")

    # Return parameters along with solution
    params = {
        'Ta': Ta,
        'lamb': lamb,
        'beta': beta,
        'w1_q1': w1_q1,
        'w2_q1': w2_q1,
        'w1_q2': w1_q2,
        'w2_q2': w2_q2,
        'w1_q3': w1_q3,
        'w2_q3': w2_q3,
        'w1_q4': w1_q4,
        'w2_q4': w2_q4
    }

    return x, y, t, T_sol, F_sol, params

def save_to_single_csv(x, y, t, T_sol, F_sol, params, prefix="fire_dataset"):
    """
    Save fire dataset to a CSV file
    - Columns: x, y, t, T, F, Ta, lamb, beta, w1_q1, w2_q1, w1_q2, w2_q2, w1_q3, w2_q3, w1_q4, w2_q4
    """

    # Create data directory if it doesn't exist
    os.makedirs("Fire_PINN_Multi_diffusion/data", exist_ok=True)

    print("\nSaving fire dataset to single CSV file...")

    # T_sol has shape (Nt, Nx, Ny), indexed as T_sol[t_idx, x_idx, y_idx]
    Nt = len(t)
    Nx = len(x)
    Ny = len(y)

    # Create coordinate arrays
    T_mesh, X_mesh, Y_mesh = np.meshgrid(t, x, y, indexing='ij')

    # Flatten
    x_flat = X_mesh.flatten()
    y_flat = Y_mesh.flatten()
    t_flat = T_mesh.flatten()
    T_flat = T_sol.flatten()
    F_flat = F_sol.flatten()

    # Create the arrays for parameters
    total_points = len(x_flat)
    Ta_col = [params['Ta']] + [''] * (total_points - 1)
    lamb_col = [params['lamb']] + [''] * (total_points - 1)
    beta_col = [params['beta']] + [''] * (total_points - 1)
    w1_q1_col = [params['w1_q1']] + [''] * (total_points - 1)
    w2_q1_col = [params['w2_q1']] + [''] * (total_points - 1)
    w1_q2_col = [params['w1_q2']] + [''] * (total_points - 1)
    w2_q2_col = [params['w2_q2']] + [''] * (total_points - 1)
    w1_q3_col = [params['w1_q3']] + [''] * (total_points - 1)
    w2_q3_col = [params['w2_q3']] + [''] * (total_points - 1)
    w1_q4_col = [params['w1_q4']] + [''] * (total_points - 1)
    w2_q4_col = [params['w2_q4']] + [''] * (total_points - 1)

    # Create the DataFrame
    data_df = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        't': t_flat,
        'T': T_flat,
        'F': F_flat,
        'Ta': Ta_col,
        'lamb': lamb_col,
        'beta': beta_col,
        'w1_q1': w1_q1_col,
        'w2_q1': w2_q1_col,
        'w1_q2': w1_q2_col,
        'w2_q2': w2_q2_col,
        'w1_q3': w1_q3_col,
        'w2_q3': w2_q3_col,
        'w1_q4': w1_q4_col,
        'w2_q4': w2_q4_col
    })

    # Save to CSV file
    csv_filename = f"Fire_PINN_Multi_diffusion/data/{prefix}_columns.csv"
    data_df.to_csv(csv_filename, index=False)
    print(f"CSV file created: {csv_filename}")
    print(f"    x, y, t, T, F (all {total_points} rows)")
    print(f"    Ta, lamb, beta, w1/w2 per quadrant")
    print(f"    Total rows: {total_points} (= {len(t)} time points x {len(x)}x{len(y)} space points)")
    print(f"\nParameters saved in first row:")
    print(f"    Ta={params['Ta']}, lamb={params['lamb']}, beta={params['beta']}")
    print(f"    w1_q1={params['w1_q1']}, w2_q1={params['w2_q1']}")
    print(f"    w1_q2={params['w1_q2']}, w2_q2={params['w2_q2']}")
    print(f"    w1_q3={params['w1_q3']}, w2_q3={params['w2_q3']}")
    print(f"    w1_q4={params['w1_q4']}, w2_q4={params['w2_q4']}")
    print(f"\nFirst 10 rows:")
    print(data_df.head(10).to_string(index=False))

    return csv_filename


def main():
    """
    Main function to generate fire dataset CSV files for PINN training
    """
    print("=" * 60)
    print("FIRE DATASET CSV GENERATOR")
    print("=" * 60)

    # Generate dataset
    x, y, t, T_sol, F_sol, params = create_fire_dataset()

    save_to_single_csv(x, y, t, T_sol, F_sol, params, prefix="fire_dataset")

    print("\n" + "=" * 60)
    print("FIRE DATASET CSV FILE CREATED")
    print("=" * 60)

if __name__ == "__main__":
    main()
