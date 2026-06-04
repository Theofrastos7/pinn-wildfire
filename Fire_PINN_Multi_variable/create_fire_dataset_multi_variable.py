import os
import pandas as pd
import numpy as np
from functions_multi_variable import Diffusion_Reaction_Windless_Operator_Splitting

def create_fire_dataset():
    """
    Generate fire propagation dataset
    - x: spatial array
    - y: spatial array   
    - t: time array
    - T_sol: temperature solution array (space x time)
    - F_sol: fuel solution array (space x time)
    - params: dictionary of parameters used in the simulation
    (Ta, lamb1, beta1, lamb2, beta2, lamb3, beta3, lamb4, beta4, w1, w2)
    """
    
    #Parameters changed to fit a non-dimensional model 

    # fixed constants
    k, A, C = 0.28, 187, 0.000056
    B = 558.49
    CS = 0.1625
    Ta = 0

    # Initial area of ignition
    start_x = 50
    start_y = 50
    end_x = 25
    end_y = 25

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

    # dispersion weights in y and x axis. change at will to create different fire patterns
    w1 = 2.5
    w2 = 1.5

    # Define 4 quadrants with different parameters
    # Q1 top-right:    x >= mid, y >= mid = lamb1, beta1
    # Q2 top-left:     x < mid,  y >= mid = lamb2, beta2
    # Q3 bottom-left:  x < mid,  y < mid  = lamb3, beta3
    # Q4 bottom-right: x >= mid, y < mid  = lamb4, beta4

    q1 = 1.2  # different qs for each quadrant. change as you wish
    q2 = 1.1
    q3 = 0.9  
    q4 = 0.8  

    # Calculate lamb and beta for each quadrant. Note: these are scaled versions of the base lamb and beta
    # The scaled version are used so the lamb1-4 and beta1-4 wont differ to much from the original. 
    # Also we don't calculate them from the start because the variables A,B,C,CS 
    # are used in the reaction step and for the normalization of the temperature and fuel. 
    # So we want to keep them as they are and just scale the lamb and beta for each quadrant. 

    lamb1 = q1 * lamb
    beta1 = q1 * beta

    lamb2 = q2 * lamb
    beta2 = q2 * beta

    lamb3 = q3 * lamb
    beta3 = q3 * beta

    lamb4 = q4 * lamb
    beta4 = q4 * beta

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
    start_x = np.sqrt(A) * start_x / np.sqrt(B * k)
    end_x = np.sqrt(A) * end_x / np.sqrt(B * k)
    start_y = np.sqrt(A) * start_y / np.sqrt(B * k)
    end_y = np.sqrt(A) * end_y / np.sqrt(B * k)
    dy= np.sqrt(A)*dy/np.sqrt(B*k)
    Ny= np.sqrt(A)*Ny/np.sqrt(B*k)

    # Convert to integers for array indexing
    start_x_int = int(start_x)
    start_y_int = int(start_y)
    end_x_int = int(end_x)
    end_y_int = int(end_y)
    Nx_int = int(Nx)
    Ny_int = int(Ny)

    print(f"start_x: {start_x}, end_x: {end_x}, Nx_int: {Nx_int}")
    print(f"start_y: {start_y}, end_y: {end_y}, Ny_int: {Ny_int}")

    print(f"For time: {A/B:.2f}")
    print(f"For space: {np.sqrt(A/(B*k)):.2f}")
    print(f"For fuel: {B*CS/A:.5f}")
    
    print("Generating fire propagation dataset...")
    print(f"Spatial domain: {Nx_int} points, dx = {dx}")
    print(f"Spatial domain: {Ny_int} points, dy = {dy}")
    print(f"Time domain: T = {T_total}, dt = {dt}")

    print("Q1 (top-right):    Lambda1:", lamb1, "Beta1:", beta1)
    print("Q2 (top-left):     Lambda2:", lamb2, "Beta2:", beta2)
    print("Q3 (bottom-left):  Lambda3:", lamb3, "Beta3:", beta3)
    print("Q4 (bottom-right): Lambda4:", lamb4, "Beta4:", beta4)

    

    T_sol,F_sol=Diffusion_Reaction_Windless_Operator_Splitting(1,w1,w2,1,1,lamb1,beta1,lamb2,beta2,lamb3,beta3,lamb4,beta4,0,Nx_int,Ny_int,dx,dy,dt,T_total,start_x_int,start_y_int,end_x_int,end_y_int,temp1,temp2,fuel1,fuel2)

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
        'lamb1': lamb1,
        'beta1': beta1,
        'lamb2': lamb2,
        'beta2': beta2,
        'lamb3': lamb3,
        'beta3': beta3,
        'lamb4': lamb4,
        'beta4': beta4,
        'w1': w1,
        'w2': w2
    }

    return x, y, t, T_sol, F_sol, params

def save_to_single_csv(x, y, t, T_sol, F_sol, params, prefix="fire_dataset"):
    """
    Save fire dataset to a CSV file
    - Columns: x, y, t, T, F, Ta, w1, w2, lamb1, beta1, lamb2, beta2, lamb3, beta3, lamb4, beta4
    """

    # Create data directory if it doesn't exist
    os.makedirs("Fire_PINN_Multi_variable/data", exist_ok=True)

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
    w1_col = [params['w1']] + [''] * (total_points - 1)
    w2_col = [params['w2']] + [''] * (total_points - 1)
    lamb1_col = [params['lamb1']] + [''] * (total_points - 1)  
    beta1_col = [params['beta1']] + [''] * (total_points - 1)  
    lamb2_col = [params['lamb2']] + [''] * (total_points - 1)  
    beta2_col = [params['beta2']] + [''] * (total_points - 1)  
    lamb3_col = [params['lamb3']] + [''] * (total_points - 1)  
    beta3_col = [params['beta3']] + [''] * (total_points - 1)  
    lamb4_col = [params['lamb4']] + [''] * (total_points - 1)  
    beta4_col = [params['beta4']] + [''] * (total_points - 1)  

    # Create the DataFrame
    data_df = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        't': t_flat,
        'T': T_flat,
        'F': F_flat,
        'Ta': Ta_col,
        'w1': w1_col,
        'w2': w2_col,
        'lamb1': lamb1_col,
        'beta1': beta1_col,
        'lamb2': lamb2_col,
        'beta2': beta2_col,
        'lamb3': lamb3_col,
        'beta3': beta3_col,
        'lamb4': lamb4_col,
        'beta4': beta4_col
    })

    # Save to CSV file 
    csv_filename = f"Fire_PINN_Multi_variable/data/{prefix}_columns.csv"
    data_df.to_csv(csv_filename, index=False)
    print(f"CSV file created: {csv_filename}")
    print(f"    x, y, t, T, F (all {total_points} rows)")
    print(f"    Ta, w1, w2, lamb1-4, beta1-4")
    print(f"    Total rows: {total_points} (= {len(t)} time points × {len(x)}×{len(y)} space points)")
    print(f"\nParameters saved in first row:")
    print(f"    Ta={params['Ta']}, w1={params['w1']}, w2={params['w2']}")
    print(f"    lamb1={params['lamb1']}, beta1={params['beta1']}")
    print(f"    lamb2={params['lamb2']}, beta2={params['beta2']}")
    print(f"    lamb3={params['lamb3']}, beta3={params['beta3']}")
    print(f"    lamb4={params['lamb4']}, beta4={params['beta4']}")
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
