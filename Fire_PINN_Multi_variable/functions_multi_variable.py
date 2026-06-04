import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1234)

#Laplacian operator based on a Central Finite Differences scheme
def weighted_laplacian(Z, dx, dy, w1, w2):
    """2D Laplacian with Neumann BCs"""
    #The following scheme, creates the 2nd order Central FD scheme,
    #not element-wise, but row/column-wise. In particular, the first arguments
    #refer to the rows/columns used and the second arguments refer to the
    #rows/columns that aren't used.
    #E.g. [2:,1:-1] means that the rows used are from row 2 until the end
    #and the columns used are all but the first and last.
    #On the other hand, [:-2,1:-1] means that all rows except the last two and
    #interior columns only.
    #For isotropic diffusion, w1=w2=1. For anistropic diffusion, they are used as modulus of diffusion per axis.
    L = np.zeros_like(Z)
    L[1:-1,1:-1] = w1*(Z[2:,1:-1] - 2*Z[1:-1,1:-1] + Z[:-2,1:-1])/dx**2 \
                  + w2*(Z[1:-1,2:] - 2*Z[1:-1,1:-1] + Z[1:-1,:-2])/dy**2
    # Neumann BCs
    L[0,:] = L[1,:]; L[-1,:] = L[-2,:]
    L[:,0] = L[:,1]; L[:,-1] = L[:,-2]
    return L

#Used in operator splitting routines
def reaction_step(Tn, Sn, dt, A, B, C, CS, Ta):
    """Semi-implicit reaction update
    C and CS can be scalars or arrays (for spatially varying parameters)
    """
    exp_term = np.exp(-B/(Tn + 1e-8))
    T_new = Tn + dt * A * (Sn * exp_term - C * (Tn))
    S_new = Sn / (1 + dt*CS*exp_term)
    return T_new, S_new


#A standard Runge-Kutta 4th order scheme with Neumann boundary conditions
def rk4_diffusion(Tn, k, dt, dx, dy, w1, w2):
    """RK4 step for diffusion only"""
    k1 = k * weighted_laplacian(Tn, dx, dy,w1,w2)
    k2 = k * weighted_laplacian(Tn + 0.5*dt*k1, dx, dy,w1,w2)
    k3 = k * weighted_laplacian(Tn + 0.5*dt*k2, dx, dy,w1,w2)
    k4 = k * weighted_laplacian(Tn + dt*k3, dx, dy,w1,w2)
    T_new = Tn + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    # Neumann BCs
    T_new[0,:] = T_new[1,:]; T_new[-1,:] = T_new[-2,:]
    T_new[:,0] = T_new[:,1]; T_new[:,-1] = T_new[:,-2]
    return T_new

#A routine for computing numerically the Temperature T and Fuel Supply fraction F based on the model derived
#by Mandel et al, Math. Comp. Sim. (2008) - no wind present - non-dimensional equations.
def Diffusion_Reaction_Windless_Operator_Splitting(k, w1, w2, A, B, C1, CS1, C4, CS4, C3, CS3, C2, CS2, Ta, Nx, Ny, dx, dy, dt, T_total,
                                                   area_start_x, area_start_y, area_size_x, area_size_y, temp1, temp2, fuel1, fuel2):
    """
    4 quadrants with different parameters
    Q1 is top-right, where: x >= mid_x, y >= mid_y = C1, CS1 (lamb1, beta1)
    Q2 is top-left, where:  x < mid_x,  y >= mid_y = C2, CS2 (lamb2, beta2)
    Q3 is bottom-left, where: x < mid_x,  y < mid_y  = C3, CS3 (lamb3, beta3)
    Q4 is bottom-right, where: x >= mid_x, y < mid_y  = C4, CS4 (lamb4, beta4)
    """

    x = np.linspace(0, (Nx - 1) * dx, Nx)
    y = np.linspace(0, (Ny - 1) * dy, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Nt = int(T_total/dt) + 1
    T=Ta*np.ones((Nt,Nx,Ny))

    #The following sets a standard initial temperature distribution
    #T[0] = Ta*np.exp(-((X - 0.5*dx)**2 + (Y - 0.5*dy)**2)/(2*0.1**2))
    F = np.ones((Nt, Nx, Ny))

    #F = np.random.uniform(0, 1, (Nt, Nx, Ny))
    road_end_x, road_end_y = area_start_x + area_size_x, area_start_y + area_size_y

    T[0, area_start_x:road_end_x, area_start_y:road_end_y] = np.random.uniform(temp1, temp2, size=(area_size_x, area_size_y))
    F[0, area_start_x:road_end_x, area_start_y:road_end_y] = np.random.uniform(fuel1, fuel2, size=(area_size_x, area_size_y))    


    # Create spatially varying C and CS arrays for 4 each quadrant
    mid_x = Nx // 2 # middle x
    mid_y = Ny // 2 # middle y

    # Initialize C and CS arrays with zeros
    C_array = np.zeros((Nx, Ny))
    CS_array = np.zeros((Nx, Ny))

    # Q3 bottom-left: low x, low y
    C_array[:mid_x, :mid_y] = C3
    CS_array[:mid_x, :mid_y] = CS3

    # Q4 bottom-right: low y, high x
    C_array[:mid_x, mid_y:] = C4
    CS_array[:mid_x, mid_y:] = CS4

    # Q2 top-left: high y, low x
    C_array[mid_x:, :mid_y] = C2
    CS_array[mid_x:, :mid_y] = CS2

    # Q1 bottom-right: high x, high y
    C_array[mid_x:, mid_y:] = C1
    CS_array[mid_x:, mid_y:] = CS1

    for n in range(Nt - 1):
        Tn = T[n].copy()
        Fn = F[n].copy()

    # 1. RK4 diffusion step
        T_diff = rk4_diffusion(Tn, k, dt, dx, dy,w1,w2)

    # 2. Reaction step (semi-implicit) with spatially varying parameters
        T_next, F_next = reaction_step(T_diff, Fn, dt,A,B,C_array,CS_array,Ta)

    # Store
        T[n + 1] = T_next
        F[n + 1] = F_next

        # if n % (Nt // 10) == 0:
        #     print(f"Progress: {100 * n // Nt}%")
    return T, F

def Diffusion_Reaction_Windless_Explicit_Euler_Implicit(k,w1,w2,A,B,C,CS,Ta,Nx,Ny,dx,dy,dt,T_total,area_start_x, area_start_y, area_size_x, area_size_y,temp1,temp2,fuel1,fuel2):
    x = np.linspace(0, (Nx - 1) * dx, Nx)
    y = np.linspace(0, (Ny - 1) * dy, Ny)
    Nt = int(T_total / dt) + 1
    T = Ta*np.ones((Nt, Nx, Ny))
    F = np.ones((Nt, Nx, Ny))
    road_end_x, road_end_y = area_start_x + area_size_x, area_start_y + area_size_y

    T[0, area_start_x:road_end_x, area_start_y:road_end_y] = np.random.uniform(temp1, temp2,
                                                                               size=(area_size_x, area_size_y))
    F[0, area_start_x:road_end_x, area_start_y:road_end_y] = np.random.uniform(fuel1, fuel2,
                                                                               size=(area_size_x, area_size_y))
    for n in range(Nt - 1):
        Tn = T[n].copy()
        Fn = F[n].copy()

        # 1. Diffusion step (explicit)
        T_diff = Tn + dt * k * weighted_laplacian(Tn, dx, dy,w1,w2)

        # 2. Reaction step (semi-implicit)
        T_next, F_next = reaction_step(T_diff, Fn, dt,A,B,C,CS,Ta)

        # Apply Neumann BCs
        T_next[0, :] = T_next[1, :];
        T_next[-1, :] = T_next[-2, :]
        T_next[:, 0] = T_next[:, 1];
        T_next[:, -1] = T_next[:, -2]
        F_next[0, :] = F_next[1, :];
        F_next[-1, :] = F_next[-2, :]
        F_next[:, 0] = F_next[:, 1];
        F_next[:, -1] = F_next[:, -2]

        # Store next step
        T[n + 1] = T_next
        F[n + 1] = F_next

    return T, F

def Diffusion_Reaction_Windless_Explicit_Euler_Central(k,w1,w2,A,B,C,CS,Ta,Nx,Ny,dx,dy,dt,T_total,area_start_x, area_start_y, area_size_x, area_size_y,temp1,temp2,fuel1,fuel2):
    x = np.linspace(0, (Nx - 1) * dx, Nx)
    y = np.linspace(0, (Ny - 1) * dy, Ny)
    Nt = int(T_total / dt) + 1
    T = np.zeros((Nt, Nx, Ny))
    F = np.ones((Nt, Nx, Ny))
    road_end_x, road_end_y = area_start_x + area_size_x, area_start_y + area_size_y

    T[0, area_start_x:road_end_x, area_start_y:road_end_y] = np.random.uniform(temp1, temp2,
                                                                               size=(area_size_x, area_size_y))
    F[0, area_start_x:road_end_x, area_start_y:road_end_y] = np.random.uniform(fuel1, fuel2,
                                                                               size=(area_size_x, area_size_y))

    for n in range(Nt - 1):
        Tn = T[n].copy()
        Fn = F[n].copy()

        # Laplacian for T
        d2Tdx2 = (Tn[2:, 1:-1] - 2 * Tn[1:-1, 1:-1] + Tn[:-2, 1:-1]) / dx ** 2
        d2Tdy2 = (Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) / dy ** 2

        # Reaction term
        exp_term = np.exp(-B / (Tn[1:-1, 1:-1] - Ta + 1e-8))
        reaction_T = A * (Fn[1:-1, 1:-1] * exp_term - C * (Tn[1:-1, 1:-1] - Ta))

        # Update T
        T[n + 1, 1:-1, 1:-1] = Tn[1:-1, 1:-1] + dt * (k * (w1*d2Tdx2 + w2*d2Tdy2) + reaction_T)

        # Neumann BCs (zero-flux) for T
        T[n + 1, 0, :] = T[n + 1, 1, :]
        T[n + 1, -1, :] = T[n + 1, -2, :]
        T[n + 1, :, 0] = T[n + 1, :, 1]
        T[n + 1, :, -1] = T[n + 1, :, -2]

        # Update S (reaction only)
        mask = Tn[1:-1, 1:-1] > Ta
        dFdt = -CS * Fn[1:-1, 1:-1] * exp_term * mask
        F[n + 1, 1:-1, 1:-1] = Fn[1:-1, 1:-1] + dt * dFdt

        # Neumann BCs for S
        F[n + 1, 0, :] = F[n + 1, 1, :]
        F[n + 1, -1, :] = F[n + 1, -2, :]
        F[n + 1, :, 0] = F[n + 1, :, 1]
        F[n + 1, :, -1] = F[n + 1, :, -2]

    return T, F

