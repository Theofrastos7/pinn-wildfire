import numpy as np
import matplotlib.pyplot as plt
from functions_multi_variable import *

#Parameters changed to fit a non-dimensional model 
# fixed constants
k = 0.28
A = 187
C = 0.000048
B = 558.49
CS = 0.1625
Ta = 0

# Initial area of ignition
start_x = 50
start_y = 50
end_x = 25
end_y = 25

# Total time and time step
dt = 0.5
T_total = 300
original_dt = dt

# Spatial discretization
dx = 2
dy = 2
Nx = 125
Ny = 125

w1 = 2.5
w2 = 1.5

# lamb is the cooling parameter, beta is the fuel consumption rate
lamb = C * B
beta = B * CS / A

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
# The scaled version are used so the lamb1-4 and beta1-4 should not differ to much from the original. 
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

print("Q1 (top-right):    Lambda1:", lamb1, "Beta1:", beta1)
print("Q2 (top-left):     Lambda2:", lamb2, "Beta2:", beta2)
print("Q3 (bottom-left):  Lambda3:", lamb3, "Beta3:", beta3)
print("Q4 (bottom-right): Lambda4:", lamb4, "Beta4:", beta4)


# Fire at ignition 
temp1 = (1200 - Ta) / B
temp2 = (1260 - Ta) / B

# Time for non-dimensional model
original_T_total = T_total
dt = (A / B) * dt
T_total = (A / B) * (T_total)

# Space for non-dimensional model
dx = np.sqrt(A) * dx / np.sqrt(B * k)
dy = np.sqrt(A) * dy / np.sqrt(B * k)
Nx = np.sqrt(A) * Nx / np.sqrt(B * k)
Ny = np.sqrt(A) * Ny / np.sqrt(B * k)
start_x = np.sqrt(A) * start_x / np.sqrt(B * k)
end_x = np.sqrt(A) * end_x / np.sqrt(B * k)
start_y = np.sqrt(A) * start_y / np.sqrt(B * k)
end_y = np.sqrt(A) * end_y / np.sqrt(B * k)

# Convert to integers for array indexing
start_x_int = int(start_x)
start_y_int = int(start_y)
end_x_int = int(end_x)
end_y_int = int(end_y)
Nx_int = int(Nx)
Ny_int = int(Ny)

# These are the scales of the parameters. 
print(f"For time: {A/B:.2f}")
print(f"For space: {np.sqrt(A/(B*k)):.2f}")
print(f"For fuel: {B*CS/A:.5f}")

T1,F1=Diffusion_Reaction_Windless_Operator_Splitting(1,w1,w2,1,1,lamb1,beta1,lamb2,beta2,lamb3,beta3,lamb4,beta4,0,Nx_int,Ny_int,dx,dy,dt,T_total,start_x_int, start_y_int, end_x_int, end_y_int,temp1,temp2,0.45,0.55)

# --- Plot snapshots ---

# time snapshots considering the original t_total
time_snapshots = [int(i * original_T_total / 4) for i in range(5)]
indices = [int(t/original_dt) for t in time_snapshots]


fig, axs = plt.subplots(len(time_snapshots), 2, figsize=(12, 20))
for i, idx in enumerate(indices):
    imT = axs[i,0].imshow(T1[idx].T, origin='lower', cmap='hot', extent=[0,(Nx_int-1)*dx,0,(Ny_int-1)*dy])
    axs[i,0].set_title(f"T at t={time_snapshots[i]}")
    axs[i,0].set_xlabel('x'); axs[i,0].set_ylabel('y')
    fig.colorbar(imT, ax=axs[i,0])

    imS = axs[i,1].imshow(F1[idx].T, origin='lower', cmap='viridis', extent=[0,(Nx_int-1)*dx,0,(Ny_int-1)*dy])
    axs[i,1].set_xlabel('x'); axs[i,1].set_ylabel('y')
    fig.colorbar(imS, ax=axs[i,1])

plt.tight_layout()
plt.savefig('Fire_PINN_Multi_variable/graphs/snapshots.png', dpi=150, bbox_inches='tight')
plt.show()

t=np.linspace(0,T_total,T1.shape[0])
x=np.linspace(0,(Nx_int-1)*dx,Nx_int)
y=np.linspace(0,(Ny_int-1)*dy,Ny_int)

# plot spatial snapshots as 3D surfaces
X, Y = np.meshgrid(x, y, indexing='ij')

# Choose time indices to plot 
time_indices = [0, len(t)//3, 2*len(t)//3, -1]

fig = plt.figure(figsize=(18, 4))
for i, idx in enumerate(time_indices):
    ax = fig.add_subplot(1, len(time_indices), i+1, projection='3d')
    surf = ax.plot_surface(X, Y, T1[idx, :, :],
                           cmap='hot', edgecolor='none')
    ax.set_title(f't = {t[idx]:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature (non-dim.)')
    fig.colorbar(surf, ax=ax, shrink=0.5)
plt.tight_layout()
plt.savefig('Fire_PINN_Multi_variable/graphs/temperature_3d_snapshots.png', dpi=150, bbox_inches='tight')
plt.show()

# Diffusion fronts and reaction zones propagate over time
idxes = [80,70,60,90]

# Filter idxes to be within bounds
valid_idxes = [idx for idx in idxes if idx < Ny_int]

plt.figure(figsize=(15, 10))
for i, y_idx in enumerate(valid_idxes):
  plt.subplot(2, 3, i+1)
  plt.imshow(
    T1[:, :, y_idx],
    aspect='auto',
    origin='lower',
    extent=[x[0], x[-1], t[0], t[-1]],
    cmap='hot'
  )
  plt.xlabel('x')
  plt.ylabel('time')
  plt.title(f'Space–time slice at y = {y[y_idx]:.3f}')
  plt.colorbar(label='Temperature (non-dim.)')

plt.tight_layout()
plt.show()