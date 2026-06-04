
import numpy as np
import matplotlib.pyplot as plt
from functions import Diffusion_Reaction_Windless_Operator_Splitting
#For the non-dimensional model use the following
#A--->
#B--->
#k--->1
#C--->B*C
#CS--->B*CS/A
#Ta--->0
#x--->np.sqrt(A)*x/np.sqrt(B*k)
#t--->t*A/B
k,A,C = 0.28,187.93, 0.000048372
B = 558.49
CS = 0.1625
Ta = 306
fire_size_x = 50            # width of fire square in grid cells
fire_size_y = 50            # height of fire square in grid cells
fire_start_x = (250 - fire_size_x) // 2   # = 100, centered in x
fire_start_y = (250 - fire_size_y) // 2   # = 100, centered in y
dt=0.5
T_total=500
dx=2
dy=2
temp1=1200
temp2=1260
w1 = 1
w2 = 1
#dt,T=(A/B)*(dt,T)
#x,y,dx,dy,start_x,end_x,start_y,end_y-->np.sqrt(A/(k*B))*(x,y,dx,dy,start_x,end_x,start_y,end_y)
T1,F1=Diffusion_Reaction_Windless_Operator_Splitting(k,w1,w2,A,B,C,CS,0,250,250,dx,dy,dt,T_total,fire_start_x,fire_start_y,fire_size_x,fire_size_y,temp1-Ta,temp2-Ta,0.45,0.55)
# T1=abs((T1-T2)/T1)
# F1=F1-F2
# T3,F3=MWL.Diffusion_Reaction_Windless_Explicit_Euler_Central(k,1,1,A,B,C,CS,Ta,250,250,2,2,0.1,110,70, 70, 50, 50,1200,1260,0.25,0.35)
# --- Plot snapshots ---
time_snapshots = [20, 140, 260, 380, 500]
indices = [int(t/0.5) for t in time_snapshots]

fig, axs = plt.subplots(len(time_snapshots), 2, figsize=(12, 20))
for i, idx in enumerate(indices):
    imT = axs[i,0].imshow(T1[idx], origin='lower', cmap='hot', extent=[0,500,0,500])
    axs[i,0].set_title(f"T at t={time_snapshots[i]}")
    axs[i,0].set_xlabel('x'); axs[i,0].set_ylabel('y')
    fig.colorbar(imT, ax=axs[i,0])

    imS = axs[i,1].imshow(F1[idx], origin='lower', cmap='viridis', extent=[0,500,0,500])
    axs[i,1].set_xlabel('x'); axs[i,1].set_ylabel('y')
    fig.colorbar(imS, ax=axs[i,1])

t=np.linspace(0,500,int(500/0.5))
x=np.linspace(0,500,250)
y=np.linspace(0,500,250)
#plot spatial snapshots as 3D surfaces
X, Y = np.meshgrid(x, y, indexing='ij')

# Choose time indices to plot (e.g., start, middle, end)
time_indices = [0, len(t)//3, 2*len(t)//3, -1]

fig = plt.figure(figsize=(18, 4))
for i, idx in enumerate(time_indices):
    ax = fig.add_subplot(1, len(time_indices), i+1, projection='3d')
    surf = ax.plot_surface(X, Y, T1[idx, :, :],
                           cmap='hot', edgecolor='none')
    ax.set_title(f't = {t[idx]:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperature (K)')
    fig.colorbar(surf, ax=ax, shrink=0.5)
plt.tight_layout()
plt.savefig('Non_normalized_2D/graphs/temperature_3d_snapshots.png', dpi=150, bbox_inches='tight')
plt.show()

# Diffusion fronts and reaction zones propagate over time
idxes = [80,90,100,110,120]
plt.figure(figsize=(6,5))
for y_idx in idxes:
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
  plt.colorbar(label='Temperature (K)')
  plt.show()

# tt=np.linspace(0,110,int(110/0.1))
# u=np.zeros(int(110/0.1))
# v=np.zeros(int(110/0.1))
# w=np.zeros(int(110/0.1))
# f=np.zeros(int(110/0.1))
# g=np.zeros(int(110/0.1))
# for i in range(int(110/0.1)):
  # u[i]=T1[i,4,20]
  # v[i]=T1[i,200,220]
  # f[i]=T1[i,0,15]
  # w[i]=T1[i,85,95]
  # g[i]=T1[i,110,105]
# plt.plot(tt,u,color='b')
# plt.plot(tt,v,color='g',linestyle='dashed')
# plt.plot(tt,w,color='purple',linestyle='dotted')
# plt.plot(tt,f,color='r')
# plt.plot(tt,g,color='orange',linestyle='dashdot')
# plt.show()