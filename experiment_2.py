import numpy as np 
import matplotlib.pyplot as plt

########################
# Fixed parameters
########################
L_domain=1.0
T_final=1.0
velocity = 1.0 
dt = 0.005 
N_t_steps=int(T_final/dt)

# for figure
time_fractiosns=[0,0.2,0.4,0.6,0.8,1.0] 
plot_indices=[int(frac*N_t_steps) for frac in time_fractiosns]
colors=['k','r','g','b','m','c']

########################################
### Functions 
########################################
def initial_condition_gaussian(x):
    mu=0.5
    sigma=0.1
    u=np.exp(-0.5*((x-mu)/sigma)**2)
    return u



### upwind scheme
def upwind_step(u, C):
    u_new = np.copy(u)
    if C > 0: #FTBS
        u_new[1:] = u[1:] - C * (u[1:] - u[:-1])
        u_new[0] = u[0] - C * (u[0] - u[-1])
    else:   #FTFS
        u_new[:-1] = u[:-1] - C * (u[1:] - u[:-1])
        u_new[-1] = u[-1] - C * (u[0] - u[-1])
    return u_new

### lax-wendroff scheme
def lax_wendroff_step(u, C):
    u_new = np.copy(u)
    u_new[1:-1] = (u[1:-1] - 0.5 * C * (u[2:] - u[:-2]) + 0.5 * C**2 * (u[2:] - 2 * u[1:-1] + u[:-2]))
    # Periodic boundary conditions
    u_new[0] = (u[0] - 0.5 * C * (u[1] - u[-1]) + 0.5 * C**2 * (u[1] - 2 * u[0] + u[-1]))
    u_new[-1] = (u[-1] - 0.5 * C * (u[0] - u[-2]) + 0.5 * C**2 * (u[0] - 2 * u[-1] + u[-2]))
    return u_new

### calculate numerical solution at time= T_final
def numerical_solution(initial_condition, scheme_step, x_grid, N_t_steps, C):
    u_current = initial_condition
    for n in range(N_t_steps):
        u_current = scheme_step(u_current, C)
    return u_current

def exact_solution(x):
    shifted_x = x - velocity * T_final
    periodic_shifted_x = np.mod(shifted_x, L_domain) 
    return initial_condition_gaussian(periodic_shifted_x)

### Calculate the error between numerical and exact solution
def calculate_error(numerical, exact):
    return np.linalg.norm(exact-numerical) 


#######################################
# Run simulation for different grid resolutions
#######################################
N_x_points_list= [20,40,60,80,100,120,140,160,180,200]

######## Upwind Scheme ##########
fig, axes = plt.subplots(5,2, figsize=(10,15)) 
fig.suptitle(f'Numerical Solution at $t={T_final}$ (Upwind)',fontsize=13)
fig.subplots_adjust(hspace=10, wspace=5)
axes_flat = axes.flatten()
error_list=[]

for N_x_points, ax in zip(N_x_points_list, axes_flat):
    dx=L_domain/N_x_points
    x_grid=np.linspace(0,L_domain,N_x_points,endpoint=False)
    C=velocity*dt/dx
    initial_sol=initial_condition_gaussian(x_grid)
    numerical_sol=numerical_solution(initial_sol, upwind_step, x_grid, N_t_steps, C)
    exact_sol=exact_solution(x_grid)
    error_final=calculate_error(numerical_sol, exact_sol) 
    error_list.append(error_final) 
    ax.plot(x_grid, exact_sol, 'k-', label='Exact Solution')
    ax.plot(x_grid, numerical_sol, 'r--', label='Numerical Solution')
    ax.set_title(f'$N_x = {N_x_points}$')

axes_flat[0].legend(loc='upper right', fontsize=6)
axes_flat[0].set_xlabel('x', fontsize=8)
axes_flat[0].set_ylabel('u(x,T)', fontsize=8)
plt.tight_layout()
plt.savefig('figure_delta_x_upwind.png', dpi=300)
plt.show()

# Plot error vs grid points
plt.figure(figsize=(8,6))
plt.plot(N_x_points_list, error_list, 'o-', label='Upwind Scheme', color='b')
plt.xlabel('Number of Spatial Grid Points')
plt.ylabel('Error')
plt.title('Error vs Grid Resolution')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig('error_vs_grid_resolution_upwind.png', dpi=300)
plt.show()

######### Lax-Wendroff Scheme ##########
fig, axes = plt.subplots(5,2, figsize=(10,15)) 
fig.suptitle(f'Numerical Solution at $t={T_final}$ (Lax-Wendroff)',fontsize=13)
fig.subplots_adjust(hspace=10, wspace=5)
axes_flat = axes.flatten()
error_list=[]

for N_x_points, ax in zip(N_x_points_list, axes_flat):
    dx=L_domain/N_x_points
    x_grid=np.linspace(0,L_domain,N_x_points,endpoint=False)
    C=velocity*dt/dx
    initial_sol=initial_condition_gaussian(x_grid)
    numerical_sol=numerical_solution(initial_sol,lax_wendroff_step, x_grid, N_t_steps, C)
    exact_sol=exact_solution(x_grid)
    error_final=calculate_error(numerical_sol, exact_sol) 
    error_list.append(error_final) 
    ax.plot(x_grid, exact_sol, 'k-', label='Exact Solution')
    ax.plot(x_grid, numerical_sol, 'r--', label='Numerical Solution')
    ax.set_title(f'$N_x = {N_x_points}$')

axes_flat[0].legend(loc='upper right', fontsize=6)
axes_flat[0].set_xlabel('x', fontsize=8)
axes_flat[0].set_ylabel('u(x,T)', fontsize=8)
plt.tight_layout()
plt.savefig('figure_delta_x_lax_wendroff.png', dpi=300)
plt.show()

# Plot error vs grid points
plt.figure(figsize=(8,6))
plt.plot(N_x_points_list, error_list, 'o-', label='Lax-Wendroff Scheme', color='b')
plt.xlabel('Number of Spatial Grid Points')
plt.ylabel('Error')
plt.title('Error vs Grid Resolution')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig('error_vs_grid_resolution_lax_wendroff.png', dpi=300)
plt.show()