import numpy as np 
import matplotlib.pyplot as plt

########################
# Setting up parameters
########################
L_domain=1.0
T_final=1.0
N_x_points=100
velocity = 1.0  # advection velocity
dx=L_domain/N_x_points # Discretization
dt = 0.005
C=velocity*dt/dx # Courant number
N_t_steps=int(T_final/dt)
x_grid=np.linspace(0,L_domain,N_x_points,endpoint=False) # Grid Initialization 

print(f"CFL number: {C}") # Courant Friedrichs Lewy number
print(f"spatial step size : {dx}, total Spatial points: {N_x_points}")
print(f"time step size : {dt}, total time steps: {N_t_steps}")

### setting for figure
time_fractiosns=[0,0.2,0.4,0.6,0.8,1.0] # time fractions to plot
plot_indices=[int(frac*N_t_steps) for frac in time_fractiosns]
colors=['k','r','g','b','m','c']


########################################
### Define functions for simulation
########################################
### Initial condition
# initial condition: Gaussian Pulse
def initial_condition_gaussian(x):
    mu=0.5
    sigma=0.1
    u=np.exp(-0.5*((x-mu)/sigma)**2)
    return u

# initial condition: Square Pulse
def initial_condition_square(x):
    u=np.zeros_like(x)
    u[(x>=0.1) & (x<=0.2)]=1.0
    return u

### schemes
# upwind scheme
def upwind_step(u, C):
    u_new = np.copy(u)
    if C > 0: #FTBS
        u_new[1:] = u[1:] - C * (u[1:] - u[:-1])
        u_new[0] = u[0] - C * (u[0] - u[-1])
    else:   #FTFS
        u_new[:-1] = u[:-1] - C * (u[1:] - u[:-1])
        u_new[-1] = u[-1] - C * (u[0] - u[-1])
    return u_new

# lax-wendroff scheme
def lax_wendroff_step(u, C):
    u_new = np.copy(u)
    u_new[1:-1] = (u[1:-1] - 0.5 * C * (u[2:] - u[:-2]) + 0.5 * C**2 * (u[2:] - 2 * u[1:-1] + u[:-2]))
    # Periodic boundary conditions
    u_new[0] = (u[0] - 0.5 * C * (u[1] - u[-1]) + 0.5 * C**2 * (u[1] - 2 * u[0] + u[-1]))
    u_new[-1] = (u[-1] - 0.5 * C * (u[0] - u[-2]) + 0.5 * C**2 * (u[0] - 2 * u[-1] + u[-2]))
    return u_new

### Excute simulation
def run_advection_simulation(initial_condition_func, upwind_step_func, x_grid, N_t_steps, C, plot_indices):
    u = initial_condition_func(x_grid)
    solution_snapshots = {}
    solution_snapshots[0] = np.copy(u)  # Store the initial condition for plotting
    
    for n in range(1, N_t_steps + 1):
        u = upwind_step_func(u, C)
        
        # Check if the current time step index should be saved
        if n in plot_indices:
            # Store the solution at the current time step
            solution_snapshots[n] = np.copy(u)
            
    return solution_snapshots

def plot_solution_snapshots(solution_snapshots, x_grid, L_domain, dt, velocity, colors, scheme_name='Upwind Scheme', initial_condition='Gaussian Pulse'):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlim(0, L_domain)
    ax.set_ylim(min(0, np.min(list(solution_snapshots.values())) - 0.1), 
                max(1.5, np.max(list(solution_snapshots.values())) + 0.1)) 
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title(f'1D Linear Advection using {scheme_name} (initial condition: {initial_condition})')
    ax.grid(True)
    
    # Sort the snapshots by time step index for orderly plotting
    sorted_steps = sorted(solution_snapshots.keys())
    
    i = 0
    for n_step in sorted_steps:
        u_snap = solution_snapshots[n_step]
        # Calculate physical time
        t_physical = n_step * dt
        
        # Initial condition (n_step=0) is usually plotted with a dashed line
        line_style = '--' if n_step == 0 else '-'
        
        ax.plot(x_grid, u_snap, line_style, 
                color=colors[i % len(colors)], 
                label=f't={t_physical:.3f}s')
        i += 1
        
    ax.legend()
    plt.show()
    
    
########################################
# Running the Simulation and Plotting
########################################
# case1. upwind scheme, gaussian initial condition
# case2. upwind scheme, square initial condition
# case3. lax-wendroff scheme, gaussian initial condition
# case4. lax-wendroff scheme, square initial condition

snapshots_1 = run_advection_simulation(
    initial_condition_gaussian, 
    upwind_step, 
    x_grid, 
    N_t_steps, 
    C, 
    plot_indices
)

# 2. Plot the results
plot_solution_snapshots(
    snapshots_1, 
    x_grid, 
    L_domain, 
    dt, 
    velocity, 
    colors,
    scheme_name='Upwind Scheme',
    initial_condition='Gaussian Pulse'
)

snapshots_2 = run_advection_simulation(
    initial_condition_square, upwind_step, x_grid, N_t_steps, C, plot_indices
)
plot_solution_snapshots(
    snapshots_2, x_grid, L_domain, dt, velocity, colors,scheme_name='Upwind Scheme',initial_condition='Square Pulse'
)
snapshots_3 = run_advection_simulation(
    initial_condition_gaussian, lax_wendroff_step, x_grid, N_t_steps, C, plot_indices
)
plot_solution_snapshots(
    snapshots_3, x_grid, L_domain, dt, velocity, colors,scheme_name='Lax-Wendroff Scheme',initial_condition='Gaussian Pulse'
)
snapshots_4 = run_advection_simulation(
    initial_condition_square, lax_wendroff_step, x_grid, N_t_steps, C, plot_indices
)
plot_solution_snapshots(
    snapshots_4, x_grid, L_domain, dt, velocity, colors,scheme_name='Lax-Wendroff Scheme',initial_condition='Square Pulse'
)
