import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from casadi import *
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Load file
x_data = pd.read_excel('map/x_wp.xls', usecols=[0], header=None) 
y_data = pd.read_excel('map/y_wp.xls', usecols=[0], header=None)  

if len(x_data) != len(y_data):
    raise ValueError("X and Y files have different numbers of entries.")

# Create waypoints
path_points = np.column_stack((x_data.values.flatten(), y_data.values.flatten()))

# path_x = np.linspace(0, 20, 200)  
# path_y = np.sin(path_x/2)*2 

# Simulation parameters
N = 4
dt = 0.1
L = 2.0  

# Constraints
delta_max = np.pi / 4 
a_max = 1.0 
a_min = -a_max  

# Initialization
x0, y0, theta0, v0 = path_points[0,0], path_points[0,1], 0, 0
# x0, y0, theta0, v0 = 0, 0, 0, 0
x_traj, y_traj = [x0], [y0]

# MPC setup
for i in range(len(path_points)-N):
    opti = Opti()  

    A = opti.variable(N)
    Delta = opti.variable(N)
    X = opti.variable(N+1)
    Y = opti.variable(N+1)
    theta = opti.variable(N+1)
    V = opti.variable(N+1)

    # Provide an initial guess
    opti.set_initial(X[0], x0)
    opti.set_initial(Y[0], y0)
    opti.set_initial(theta[0], theta0)
    opti.set_initial(V[0], v0)
    opti.set_initial(Delta, 0)
    opti.set_initial(A, 0)

    # Constraints
    for j in range(N):
        opti.subject_to(X[j+1] == X[j] + dt * V[j] * cos(theta[j]))
        opti.subject_to(Y[j+1] == Y[j] + dt * V[j] * sin(theta[j]))
        opti.subject_to(theta[j+1] == theta[j] + dt * V[j] / L * tan(Delta[j]))
        opti.subject_to(V[j+1] == V[j] + dt * A[j])
        opti.subject_to(-delta_max <= Delta[j])
        opti.subject_to(Delta[j] <= delta_max)
        opti.subject_to(a_min <= A[j])
        opti.subject_to(A[j] <= a_max)

    # Set up the optimization problem
    objective = 0
    for j in range(N):
        objective += ((X[j] - path_points[i+j,0])**2 + (Y[j] - path_points[i+j,1])**2)
        # objective += ((X[j] - path_x[i+j])**2 + (Y[j] - path_y[i+j])**2)
    opti.minimize(objective)

    # Set solver options for debugging
    opti.solver('ipopt', {'ipopt': {'print_level': 5, 'tol': 1e-6}})
    sol = opti.solve()

    # Update to next state
    x0, y0, theta0, v0 = sol.value(X[1]), sol.value(Y[1]), sol.value(theta[1]), sol.value(V[1])

    # Store the new position
    x_traj.append(x0)
    y_traj.append(y0)

# Creating the figure and axis for the animation
fig, ax = plt.subplots()
ax.set_xlim(np.min(path_points[:,0]) - 1, np.max(path_points[:,0]) + 1)
ax.set_ylim(np.min(path_points[:,1]) - 1, np.max(path_points[:,1]) + 1)

vehicle_size = [4,12]

# Plotting waypoints
plt.scatter(path_points[:,0], path_points[:,1], label='Waypoints')

# Initial vehicle representation (this will be updated during the animation)
vehicle_rect = Rectangle((0 - vehicle_size[0]/2, 0 - vehicle_size[1]/2), vehicle_size[0], vehicle_size[1], fill=False, color="blue")
ax.add_patch(vehicle_rect)

# Line object to show the trajectory
trajectory_line, = ax.plot([], [], 'b-', lw=2)

# Initialization function for the animation
def init():
    trajectory_line.set_data([], [])
    return trajectory_line, vehicle_rect

# Function to update the position of the car for each frame of the animation
def update(frame):
    # Updating the vehicle's position
    x, y = x_traj[frame], y_traj[frame]
    vehicle_rect.set_xy((x - vehicle_size[0]/2, y - vehicle_size[1]/2))
    
    # Updating the trajectory line
    trajectory_line.set_data(x_traj[:frame+1], y_traj[:frame+1])
    return trajectory_line, vehicle_rect

# Creating the animation
ani = FuncAnimation(fig, update, frames=range(len(x_traj)), init_func=init, blit=True, interval=100)

plt.show()
# # Plot
# plt.scatter(path_points[:,0], path_points[:,1], label='Waypoints')
# # plt.scatter(path_x, path_y, label='Waypoints')
# plt.plot(x_traj, y_traj, 'b-', label='Vehicle Trajectory')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.show()




