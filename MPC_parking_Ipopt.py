import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


#Vehicle Param
L = 2.0         
delta_max = np.pi/4 # max steering angle  
safety_margin = 0.1  
dt = 0.1   

def simulate_vehicle(x, y, theta, v, delta, dt, L):
    """
    Simulate vehicle dynamics for one time step.
    """
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + v * np.tan(delta) / L * dt
    return x_next, y_next, theta_next

# Setup the MPC problem 
def setup_mpc_front(N, dt, L, x_start, y_start, theta_start, x_goal, y_goal, theta_goal, obstacles):
    """
    Setup and solve the MPC problem.
    Returns the first control inputs (v, delta) from the optimized trajectory.
    """

    # Steering angle and velocity limits
    delta_min, delta_max = -np.pi/4, np.pi/4
    a_min, a_max = 0, 0.5
    v_min, v_max = 0, 1
    # w_vehicle = 0.5
    # h_vehicle = 1

    opti = ca.Opti()  # Create an optimization problem

    # Variables
    X = opti.variable(N+1)
    Y = opti.variable(N+1)
    Theta = opti.variable(N+1)
    V = opti.variable(N+1)
    Delta = opti.variable(N)
    A = opti.variable(N)

    # Objective function (for simplicity, just trying to reach the goal)
    objective = ca.sum1((X - x_goal)**2 + (Y - y_goal)**2 + (Theta - theta_goal)**2)

    # Dynamics constraints
    constraints = [X[0] == x_start, Y[0] == y_start, Theta[0] == theta_start]
    for i in range(N):
        constraints += [
            X[i+1] == X[i] + V[i] * ca.cos(Theta[i]) * dt,
            Y[i+1] == Y[i] + V[i] * ca.sin(Theta[i]) * dt,
            Theta[i+1] == Theta[i] + V[i] * ca.tan(Delta[i]) / L * dt,
            V[i+1] == V[i] + A[i] *dt,
            A[i] >= a_min, A[i] <= a_max,  # Acceleration constraints
            V[i] >= v_min, V[i] <= v_max,
            Delta[i] >= delta_min, Delta[i] <= delta_max  # Steering angle constraints
        ]

    # Set up the optimization problem
    opti.minimize(objective)
    opti.subject_to(constraints)

    # Provide an initial guess
    opti.set_initial(X, np.linspace(x_start, x_goal, N+1))
    opti.set_initial(Y, np.linspace(y_start, y_goal, N+1))
    opti.set_initial(Theta, np.linspace(theta_start, theta_goal, N+1))
    opti.set_initial(V, 0)
    opti.set_initial(Delta, 0)
    opti.set_initial(A, 0)

    # Set solver options for debugging
    opti.solver('ipopt', {'ipopt': {'print_level': 5, 'tol': 1e-6}})

    try:
        solution = opti.solve()
        # Extract and return the first control inputs from the solution
        v_opt = solution.value(V[0])
        delta_opt = solution.value(Delta[0])
        return v_opt, delta_opt
    except Exception as e:
        print("Solver encountered an error:", e)
        return None, None  # Return a tuple of None to indicate failure

def setup_mpc_back(N, dt, L, x_start, y_start, theta_start, x_goal, y_goal, theta_goal, obstacles):
    """
    Setup and solve the MPC problem.
    Returns the first control inputs (v, delta) from the optimized trajectory.
    """

    # Steering angle and velocity limits
    delta_min, delta_max = -np.pi/4, np.pi/4
    a_min, a_max = -0.5, 0
    v_min, v_max = -1, 0
    # w_vehicle = 0.5
    # h_vehicle = 1

    opti = ca.Opti()  # Create an optimization problem

    # Variables
    X = opti.variable(N+1)
    Y = opti.variable(N+1)
    Theta = opti.variable(N+1)
    V = opti.variable(N+1)
    Delta = opti.variable(N)
    A = opti.variable(N)

    # Objective function (for simplicity, just trying to reach the goal)
    objective = ca.sum1((X - x_goal)**2 + (Y - y_goal)**2 + (Theta - theta_goal)**2)

    # Dynamics constraints
    constraints = [X[0] == x_start, Y[0] == y_start, Theta[0] == theta_start]
    for i in range(N):
        constraints += [
            X[i+1] == X[i] + V[i] * ca.cos(Theta[i]) * dt,
            Y[i+1] == Y[i] + V[i] * ca.sin(Theta[i]) * dt,
            Theta[i+1] == Theta[i] + V[i] * ca.tan(Delta[i]) / L * dt,
            V[i+1] == V[i] + A[i] *dt,
            A[i] >= a_min, A[i] <= a_max,  # Acceleration constraints
            V[i] >= v_min, V[i] <= v_max,
            Delta[i] >= delta_min, Delta[i] <= delta_max  # Steering angle constraints
        ]

    # Set up the optimization problem
    opti.minimize(objective)
    opti.subject_to(constraints)

    # Provide an initial guess
    opti.set_initial(X, np.linspace(x_start, x_goal, N+1))
    opti.set_initial(Y, np.linspace(y_start, y_goal, N+1))
    opti.set_initial(Theta, np.linspace(theta_start, theta_goal, N+1))
    opti.set_initial(V, 0)
    opti.set_initial(Delta, 0)
    opti.set_initial(A, 0)
    # Set solver options for debugging
    opti.solver('ipopt', {'ipopt': {'print_level': 5, 'tol': 1e-6}})

    try:
        solution = opti.solve()
        # Extract and return the first control inputs from the solution
        v_opt = solution.value(V[0])
        delta_opt = solution.value(Delta[0])
        return v_opt, delta_opt
    except Exception as e:
        print("Solver encountered an error:", e)
        return None, None  # Return a tuple of None to indicate failure


# Initial vehicle state
x_current, y_current, theta_current = 0, 5, 0

# Goal state
x_goal, y_goal, theta_goal = 5, 1.5, np.pi/2

# Back waypoint
x_back, y_back, theta_back = 5, y_goal + 5.5, np.pi/2

# Simulation parameters
N = 10
dt = 0.2
L = 2.0
T_sim1 = 100 
T_sim2 = 150  

w_vehicle = 1
h_vehicle = 3

# List of obstacles (as before)
obstacles = [(7,1.5,1,3),(3,1.5,1,3)]

# Initialize lists to store the vehicle's trajectory for plotting
x_trajectory = [x_current]
y_trajectory = [y_current]

# From intial position to back waypoint
for t in range(T_sim1):
    # Solve the MPC problem with the current state as the starting point
    v_opt, delta_opt = setup_mpc_front(N, dt, L, x_current, y_current, theta_current, x_back, y_back, theta_back, obstacles)
    
    if v_opt == None or delta_opt == None:
        print("Solution doesn't exist.")
        break

    # Simulate vehicle dynamics for one time step using the optimized control inputs
    x_next, y_next, theta_next = simulate_vehicle(x_current, y_current, theta_current, v_opt, delta_opt, dt, L)
    
    # Update current state for the next iteration
    x_current, y_current, theta_current = x_next, y_next, theta_next

    # Store the new position
    x_trajectory.append(x_current)
    y_trajectory.append(y_current)

    # Termination condition if the vehicle reaches the goal area
    if np.sqrt((x_current - x_back)**2 + (y_current - y_back)**2) <= 0.2:  # threshold for reaching the goal
        print("Back point reached.")
        break



# From waypoint to goal position
for t in range(T_sim2):
    # Solve the MPC problem with the current state as the starting point
    v_opt, delta_opt = setup_mpc_back(N, dt, L, x_current, y_current, theta_current, x_goal, y_goal, theta_goal, obstacles)
    
    if v_opt == None or delta_opt == None:
        print("Solution doesn't exist.")
        break

    # Simulate vehicle dynamics for one time step using the optimized control inputs
    x_next, y_next, theta_next = simulate_vehicle(x_current, y_current, theta_current, v_opt, delta_opt, dt, L)
    
    # Update current state for the next iteration
    x_current, y_current, theta_current = x_next, y_next, theta_next

    # Store the new position
    x_trajectory.append(x_current)
    y_trajectory.append(y_current)

    # Termination condition if the vehicle reaches the goal area
    if np.sqrt((x_current - x_goal)**2 + (y_current - y_goal)**2) < 0.1:  # threshold for reaching the goal
        print("Goal reached.")
        break


# Creating the figure and axis for the animation
vehicle_size = [1,3]
fig, ax = plt.subplots()
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)

# Plotting static elements (e.g., obstacles)
for ox, oy, ow, oh in obstacles:
    obstacle_rect = Rectangle((ox - ow/2, oy - oh/2), ow, oh, color="gray", alpha=0.5)
    ax.add_patch(obstacle_rect)

# Initial vehicle representation (this will be updated during the animation)
vehicle_rect = Rectangle((0 - vehicle_size[0]/2, 0 - vehicle_size[1]/2), vehicle_size[0], vehicle_size[1], fill=False, color="blue")
ax.add_patch(vehicle_rect)

# Line object to show the trajectory
trajectory_line, = ax.plot([], [], 'b-', lw=2)

# Initialization function for the animation
def init():
    trajectory_line.set_data([], [])
    return trajectory_line, vehicle_rect,

# Animation update function
def update(frame):
    # Updating the vehicle's position
    x, y = x_trajectory[frame], y_trajectory[frame]
    vehicle_rect.set_xy((x - vehicle_size[0]/2, y - vehicle_size[1]/2))
    
    # Updating the trajectory line
    trajectory_line.set_data(x_trajectory[:frame+1], y_trajectory[:frame+1])
    return trajectory_line, vehicle_rect,

# Creating the animation
ani = FuncAnimation(fig, update, frames=range(len(x_trajectory)), init_func=init, blit=True, interval=100)

plt.show()



