import numpy as np
import matplotlib.pyplot as plt
from models.quadcopter import QuadCopterWind
from plan_dubin import plan_dubins_path
import casadi

# Constants
num_tests = 5  # Reduced simulation count for faster results
dt = 0.1  # Time step for simulation
simulation_time = 50  # seconds
mass = 1.5  # kg (example mass of quadcopter)
gravity = 9.81  # m/s^2
drag_coefficient = 1  # Example drag coefficient
effective_area = 0.03  # Effective area facing the wind (m^2)
air_density = 1.225  # kg/m^3
moment_of_inertia = 0.03  # Example moment of inertia (kg*m^2)
l = 0.4  # length of the UAV
Q0 = [7, 7, 7, 0, 0, 0] # No penalty on velocities
R0 = 2
Q1 = [10, 10, 10, 0, 0, 0] # No penalty on velocities
R1 = 0.5
N = 8
r_max0 = 1 # Max angular velocity
v_max0 = 1 # Max linear velocity
r_max1 = 2 # Max angular velocity
v_max1 = 2 # Max linear velocity


# Placeholder function for control algorithm
def control_algorithm(state):
    """
    Placeholder control algorithm for UAV dynamics.
    Outputs thrust values for left and right rotors.
    """

    F1, F2 = 0.0, 0.0  # Example: No control implemented yet
    return F1, F2


# Enhanced dynamics simulation function with rotation and vertical wind
def simulate_with_rotation_and_wind(
        base_wind, gust_amplitude, initial_x, initial_z, num_tests, dt, simulation_time
):
    # Constants
    landing_positions = []
    for _ in range(num_tests):
        x, z, theta = initial_x, initial_z, 0.0  # Initial position and orientation
        t0 = 0

        init_state = [x, z, theta, 0, 0, 0]
        path_x, path_y, path_yaw, _, _ = plan_dubins_path(x, z, theta, 0, 0, 0, r_max0, step_size=v_max0*dt)
        ref_states = np.array([path_x, path_y, -path_yaw]).T
        ref_states = np.concatenate([ref_states, np.zeros(ref_states.shape)], axis=-1)

        gust_horizontal = np.random.uniform(-gust_amplitude, gust_amplitude)
        gust_vertical = 0.0  # Placeholder: vertical wind set to zero
        wind_horizontal = base_wind + gust_horizontal
        wind_vertical = gust_vertical
        wind = [wind_horizontal, wind_vertical]
        qc = QuadCopterWind(init_state, Q0, R0, Q1, R1, N, v_max0, r_max0, v_max1, r_max1, wind)
        state_0 = casadi.DM(init_state)
        u0 = casadi.DM.zeros((qc.n_controls, N))
        X0 = casadi.repmat(state_0, 1, N + 1)
        cat_states = np.array(X0.full())
        cat_controls = np.array(u0[:, 0].full())
        for i in range(len(ref_states)):
            u, X_pred = qc.solve(X0, u0, ref_states, i)
            
            cat_states = np.dstack((cat_states, np.array(X_pred.full())))
            cat_controls = np.dstack((cat_controls, np.array(u[:, 0].full())))
            
            t0, X0, u0 = qc.shift_timestep(t0, X_pred, u)
            qc.states = X0[:, 1]
            if qc.states[1] <= 0:
                break

        # Ensure positions are appended as tuples (x, z, theta)
        landing_positions.append(np.array(qc.states))
    return landing_positions


# Simulation settings
initial_heights = np.arange(5, 50, 5)  # Mode 1: Vary initial z from 3m to 100m
initial_x_positions = np.arange(-20, 21, 4)  # Mode 2: Vary initial x from -20m to 20m
# Wind range settings
base_wind_speeds = np.arange(5, 15, 3)  # Base wind speeds from 5 to 10 m/s
gust_range = (0, 5)  # Gust component varies from 0 to 5 m/s

# Collect results for Mode 1: Varying initial heights (z_0)
results_with_rotation_varying_z = {}
for initial_x in [0]:  # Test with x fixed at 0 for varying z
    results_with_rotation_varying_z[initial_x] = []
    for initial_z in initial_heights:
        wind_results = []
        for base_wind in base_wind_speeds:
            positions = simulate_with_rotation_and_wind(
                base_wind=base_wind,
                gust_amplitude=gust_range[1],
                initial_x=initial_x,
                initial_z=initial_z,
                num_tests=num_tests,
                dt=dt,
                simulation_time=simulation_time,
            )
            wind_results.extend(positions)
        results_with_rotation_varying_z[initial_x].append(wind_results)

# Collect results for Mode 2: Varying x positions (horizontal position) with z fixed at 10
results_with_rotation_varying_x = {}
for initial_z in [10]:  # Test with z fixed at 10m for varying x
    results_with_rotation_varying_x[initial_z] = []
    for initial_x in initial_x_positions:
        wind_results = []
        for base_wind in base_wind_speeds:
            positions = simulate_with_rotation_and_wind(
                base_wind=base_wind,
                gust_amplitude=gust_range[1],
                initial_x=initial_x,
                initial_z=initial_z,
                num_tests=num_tests,
                dt=dt,
                simulation_time=simulation_time,
            )
            wind_results.extend(positions)
        results_with_rotation_varying_x[initial_z].append(wind_results)
# Prepare data for Mode 1: Varying initial heights (z_0)
boxplot_data_with_rotation_varying_z = [
    [np.linalg.norm((pos[0], pos[1])) for pos in results]
    for results in results_with_rotation_varying_z[0]
]

# Prepare data for Mode 2: Varying initial horizontal positions (x_0)
boxplot_data_with_rotation_varying_x = [
    [np.linalg.norm((pos[0], pos[1])) for pos in results]
    for results in results_with_rotation_varying_x[10]
]

# Plot Boxplot for Mode 1: Varying Initial Height (z_0)
plt.figure(figsize=(14, 8))
plt.boxplot(
    boxplot_data_with_rotation_varying_z,
    notch=False,
    patch_artist=False,
    showmeans=True,
    meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"},
    flierprops={"marker": "o", "markerfacecolor": "white", "markersize": 4, "linestyle": "none"},
    boxprops={"color": "black"},
    whiskerprops={"color": "black"},
    capprops={"color": "black"}
)
plt.xticks(
    ticks=np.arange(1, len(initial_heights) + 1, step=10),
    labels=[f"{height}" for height in initial_heights[::10]],
    rotation=45
)
plt.xlabel("Initial Height (z_0) [m]")
plt.ylabel("Absolute Landing Distance [m]")
plt.title("Reachability with Rotation: Boxplot for Varying Initial Height (z_0)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Boxplot for Mode 2: Varying Initial Horizontal Position (x_0)
plt.figure(figsize=(14, 8))
plt.boxplot(
    boxplot_data_with_rotation_varying_x,
    notch=False,
    patch_artist=False,
    showmeans=True,
    meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "black"},
    flierprops={"marker": "o", "markerfacecolor": "white", "markersize": 4, "linestyle": "none"},
    boxprops={"color": "black"},
    whiskerprops={"color": "black"},
    capprops={"color": "black"}
)
plt.xticks(
    ticks=np.arange(1, len(initial_x_positions) + 1, step=5),
    labels=[f"{x}" for x in initial_x_positions[::5]],
    rotation=45
)
plt.xlabel("Initial Horizontal Position (x_0) [m]")
plt.ylabel("Absolute Landing Distance [m]")
plt.title("Reachability with Rotation: Boxplot for Varying Initial Horizontal Position (x_0)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
