import numpy as np
import matplotlib.pyplot as plt

# Constants
num_tests = 500
base_wind_speeds = np.arange(0, 15.1, 0.1)  # Base wind speed from 0 to 15 m/s
gust_range = (0, 5)  # Gusts range from 0 to 5 m/s
dt = 0.01  # Time step for simulation
simulation_time = 10  # seconds
mass = 1.0  # kg (example mass of quadcopter)
gravity = 9.81  # m/s^2
drag_coefficient = 0.1  # Example drag coefficient
effective_area = 0.5  # Effective area facing the wind (m^2)
air_density = 1.225  # kg/m^3

# Placeholder function for control algorithm
def control_algorithm(state):
    """
    Placeholder control algorithm for UAV dynamics.
    Outputs thrust values for left and right rotors.
    """
    F1, F2 = 0.0, 0.0  # Example: No control implemented yet
    return F1, F2

# Enhanced dynamics simulation function
def simulate_with_positions(base_wind, gust_amplitude, initial_x, initial_z, num_tests, dt, simulation_time):
    landing_distances = []
    for _ in range(num_tests):
        x, z = initial_x, initial_z  # Initial position
        vx, vz = 0.0, 0.0  # Initial velocity
        time = 0.0
        while time < simulation_time:
            # Wind velocity components (constant base wind + gust)
            gust = np.random.uniform(-gust_amplitude, gust_amplitude)
            wind_speed = base_wind + gust

            # Relative velocity
            u_rel = vx - wind_speed
            w_rel = vz

            # Aerodynamic drag forces
            f_x_drag = -0.5 * air_density * drag_coefficient * effective_area * u_rel * abs(u_rel)
            f_z_drag = -0.5 * air_density * drag_coefficient * effective_area * w_rel * abs(w_rel)

            # Control forces (currently zero)
            F1, F2 = control_algorithm([x, z, vx, vz])
            total_thrust = F1 + F2

            # Net forces
            f_x = f_x_drag
            f_z = f_z_drag + total_thrust - mass * gravity

            # Update accelerations
            ax = f_x / mass
            az = f_z / mass

            # Update velocities
            vx += ax * dt
            vz += az * dt

            # Update positions
            x += vx * dt
            z += vz * dt

            # Stop if the quadcopter reaches the ground
            if z <= 0:
                break

            time += dt

        landing_distances.append((x, z))  # Record landing position
    return landing_distances

# Simulation settings
initial_heights = np.arange(3, 101, 1)  # Mode 1: Vary initial z from 3m to 100m
initial_x_positions = np.arange(-20, 21, 1)  # Mode 2: Vary initial x from -20m to 20m
num_tests = 5  # Reduced simulation count

# Collect results for varying z positions (Mode 1)
results_varying_z = {}
for initial_x in [0]:  # Test with x fixed at 0 for varying z
    results_varying_z[initial_x] = []
    for initial_z in initial_heights:
        distances = simulate_with_positions(
            base_wind=5, gust_amplitude=2,  # Example base wind and gust amplitude
            initial_x=initial_x, initial_z=initial_z,
            num_tests=num_tests, dt=dt, simulation_time=simulation_time
        )
        results_varying_z[initial_x].append(distances)

# Collect results for varying x positions (Mode 2)
results_varying_x = {}
for initial_z in [10]:  # Test with z fixed at 10m for varying x
    results_varying_x[initial_z] = []
    for initial_x in initial_x_positions:
        distances = simulate_with_positions(
            base_wind=5, gust_amplitude=2,  # Example base wind and gust amplitude
            initial_x=initial_x, initial_z=initial_z,
            num_tests=num_tests, dt=dt, simulation_time=simulation_time
        )
        results_varying_x[initial_z].append(distances)

# Mode 1: Varying z (initial height) with x fixed at 0
boxplot_data_varying_z = [
    [np.linalg.norm(pos) for sim in results for pos in sim]
    for results in results_varying_z[0]
]

# Mode 2: Varying x (initial horizontal position) with z fixed at 10
boxplot_data_varying_x = [
    [np.linalg.norm(pos) for sim in results for pos in sim]
    for results in results_varying_x[10]
]

# Plot boxplot for Mode 1: Varying z
plt.figure(figsize=(14, 8))
plt.boxplot(
    boxplot_data_varying_z,
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
plt.title("Reachability Analysis: Boxplot for Varying Initial Height (z_0)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot boxplot for Mode 2: Varying x
plt.figure(figsize=(14, 8))
plt.boxplot(
    boxplot_data_varying_x,
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
plt.title("Reachability Analysis: Boxplot for Varying Initial Horizontal Position (x_0)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
