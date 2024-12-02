import numpy as np
import matplotlib.pyplot as plt

# Constants
num_tests = 5  # Reduced simulation count for faster results
dt = 0.01  # Time step for simulation
simulation_time = 50  # seconds
mass = 1.5  # kg (example mass of quadcopter)
gravity = 9.81  # m/s^2
drag_coefficient = 1  # Example drag coefficient
effective_area = 0.03  # Effective area facing the wind (m^2)
air_density = 1.225  # kg/m^3
moment_of_inertia = 0.03  # Example moment of inertia (kg*m^2)
l = 0.2  # length of the UAV


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
    landing_positions = []
    for _ in range(num_tests):
        x, z, theta = initial_x, initial_z, 0.0  # Initial position and orientation
        vx, vz, omega = 0.0, 0.0, 0.0  # Initial velocities and angular velocity
        time = 0.0

        while time < simulation_time:
            gust_horizontal = np.random.uniform(-gust_amplitude, gust_amplitude)
            gust_vertical = 0.0  # Placeholder: vertical wind set to zero
            wind_horizontal = base_wind + gust_horizontal
            wind_vertical = gust_vertical

            u_rel = vx - wind_horizontal  # Relative horizontal velocity
            w_rel = vz - wind_vertical  # Relative vertical velocity

            effective_area_x = effective_area * abs(np.cos(theta))  # Horizontal projected area
            effective_area_z = effective_area * abs(np.sin(theta))  # Vertical projected area

            f_x_drag = -0.5 * air_density * drag_coefficient * effective_area_x * u_rel * abs(u_rel)
            f_z_drag = -0.5 * air_density * drag_coefficient * effective_area_z * w_rel * abs(w_rel)

            F1, F2 = control_algorithm([x, z, vx, vz, theta, omega])
            total_thrust = F1 + F2
            torque = (F2 - F1) * l

            f_x = f_x_drag
            f_z = f_z_drag + total_thrust - mass * gravity

            alpha = torque / moment_of_inertia  # Angular acceleration
            omega += alpha * dt
            theta += omega * dt  # Update angle

            ax = f_x / mass
            az = f_z / mass

            vx += ax * dt
            vz += az * dt

            x += vx * dt
            z += vz * dt

            if z <= 0:
                break

            time += dt

        # Ensure positions are appended as tuples (x, z, theta)
        landing_positions.append((x, z, theta))
    return landing_positions


# Simulation settings
initial_heights = np.arange(3, 101, 1)  # Mode 1: Vary initial z from 3m to 100m
initial_x_positions = np.arange(-20, 21, 1)  # Mode 2: Vary initial x from -20m to 20m
# Wind range settings
base_wind_speeds = np.arange(5, 10.5, 0.5)  # Base wind speeds from 5 to 10 m/s
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
