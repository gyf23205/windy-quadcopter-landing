import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def polynomial_derivative(coeffs, order):
    """Compute the derivative of a polynomial given its coefficients."""
    n = len(coeffs)
    for _ in range(order):
        coeffs = [coeffs[i] * i for i in range(1, n)]
        n -= 1
    return coeffs

def evaluate_polynomial(coeffs, t):
    """Evaluate a polynomial at time t."""
    return sum(c * t**i for i, c in enumerate(coeffs))

def evaluate_polynomial_derivative(coeffs, t, order):
    """Evaluate the derivative of a polynomial at time t."""
    derivative_coeffs = polynomial_derivative(coeffs, order)
    return evaluate_polynomial(derivative_coeffs, t)

def minimum_snap_trajectory_2d(waypoints, times, derivative_order=4):
    """
    Compute a 2D minimum snap trajectory passing through waypoints.
    
    Parameters:
    - waypoints: List of waypoints [(x1, y1), (x2, y2), ...].
    - times: List of time durations for each segment.
    - derivative_order: Order of optimization (4 for snap).
    
    Returns:
    - x_coeffs, y_coeffs: Coefficients for x(t) and y(t).
    """
    n_segments = len(waypoints) - 1
    n_coeffs = 2 * derivative_order  # Number of coefficients per segment
    total_coeffs = n_segments * n_coeffs
    
    # Separate x and y waypoints
    x_waypoints = [wp[0] for wp in waypoints]
    y_waypoints = [wp[1] for wp in waypoints]
    
    def snap_cost(coeffs):
        """Objective function to minimize snap."""
        total_snap = 0
        for i in range(n_segments):
            segment_coeffs = coeffs[i * n_coeffs : (i + 1) * n_coeffs]
            snap_coeffs = polynomial_derivative(segment_coeffs, derivative_order)
            snap_integral = sum(c**2 / (2 * j - 1) for j, c in enumerate(snap_coeffs, start=4))
            total_snap += snap_integral * times[i]
        return total_snap

    def constraints(coeffs, waypoints):
        """Equality constraints for continuity and waypoint matching."""
        cons = []
        for i in range(n_segments):
            segment_coeffs = coeffs[i * n_coeffs : (i + 1) * n_coeffs]
            # Position at segment start and end
            if i == 0:
                cons.append(evaluate_polynomial(segment_coeffs, 0) - waypoints[i])
            cons.append(evaluate_polynomial(segment_coeffs, times[i]) - waypoints[i + 1])
            # Continuity at internal waypoints
            if i < n_segments - 1:
                next_coeffs = coeffs[(i + 1) * n_coeffs : (i + 2) * n_coeffs]
                for d in range(1, derivative_order):  # Derivatives: velocity, acceleration, etc.
                    derivative_coeffs = polynomial_derivative(segment_coeffs, d)
                    next_derivative_coeffs = polynomial_derivative(next_coeffs, d)
                    cons.append(evaluate_polynomial(derivative_coeffs, times[i]) - evaluate_polynomial(next_derivative_coeffs, 0))
        return np.array(cons)

    # Solve for x and y trajectories separately
    def solve_trajectory(waypoints):
        initial_guess = np.zeros(total_coeffs)
        result = minimize(
            snap_cost, 
            initial_guess, 
            constraints={"type": "eq", "fun": lambda coeffs: constraints(coeffs, waypoints)},
            method="SLSQP"
        )
        if not result.success:
            raise ValueError("Optimization failed: " + result.message)
        return np.reshape(result.x, (n_segments, n_coeffs))

    x_coeffs = solve_trajectory(x_waypoints)
    y_coeffs = solve_trajectory(y_waypoints)

    return x_coeffs, y_coeffs


def minimum_snap_trajectory_2d_with_theta(waypoints, times, derivative_order=4):
    """
    Compute a 2D minimum snap trajectory with orientation angle references.
    
    Parameters:
    - waypoints: List of waypoints [(x1, y1), (x2, y2), ...].
    - times: List of time durations for each segment.
    - derivative_order: Order of optimization (4 for snap).
    
    Returns:
    - x_coeffs, y_coeffs: Coefficients for x(t) and y(t).
    - theta_refs: List of reference angles (\theta) over the trajectory.
    """
    x_coeffs, y_coeffs = minimum_snap_trajectory_2d(waypoints, times, derivative_order)

    # Compute theta references
    time_samples = np.linspace(0, sum(times), 100)
    theta_refs = []
    for t in time_samples:
        current_segment = next(i for i, t_end in enumerate(np.cumsum(times)) if t <= t_end)
        segment_time = t - (np.cumsum(times)[current_segment - 1] if current_segment > 0 else 0)
        
        # Compute accelerations
        ax = evaluate_polynomial_derivative(x_coeffs[current_segment], segment_time, 2)
        ay = evaluate_polynomial_derivative(y_coeffs[current_segment], segment_time, 2)
        
        # Compute theta reference
        theta_refs.append(np.arctan2(ay, ax))

    return x_coeffs, y_coeffs, theta_refs, time_samples


if __name__=='__main__':
    # Define 2D waypoints and segment durations
    waypoints = [(0, 0), (2, 2), (5, 4), (6, 6)]  # Waypoints in 2D
    times = [2, 2, 2]  # Equal time for each segment

    # Generate 2D trajectory with theta references
    x_coeffs, y_coeffs, theta_refs, time_samples = minimum_snap_trajectory_2d_with_theta(waypoints, times)

    # Plot trajectory
    trajectory_points = []
    for t in time_samples:
        current_segment = next(i for i, t_end in enumerate(np.cumsum(times)) if t <= t_end)
        segment_time = t - (np.cumsum(times)[current_segment - 1] if current_segment > 0 else 0)
        x = evaluate_polynomial(x_coeffs[current_segment], segment_time)
        y = evaluate_polynomial(y_coeffs[current_segment], segment_time)
        trajectory_points.append((x, y))

    trajectory_points = np.array(trajectory_points)

    plt.figure(figsize=(12, 6))

    # Plot 2D trajectory
    plt.subplot(1, 2, 1)
    plt.plot([wp[0] for wp in waypoints], [wp[1] for wp in waypoints], "ro", label="Waypoints")
    plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], "b-", label="Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Trajectory")
    plt.legend()
    plt.grid()

    # Plot theta references
    plt.subplot(1, 2, 2)
    plt.plot(time_samples, np.rad2deg(theta_refs), "g-", label="Theta (deg)")
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (degrees)")
    plt.title("Theta References")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
