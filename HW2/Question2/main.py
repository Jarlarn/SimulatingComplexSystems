import numpy as np
import time  # Import time module for tracking elapsed time

# Parameters
R = 1e-6
kb = 1.380e-23
eta = 1e-3
T = 300
k = 1e-6
delta_t = 2e-3
N_ensemble = 100  # 100


gamma = 6 * np.pi * eta * R
D = kb * T / gamma
tau = gamma / k
T_tot = 360  # 360
time_steps = int(T_tot / delta_t)
times = np.linspace(0, T_tot, time_steps)


def generate_noise():
    return np.random.normal(loc=0, scale=1)


# Function to simulate a single trajectory
def simulate_trajectory():
    x, y = 0.0, 0.0  # Start at x = 0, y = 0
    x_vals, y_vals = [x], [y]

    for t in times[1:]:
        # Update positions using Eq. 6 (Langevin dynamics)
        x += -k * x / gamma * delta_t + np.sqrt(2 * D * delta_t) * generate_noise()
        y += -k * y / gamma * delta_t + np.sqrt(2 * D * delta_t) * generate_noise()
        x_vals.append(x)
        y_vals.append(y)

    return np.array(x_vals), np.array(y_vals)


# Simulate ensemble trajectories
ensemble_x = np.zeros((N_ensemble, time_steps))
ensemble_y = np.zeros((N_ensemble, time_steps))

for i in range(N_ensemble):
    x_vals, y_vals = simulate_trajectory()
    ensemble_x[i, :] = x_vals
    ensemble_y[i, :] = y_vals


# Function to calculate EMSD
def calculate_emsd(ensemble_x, ensemble_y, times):
    emsd = []
    for dt in range(1, len(times)):
        squared_displacements = []
        for i in range(N_ensemble):
            dx = ensemble_x[i, dt:] - ensemble_x[i, :-dt]
            dy = ensemble_y[i, dt:] - ensemble_y[i, :-dt]
            squared_displacements.append(np.mean(dx**2 + dy**2))
        emsd.append(np.mean(squared_displacements))
    return np.array(emsd)


# Function to calculate TMSD
def calculate_tmsd(trajectory_x, trajectory_y, times):
    tmsd = []
    total_steps = len(times) - 1  # Total number of time differences
    start_time = time.time()  # Record the start time

    for dt in range(1, len(times)):
        squared_displacements = []
        for t in range(len(times) - dt):
            dx = trajectory_x[t + dt] - trajectory_x[t]
            dy = trajectory_y[t + dt] - trajectory_y[t]
            squared_displacements.append(dx**2 + dy**2)
        tmsd.append(np.mean(squared_displacements))

        # Progress indicator with time estimation
        if dt % 100 == 0 or dt == total_steps:  # Update every 100 steps or at the end
            elapsed_time = time.time() - start_time
            progress = dt / total_steps
            estimated_total_time = elapsed_time / progress
            time_left = estimated_total_time - elapsed_time
            print(
                f"TMSD Calculation Progress: {progress * 100:.2f}% | Time Left: {time_left:.2f} seconds"
            )

    return np.array(tmsd)


# Calculate EMSD
emsd = calculate_emsd(ensemble_x, ensemble_y, times)

# Calculate TMSD for one trajectory (e.g., the first trajectory)
tmsd = calculate_tmsd(ensemble_x[0, :], ensemble_y[0, :], times)

# Time differences for plotting
time_differences = times[1:]

# Plot ensemble averages
import matplotlib.pyplot as plt

# Plot EMSD and TMSD in a log-log plot
plt.figure(figsize=(8, 5))
plt.loglog(time_differences, emsd, label="EMSD", marker="o")
plt.loglog(time_differences, tmsd, label="TMSD", marker="s")
plt.xlabel("Time Difference (s)")
plt.ylabel("Mean Squared Displacement (m²)")
plt.title("EMSD and TMSD as a Function of Time Difference (N = 100, T_tot = 360)")
plt.legend()
plt.tight_layout()
plt.savefig("EMSD_TMSD_LogLog_360.png")
plt.show()
