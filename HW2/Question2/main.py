import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Physical parameters
R = 1e-6  # Particle radius (m)
kb = 1.380e-23  # Boltzmann constant (J/K)
eta = 1e-3  # Fluid viscosity (Pa·s)
T = 300  # Temperature (K)
k = 1e-6  # Trap stiffness (N/m)
delta_t = 2e-3  # Time step (s)

# Simulation parameters
N_ensemble = 100  # Number of trajectories
T_tot = 360  # Total simulation time (s)
time_steps = int(T_tot / delta_t)
times = np.linspace(0, T_tot, time_steps)

# Derived parameters
gamma = 6 * np.pi * eta * R  # Friction coefficient
D = kb * T / gamma  # Diffusion coefficient
tau = gamma / k  # Trap characteristic time


def simulate_ensemble_trajectories(N_ensemble, time_steps, equilibration_steps):
    """Simulate N_ensemble independent particle trajectories with equilibration."""
    total_steps = time_steps + equilibration_steps

    # Pre-generate all random noise
    noise_x = np.random.normal(size=(N_ensemble, total_steps))
    noise_y = np.random.normal(size=(N_ensemble, total_steps))

    # Initialize position arrays
    x = np.zeros((N_ensemble, total_steps))
    y = np.zeros((N_ensemble, total_steps))

    # Simulate trajectories using Eq. 6
    for t in range(1, total_steps):
        if t % 20000 == 0:
            print(f"Simulation progress: {100 * t / total_steps:.1f}%")

        # Update x and y positions
        x[:, t] = (
            x[:, t - 1]
            - (k / gamma) * x[:, t - 1] * delta_t
            + np.sqrt(2 * D * delta_t) * noise_x[:, t]
        )
        y[:, t] = (
            y[:, t - 1]
            - (k / gamma) * y[:, t - 1] * delta_t
            + np.sqrt(2 * D * delta_t) * noise_y[:, t]
        )

    # Return only the equilibrated portion
    return x[:, equilibration_steps:], y[:, equilibration_steps:]


def logspaced_lags(N, num_lags=200):
    """Generate logarithmically spaced lag values for efficient MSD calculation."""
    lags = np.unique(np.logspace(0, np.log10(N - 1), num=num_lags, dtype=int))
    return lags[lags > 0]  # Remove zero lag


def calculate_emsd(ensemble_x, ensemble_y, num_lags=200):
    """Calculate ensemble-averaged MSD (eMSD) using log-spaced lags."""
    N_ens, N = ensemble_x.shape
    lags = logspaced_lags(N, num_lags)
    msd = np.empty(len(lags))
    start = time.time()

    for i, lag in enumerate(lags):
        # Calculate displacement for all particles at this lag
        dx = ensemble_x[:, lag:] - ensemble_x[:, :-lag]
        dy = ensemble_y[:, lag:] - ensemble_y[:, :-lag]
        msd[i] = np.mean(dx**2 + dy**2)

        # Print progress every 10%
        if i % max(1, len(lags) // 10) == 0 or i == len(lags) - 1:
            elapsed = time.time() - start
            percent = 100 * (i + 1) / len(lags)
            est_left = (elapsed / percent * 100 - elapsed) if percent > 0 else 0
            print(f"eMSD progress: {percent:.1f}% | Time left: {est_left:.1f}s")
            sys.stdout.flush()

    return lags, msd


def calculate_tmsd(x, y, num_lags=200):
    """Calculate time-averaged MSD (tMSD) for a single trajectory using log-spaced lags."""
    N = len(x)
    lags = logspaced_lags(N, num_lags)
    msd = np.empty(len(lags))
    start = time.time()

    for i, lag in enumerate(lags):
        # Calculate displacement at this lag
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        msd[i] = np.mean(dx**2 + dy**2)

        # Print progress every 10%
        if i % max(1, len(lags) // 10) == 0 or i == len(lags) - 1:
            elapsed = time.time() - start
            percent = 100 * (i + 1) / len(lags)
            est_left = (elapsed / percent * 100 - elapsed) if percent > 0 else 0
            print(f"tMSD progress: {percent:.1f}% | Time left: {est_left:.1f}s")
            sys.stdout.flush()

    return lags, msd


# Main simulation
print("Starting simulation with equilibration...")
t0 = 5 * tau  # Equilibration time (5 trap characteristic times)
equilibration_steps = int(t0 / delta_t)
ensemble_x, ensemble_y = simulate_ensemble_trajectories(
    N_ensemble, time_steps, equilibration_steps
)

print("\nCalculating eMSD...")
lags_emsd, emsd = calculate_emsd(ensemble_x, ensemble_y)

print("\nCalculating tMSD...")
lags_tmsd, tmsd = calculate_tmsd(ensemble_x[0], ensemble_y[0])

# Convert lag indices to actual time differences
time_differences_emsd = times[lags_emsd]
time_differences_tmsd = times[lags_tmsd]

# Plot results
plt.figure(figsize=(8, 5))
plt.loglog(
    time_differences_emsd, emsd, label="eMSD", color="orange", marker="o", markersize=3
)
plt.loglog(
    time_differences_tmsd, tmsd, label="tMSD", color="blue", marker="s", markersize=3
)
plt.xlabel("Δt (s)")
plt.ylabel("MSD (m²)")
plt.title(f"eMSD and tMSD vs Time Difference (N={N_ensemble}, T_tot={T_tot}s)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"EMSD_TMSD_LogLog_{N_ensemble}.png", dpi=600)
plt.show()

print("\nDone! Plot saved.")
