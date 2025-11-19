import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
kb = 1.380e-23  # Boltzmann constant (J/K)
T = 300  # Temperature (K)
eta = 1e-3  # Fluid viscosity (Pa·s)
R = 1e-6  # Particle radius (m)
k = 1e-6  # Trap stiffness (N/m)
delta_t = 2e-3  # Time step (s)

# Simulation parameters for Task B
T_tot = 60  # Total simulation time (s)
N_ensemble = 10  # Number of trajectories

# Derived parameters
gamma = 6 * np.pi * eta * R
D = kb * T / gamma
tau = gamma / k

# Time arrays
time_steps = int(T_tot / delta_t)
times = np.linspace(0, T_tot, time_steps)


def logspaced_lags(N, num_lags=200):
    lags = np.unique(np.logspace(0, np.log10(N - 1), num=num_lags, dtype=int))
    return lags[lags > 0]


def simulate_equilibrated_start():
    """Equilibrate a single trajectory and return the final position."""
    t0 = 5 * tau
    equil_steps = int(t0 / delta_t)
    x, y = 0.0, 0.0
    for _ in range(equil_steps):
        x += -(k / gamma) * x * delta_t + np.sqrt(2 * D * delta_t) * np.random.normal()
        y += -(k / gamma) * y * delta_t + np.sqrt(2 * D * delta_t) * np.random.normal()
    return x, y


def simulate_trajectory(x0, y0, steps):
    """Simulate a single trajectory starting from (x0, y0)."""
    x = np.zeros(steps)
    y = np.zeros(steps)
    x[0], y[0] = x0, y0
    for t in range(1, steps):
        x[t] = (
            x[t - 1]
            - (k / gamma) * x[t - 1] * delta_t
            + np.sqrt(2 * D * delta_t) * np.random.normal()
        )
        y[t] = (
            y[t - 1]
            - (k / gamma) * y[t - 1] * delta_t
            + np.sqrt(2 * D * delta_t) * np.random.normal()
        )
    return x, y


def calculate_emsd(ensemble_x, ensemble_y, num_lags=200):
    N_ens, N = ensemble_x.shape
    lags = logspaced_lags(N, num_lags)
    msd = np.empty(len(lags))
    for i, lag in enumerate(lags):
        dx = ensemble_x[:, lag:] - ensemble_x[:, :-lag]
        dy = ensemble_y[:, lag:] - ensemble_y[:, :-lag]
        msd[i] = np.mean(dx**2 + dy**2)
    return lags, msd


def calculate_tmsd(x, y, num_lags=200):
    N = len(x)
    lags = logspaced_lags(N, num_lags)
    msd = np.empty(len(lags))
    for i, lag in enumerate(lags):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        msd[i] = np.mean(dx**2 + dy**2)
    return lags, msd


# --- Simulate ensemble trajectories for eMSD ---
ensemble_x = np.zeros((N_ensemble, time_steps))
ensemble_y = np.zeros((N_ensemble, time_steps))
for j in range(N_ensemble):
    x0, y0 = simulate_equilibrated_start()  # Equilibrate
    x_traj, y_traj = simulate_trajectory(x0, y0, time_steps)  # Main trajectory
    ensemble_x[j] = x_traj
    ensemble_y[j] = y_traj

# --- Simulate single trajectory for tMSD ---
x0, y0 = 0, 0
x_single, y_single = simulate_trajectory(x0, y0, time_steps)

# --- Calculate MSDs ---
lags_emsd, emsd = calculate_emsd(ensemble_x, ensemble_y)
lags_tmsd, tmsd = calculate_tmsd(x_single, y_single)

# --- Convert lag indices to time differences ---
time_diffs_emsd = times[lags_emsd]
time_diffs_tmsd = times[lags_tmsd]

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.loglog(
    time_diffs_emsd,
    emsd,
    label="eMSD (ensemble)",
    color="orange",
    marker="o",
    markersize=3,
)
plt.loglog(
    time_diffs_tmsd, tmsd, label="tMSD (single)", color="blue", marker="s", markersize=3
)
plt.xlabel("Δt (s)")
plt.ylabel("MSD (m²)")
plt.title(f"eMSD and tMSD vs Time Difference (N={N_ensemble}, T_tot={T_tot}s)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"EMSD_TMSD_TaskB_{N_ensemble}.png", dpi=600)
plt.show()

print("Done! Plot saved.")
