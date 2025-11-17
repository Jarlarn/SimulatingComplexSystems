import numpy as np
import time  # Import time module for tracking elapsed time

# Parameters
R = 1e-6
kb = 1.380e-23
eta = 1e-3
T = 300
k = 1e-6
delta_t = 2e-3
N_ensemble = 10  # 10/100


gamma = 6 * np.pi * eta * R
D = kb * T / gamma
tau = gamma / k
T_tot = 60  # 60/360
time_steps = int(T_tot / delta_t)
times = np.linspace(0, T_tot, time_steps)


def generate_noise():
    return np.random.normal(loc=0, scale=1)


# Function to simulate a single trajectory with equilibration
def simulate_trajectory_with_equilibration():
    # Equilibration phase
    t0 = 5 * tau  # At least a few τ
    equilibration_steps = int(t0 / delta_t)

    x, y = 0.0, 0.0
    for _ in range(equilibration_steps):
        x += -k * x / gamma * delta_t + np.sqrt(2 * D * delta_t) * generate_noise()
        y += -k * y / gamma * delta_t + np.sqrt(2 * D * delta_t) * generate_noise()

    # Record x0, y0 after equilibration
    x_vals, y_vals = [x], [y]

    # Generate trajectory from x0, y0
    for t in times[1:]:
        x += -k * x / gamma * delta_t + np.sqrt(2 * D * delta_t) * generate_noise()
        y += -k * y / gamma * delta_t + np.sqrt(2 * D * delta_t) * generate_noise()
        x_vals.append(x)
        y_vals.append(y)

    return np.array(x_vals), np.array(y_vals)


# Simulate ensemble trajectories
ensemble_x = np.zeros((N_ensemble, time_steps))
ensemble_y = np.zeros((N_ensemble, time_steps))

for i in range(N_ensemble):
    x_vals, y_vals = simulate_trajectory_with_equilibration()
    ensemble_x[i, :] = x_vals
    ensemble_y[i, :] = y_vals


def fast_tmsd(x, y):
    """
    Compute the time-averaged MSD using FFT convolution.
    Works in O(N log N), extremely fast.
    """

    N = len(x)

    # Compute squared magnitudes
    r2 = x * x + y * y
    r2_sum = np.zeros(N)

    # Autocorrelation via FFT
    def autocorr_fft(a):
        n = 1 << (2 * N - 1).bit_length()  # next power of 2
        A = np.fft.fft(a, n=n)
        C = A * np.conjugate(A)
        c = np.fft.ifft(C).real
        return c[:N]

    Cx = autocorr_fft(x)
    Cy = autocorr_fft(y)

    # Number of contributing terms for each lag
    norm = np.arange(N, 0, -1)

    # MSD formula
    msd_x = (r2.sum() - 2 * Cx) / norm + r2[:N] / norm
    msd_y = (r2.sum() - 2 * Cy) / norm + r2[:N] / norm

    return msd_x + msd_y


def fast_emsd(ensemble_x, ensemble_y):
    """
    Vectorized ensemble-averaged MSD.
    Much faster than looping over all dt.
    """

    N_ens, N = ensemble_x.shape
    msd = np.zeros(N - 1)

    for x, y in zip(ensemble_x, ensemble_y):
        msd += fast_tmsd(x, y)[1:]

    return msd / N_ens


emsd = fast_emsd(ensemble_x, ensemble_y)
tmsd = fast_tmsd(ensemble_x[0], ensemble_y[0])[1:]

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
plt.title(
    f"EMSD and TMSD as a Function of Time Difference (N = {N_ensemble}, T_tot = {T_tot})"
)
plt.legend()
plt.tight_layout()
plt.savefig(f"EMSD_TMSD_LogLog_{N_ensemble}.png", dpi=600)
plt.show()
