import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 100
delta_t = 1
sigma_0 = 1
x_0 = 0
lambda_var = 10
t0 = 100
T_tot = 2 * t0  # Total simulation time for task B
N_steps = int(T_tot / delta_t)
box_min = -L / 2
box_max = L / 2


def calculate_sigma(x):
    return sigma_0 * ((2 / np.pi) * np.arctan(2 * lambda_var * x / L) + 1)


def d_sigma_dx(x):
    """Calculate dσ(x)/dx according to the given formula."""
    numerator = 4 * sigma_0 * lambda_var
    denominator = np.pi * L * (1 + (2 * lambda_var * x / L) ** 2)
    return numerator / denominator


def s(x):
    """Calculate s(x) = σ(x) * dσ(x)/dx (spurious drift term)."""
    return calculate_sigma(x) * d_sigma_dx(x)


# Task A: Calculate and plot s(x) for x in [-L/2, L/2]
x_vals = np.linspace(box_min, box_max, 500)
s_vals = s(x_vals)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, s_vals, label=r"$s(x) = \sigma(x) \frac{d\sigma(x)}{dx}$")
plt.xlabel(r"$x$")
plt.ylabel(r"$s(x)$")
plt.title("Spurious Drift $s(x)$ vs Position $x$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("spurious_drift_sx.png", dpi=300)
plt.show()

## Task B Task B Task B Task B Task B Task B Task B Task B Task B Task B Task B Task B

# Simulate trajectories for α=0 (Ito) and α=1 (anti-Ito)
np.random.seed(42)  # For reproducibility
w = np.random.normal(0, 1, N_steps)  # Same noise for both trajectories

x_ito = np.zeros(N_steps + 1)
x_anti = np.zeros(N_steps + 1)
x_ito[0] = x_0
x_anti[0] = x_0

s1_t = np.zeros(N_steps + 1)  # Spurious drift along anti-Ito trajectory

for i in range(N_steps):
    # Ito (α=0)
    sigma_i_ito = calculate_sigma(x_ito[i])
    x_ito[i + 1] = x_ito[i] + sigma_i_ito * np.sqrt(delta_t) * w[i]
    # Reflective boundaries
    if x_ito[i + 1] < box_min:
        x_ito[i + 1] = box_min + (box_min - x_ito[i + 1])
    elif x_ito[i + 1] > box_max:
        x_ito[i + 1] = box_max - (x_ito[i + 1] - box_max)

    # Anti-Ito (α=1)
    sigma_i_anti = calculate_sigma(x_anti[i])
    drift_anti = sigma_i_anti * d_sigma_dx(x_anti[i]) * delta_t
    x_anti[i + 1] = x_anti[i] + drift_anti + sigma_i_anti * np.sqrt(delta_t) * w[i]
    # Reflective boundaries
    if x_anti[i + 1] < box_min:
        x_anti[i + 1] = box_min + (box_min - x_anti[i + 1])
    elif x_anti[i + 1] > box_max:
        x_anti[i + 1] = box_max - (x_anti[i + 1] - box_max)
    # Store spurious drift for anti-Ito trajectory
    s1_t[i + 1] = drift_anti

# Time array
times = np.arange(N_steps + 1) * delta_t

# Plot trajectories
plt.figure(figsize=(8, 5))
plt.plot(times, x_ito, label="Ito (α=0)")
plt.plot(times, x_anti, label="Anti-Ito (α=1)")
plt.xlabel("Time")
plt.ylabel("Position x")
plt.title("Trajectories: Ito vs Anti-Ito")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectories_ito_antiito.png", dpi=300)
plt.show()

# Plot spurious drift along anti-Ito trajectory
plt.figure(figsize=(8, 5))
plt.plot(times, s1_t, label=r"Spurious drift $s_1(t)$ (anti-Ito)")
plt.xlabel("Time")
plt.ylabel(r"$s_1(t)$")
plt.title("Spurious Drift Along Anti-Ito Trajectory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("spurious_drift_trajectory.png", dpi=300)
plt.show()


# Task c

# Parameters for Task C
N_realizations = 100000
t0_list = [t0, 5 * t0, 10 * t0, 25 * t0, 50 * t0, 100 * t0]
final_positions = {t: np.zeros(N_realizations) for t in t0_list}

for idx, T in enumerate(t0_list):
    N_steps_c = int(T / delta_t)
    for n in range(N_realizations):
        x = x_0
        w = np.random.normal(0, 1, N_steps_c)
        for i in range(N_steps_c):
            sigma_x = calculate_sigma(x)
            drift = sigma_x * d_sigma_dx(x) * delta_t
            x = x + drift + sigma_x * np.sqrt(delta_t) * w[i]
            # Reflective boundaries
            if x < box_min:
                x = box_min + (box_min - x)
            elif x > box_max:
                x = box_max - (x - box_max)
        final_positions[T][n] = x

    # Plot histogram for this time
    plt.figure(figsize=(8, 5))
    plt.hist(
        final_positions[T], bins=100, range=(box_min, box_max), density=True, alpha=0.7
    )
    plt.xlabel("Final Position x")
    plt.ylabel("Probability Density")
    plt.title(f"Distribution of Final Position after T = {T}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"final_position_distribution_T{T}.png", dpi=300)
    plt.show()
