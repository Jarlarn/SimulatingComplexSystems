import numpy as np
import matplotlib.pyplot as plt

KB = 1.380e-23
T = 300
eta = 1e-3
R = 1e-6
TRAP_WAIST = R / 2
delta_t = 2e-2
T_TOT = 3 * 3600
intervall_min = -5 * TRAP_WAIST
intervall_max = 5 * TRAP_WAIST
friction_coeff = 6 * np.pi * eta * R
diffusion_coeff = KB * T / friction_coeff
Nrep = 5
n_values = np.arange(1, 9)  # [1,2,3,4,5,6,7,8]


def generate_noise():
    return np.random.normal(loc=0, scale=1)


def calculate_force(x, trap_depth):
    force = -trap_depth / (TRAP_WAIST**2) * x * np.exp(-(x**2) / (2 * TRAP_WAIST**2))
    return force


def calculate_potential(x, trap_depth):
    return -trap_depth * np.exp(-(x**2) / (2 * TRAP_WAIST**2))


def simulate_trajectory(trap_depth):
    x = 0.0
    times = np.arange(0, T_TOT, delta_t)
    for i, t in enumerate(times):
        x = (
            x
            + calculate_force(x, trap_depth) / friction_coeff * delta_t
            + np.sqrt(2 * diffusion_coeff * delta_t) * generate_noise()
        )
        if x < intervall_min or x > intervall_max:
            return t  # Escape time
    return T_TOT  # Did not escape


# Store escape times for each n
escape_times = []

for n in n_values:
    trap_depth = n * KB * T
    esc_times_n = []
    for rep in range(Nrep):
        esc_time = simulate_trajectory(trap_depth)
        esc_times_n.append(esc_time)
    escape_times.append(esc_times_n)


# Plot U(x) for all trap depths
x_vals = np.linspace(-1.2 * intervall_max, 1.2 * intervall_max, 500)
plt.figure(figsize=(8, 5))
for n in n_values:
    trap_depth = n * KB * T
    U_vals = calculate_potential(x_vals, trap_depth)
    plt.plot(x_vals * 1e6, U_vals / KB / T, label=f"n={n}")
plt.xlabel("x (μm)")
plt.ylabel("U(x) / kB T")
plt.title("Potential Energy Profiles U(x) for Different Trap Depths")
plt.legend()
plt.tight_layout()
plt.savefig("Potential Energy Profiles")
plt.show()

# Plot F(x) for all trap depths
plt.figure(figsize=(8, 5))
for n in n_values:
    trap_depth = n * KB * T
    F_vals = calculate_force(x_vals, trap_depth)
    plt.plot(x_vals * 1e6, F_vals / KB / T, label=f"n={n}")
plt.xlabel("x (μm)")
plt.ylabel("F(x) / kB T")
plt.title("Force Profiles F(x) for Different Trap Depths")
plt.legend()
plt.tight_layout()
plt.savefig("Force Profiles")
plt.show()

# Plot sample trajectories x(t) for different trap depths
plt.figure(figsize=(8, 5))
for n in n_values:
    trap_depth = n * KB * T
    times = np.arange(0, T_TOT, delta_t)
    x_vals = []
    x = 0.0
    for t in times:
        x = (
            x
            + calculate_force(x, trap_depth) / friction_coeff * delta_t
            + np.sqrt(2 * diffusion_coeff * delta_t) * generate_noise()
        )
        x_vals.append(x)
    plt.plot(
        times / 3600, np.array(x_vals) * 1e6, label=f"n={n}"
    )  # Convert x to μm and time to hours

plt.xlabel("Time (hours)")
plt.ylabel("x(t) (μm)")
plt.title("Sample Trajectories x(t) for Different Trap Depths")
plt.legend()
plt.tight_layout()
plt.savefig("Sample Trajectories")
plt.show()

# Optional: Print escape times
for idx, esc_times_n in enumerate(escape_times):
    print(f"n={n_values[idx]}, escape times (s): {esc_times_n}")

# Calculate average and standard deviation of escape times
average_escape_times = []
std_escape_times = []

for esc_times_n in escape_times:
    avg_time = np.mean(esc_times_n)
    std_time = np.std(esc_times_n)
    average_escape_times.append(avg_time)
    std_escape_times.append(std_time)

# Print table of escape times
print("n\tAverage Escape Time (s)\tStandard Deviation (s)")
for idx, n in enumerate(n_values):
    print(f"{n}\t{average_escape_times[idx]:.2f}\t\t\t{std_escape_times[idx]:.2f}")

# Plot average escape times with error bars
plt.figure(figsize=(8, 5))
plt.errorbar(
    n_values,
    np.array(average_escape_times) / 3600,
    yerr=np.array(std_escape_times) / 3600,
    fmt="o",
    capsize=5,
)
plt.xlabel("n")
plt.ylabel("Average Escape Time (hours)")
plt.title("Average Escape Time with Error Bars")
plt.tight_layout()
plt.savefig("Escape Times with Error Bars")
plt.show()
