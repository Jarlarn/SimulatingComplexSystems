import numpy as np
import matplotlib.pyplot as plt

KB = 1.380e-23
T = 300
eta = 1e-3
R = 1e-6
x_left = -0.9 * R
x_right = 0.9 * R
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
    force_left = (
        -trap_depth
        / (TRAP_WAIST**2)
        * (x - x_left)
        * np.exp(-((x - x_left) ** 2) / (2 * TRAP_WAIST**2))
    )
    force_right = (
        -trap_depth
        / (TRAP_WAIST**2)
        * (x - x_right)
        * np.exp(-((x - x_right) ** 2) / (2 * TRAP_WAIST**2))
    )
    return force_left + force_right


def calculate_potential(x, trap_depth):
    potential_left = -trap_depth * np.exp(-((x - x_left) ** 2) / (2 * TRAP_WAIST**2))
    potential_right = -trap_depth * np.exp(-((x - x_right) ** 2) / (2 * TRAP_WAIST**2))

    return potential_left + potential_right


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

trap_depth = 8 * KB * T
esc_times_n = []
for rep in range(Nrep):
    esc_time = simulate_trajectory(trap_depth)
    esc_times_n.append(esc_time)
escape_times.append(esc_times_n)

x_vals = np.linspace(-1.2 * intervall_max, 1.2 * intervall_max, 500)

# Plot U(x) for the constant trap depth
plt.figure(figsize=(8, 5))
trap_depth = 8 * KB * T  # Constant trap depth
U_vals = calculate_potential(x_vals, trap_depth)
plt.plot(x_vals * 1e6, U_vals / KB / T, label=f"Trap Depth = 8 kB T")
plt.xlabel("x (μm)")
plt.ylabel("U(x) / kB T")
plt.title("Potential Energy Profile U(x)")
plt.legend()
plt.tight_layout()
plt.savefig("Potential Energy Profile Constant")
plt.show()

# Plot F(x) for the constant trap depth
plt.figure(figsize=(8, 5))
F_vals = calculate_force(x_vals, trap_depth)
plt.plot(x_vals * 1e6, F_vals / KB / T, label=f"Trap Depth = 8 kB T")
plt.xlabel("x (μm)")
plt.ylabel("F(x) / kB T")
plt.title("Force Profile F(x) for Constant Trap Depth")
plt.legend()
plt.tight_layout()
plt.savefig("Force Profile Constant")
plt.show()

# Plot sample trajectories x(t) for constant trap depth
plt.figure(figsize=(8, 5))
for rep in range(3):  # Simulate 3 independent trajectories
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
        times / 3600, np.array(x_vals) * 1e6, label=f"Trajectory {rep + 1}"
    )  # Convert x to μm and time to hours

plt.xlabel("Time (hours)")
plt.ylabel("x(t) (μm)")
plt.title("Sample Trajectories x(t) for Constant Trap Depth")
plt.legend()
plt.tight_layout()
plt.savefig("Sample Trajectories Constant")
plt.show()


def calculate_transition_frequency(trap_depth):
    x = 0.0
    times = np.arange(0, T_TOT, delta_t)
    transitions = 0
    previous_position = x

    for t in times:
        # Update position
        x = (
            x
            + calculate_force(x, trap_depth) / friction_coeff * delta_t
            + np.sqrt(2 * diffusion_coeff * delta_t) * generate_noise()
        )

        # Check for transition
        if previous_position < 0 and x > 0:  # Left to right
            transitions += 1
        elif previous_position > 0 and x < 0:  # Right to left
            transitions += 1

        previous_position = x

    # Calculate transition frequency
    transition_frequency = transitions / T_TOT
    return transition_frequency


# Calculate transition frequency for constant trap depth
trap_depth = 8 * KB * T  # Constant trap depth
transition_frequency = calculate_transition_frequency(trap_depth)

print(f"Transition Frequency: {transition_frequency:.4f} Hz")
"Transition Frequency: 0.0380 Hz"
