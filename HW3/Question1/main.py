import numpy as np
import matplotlib.pyplot as plt


# Parameters
delta_t = 20e-3
v = 50e-6
kb = 1.380e-23
T = 300
eta = 1e-3
R = 1e-6
T_tot = 10000

D = (kb * T) / (6 * np.pi * eta * R)
D_R = (kb * T) / (8 * np.pi * eta * R**3)
tau_r = 1 / D_R  # 6.07

# Starting conditions
x0 = 0
y0 = 0
theta0 = 0
omega = np.array([0, 1 / 2, 1, 3 / 2]) * np.pi


def generate_noise():
    return np.random.normal(loc=0, scale=1)


def update_x(current_x, current_theta, w_x):
    x_next = (
        current_x + v * np.cos(current_theta) * delta_t + np.sqrt(2 * D * delta_t) * w_x
    )
    return x_next


def update_y(current_y, current_theta, w_y):
    y_next = (
        current_y + v * np.sin(current_theta) * delta_t + np.sqrt(2 * D * delta_t) * w_y
    )
    return y_next


def update_theta(current_theta, omega, w_theta):
    next_theta = current_theta + omega * delta_t + np.sqrt(2 * D_R * delta_t) * w_theta
    return next_theta


# Simulation setup
num_steps = int(T_tot / delta_t)
time = np.arange(num_steps) * delta_t

# Generate all noise arrays once (for all steps)
w_x = np.random.normal(0, 1, num_steps)
w_y = np.random.normal(0, 1, num_steps)
w_theta = np.random.normal(0, 1, num_steps)

trajectories = []
theta_trajectories = []

for w in omega:
    x = np.zeros(num_steps)
    y = np.zeros(num_steps)
    theta = np.zeros(num_steps)
    x[0], y[0], theta[0] = x0, y0, theta0

    for i in range(1, num_steps):
        x[i] = update_x(x[i - 1], theta[i - 1], w_x[i])
        y[i] = update_y(y[i - 1], theta[i - 1], w_y[i])
        theta[i] = update_theta(theta[i - 1], w, w_theta[i])
    trajectories.append((x, y))
    theta_trajectories.append(theta)

# Plotting
tau_r = 1 / D_R
t_max = 2 * tau_r
idx_max = np.searchsorted(time, t_max)

plt.figure(figsize=(8, 6))
labels = [
    r"$\omega=0$",
    r"$\omega=\frac{1}{2}\pi$",
    r"$\omega=\pi$",
    r"$\omega=\frac{3}{2}\pi$",
]
for (x, y), label in zip(trajectories, labels):
    plt.plot(x[:idx_max], y[:idx_max], label=label)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title(r"Chiral Active Brownian Particle Trajectories ($0 \leq t \leq 2\tau_R$)")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()

# Plot theta(t) for all omega, for 0 <= t <= 2*tau_r
plt.figure(figsize=(8, 6))
for theta, label in zip(theta_trajectories, labels):
    plt.plot(time[:idx_max], theta[:idx_max], label=label)
plt.xlabel("Time (s)")
plt.ylabel(r"$\phi(t)$ (rad)")
plt.title(r"Orientation $\phi(t)$ for Different $\omega$ ($0 \leq t \leq 2\tau_R$)")
plt.legend()
plt.tight_layout()
plt.savefig("orientations")
plt.show()


def compute_msd(x, y, taus):
    N = len(x)
    msd = np.zeros(len(taus))
    for idx, tau in enumerate(taus):
        disp = (x[tau:] - x[:-tau]) ** 2 + (y[tau:] - y[:-tau]) ** 2
        msd[idx] = np.mean(disp)
    return msd


# Exponentially spaced time delays (taus)
max_tau = num_steps // 10  # up to 1/10th of total time for good statistics
taus = np.unique(np.logspace(0, np.log10(max_tau), num=50, dtype=int))
taus = taus[taus > 0]  # remove zero

plt.figure(figsize=(8, 6))
for (x, y), label in zip(trajectories, labels):
    msd = compute_msd(x, y, taus)
    plt.plot(taus * delta_t, msd, label=label)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time lag $t$ (s)")
plt.ylabel("MSD$(t)$ (m$^2$)")
plt.title(r"Mean Squared Displacement for Different $\omega$")
plt.legend()
plt.tight_layout()
plt.show()

print(f"tau_r = {tau_r:.2f} s")
# tau_r = 6.07s
