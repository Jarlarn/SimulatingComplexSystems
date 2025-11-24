import numpy as np
import matplotlib.pyplot as plt

# Parameters
tau = 1
V0 = 0.1
V_inf = 0.01
Ic = 0.1
I0 = 1.0
r0 = 0.3
delta_t = 0.05
T_tot = 1800
L = 3.0
N = 10
delta_pos = 5 * delta_t
delta_neg = -5 * delta_t
Nneg = 1  # Number of robots with negative delay
np.random.seed(42)

# Initial positions and orientations
x = np.random.uniform(0, L, N)
y = np.random.uniform(0, L, N)
theta = np.random.uniform(0, 2 * np.pi, N)

# Assign delays
delays = np.ones(N) * delta_pos
delays[:Nneg] = delta_neg

# Store trajectories
x_traj = np.zeros((N, int(T_tot / delta_t) + 1))
y_traj = np.zeros((N, int(T_tot / delta_t) + 1))
theta_traj = np.zeros((N, int(T_tot / delta_t) + 1))
x_traj[:, 0] = x
y_traj[:, 0] = y
theta_traj[:, 0] = theta

# For 2D histogram
arena_bins = 60
arena_hist = np.zeros((arena_bins, arena_bins))


def generate_noise():
    return np.random.normal(loc=0, scale=1)


def periodic_boundary(val, L):
    return val % L


def intensity(x_all, y_all, n):
    dx = x_all[n] - x_all
    dy = y_all[n] - y_all
    dx[n] = 0
    dy[n] = 0
    dist2 = dx**2 + dy**2
    dist2[n] = 1  # avoid self-contribution
    I = np.sum(I0 * np.exp(-dist2 / r0**2)) - I0  # subtract self
    return I


def v_of_I(I):
    return V_inf + (V0 - V_inf) * np.exp(-I / Ic)


# Store for plotting negative delay robot
neg_idx = 0  # index of robot with negative delay
neg_x = []
neg_y = []

# Collect all positions for all robots at all time steps
all_x = []
all_y = []

# Main simulation loop
for t_idx in range(1, int(T_tot / delta_t) + 1):
    t = t_idx * delta_t
    x_prev = x.copy()
    y_prev = y.copy()
    theta_prev = theta.copy()

    # Compute intensities with delays
    I_all = np.zeros(N)
    for n in range(N):
        delay_steps = int(np.round(delays[n] / delta_t))
        idx = max(0, t_idx - delay_steps)
        # Use previous positions for delay
        x_delay = x_traj[:, idx] if idx < x_traj.shape[1] else x_traj[:, -1]
        y_delay = y_traj[:, idx] if idx < y_traj.shape[1] else y_traj[:, -1]
        I_all[n] = intensity(x_delay, y_delay, n)

    # Update positions and orientations
    for n in range(N):
        v = v_of_I(I_all[n])
        theta_noise = generate_noise()
        theta[n] = theta_prev[n] + np.sqrt(2 / tau * delta_t) * theta_noise
        x[n] = periodic_boundary(x_prev[n] + v * np.cos(theta[n]) * delta_t, L)
        y[n] = periodic_boundary(y_prev[n] + v * np.sin(theta[n]) * delta_t, L)

    x_traj[:, t_idx] = x
    y_traj[:, t_idx] = y
    theta_traj[:, t_idx] = theta

    # For 2D histogram: collect all robots' positions
    all_x.extend(x)
    all_y.extend(y)

    # For 2D histogram of negative delay robot
    neg_x.append(x[neg_idx])
    neg_y.append(y[neg_idx])

# (A) Plot initial and final configuration
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(x_traj[:, 0], y_traj[:, 0], c="blue", label="Initial")
plt.title("Initial configuration")
plt.xlim(0, L)
plt.ylim(0, L)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x_traj[:, -1], y_traj[:, -1], c="red", label="Final")
plt.title("Final configuration")
plt.xlim(0, L)
plt.ylim(0, L)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.tight_layout()
plt.savefig("initial_final_configuration.png")  # Save plot
plt.show()

# Plot trajectory of negative delay robot
plt.figure(figsize=(5, 5))
plt.plot(neg_x, neg_y, label="Negative delay robot")
plt.scatter([neg_x[0]], [neg_y[0]], c="green", label="Start")
plt.scatter([neg_x[-1]], [neg_y[-1]], c="red", label="End")
plt.xlim(0, L)
plt.ylim(0, L)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectory of negative delay robot")
plt.legend()
plt.savefig("negative_delay_trajectory.png")  # Save plot
plt.show()

# (B) 2D histogram of area explored by all robots
hist, xedges, yedges = np.histogram2d(
    all_x, all_y, bins=arena_bins, range=[[0, L], [0, L]]
)
plt.figure(figsize=(6, 5))
plt.imshow(hist.T, origin="lower", extent=[0, L, 0, L], aspect="auto", cmap="viridis")
plt.colorbar(label="Visits")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("2D histogram: Area explored by all robots")
plt.savefig("all_robots_2dhist.png")  # Save plot
plt.show()
