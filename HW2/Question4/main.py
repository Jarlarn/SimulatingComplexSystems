import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 200
L = 80
eta = 0.02
v = 0.5
Rf = 2
dt = 1
T_tot = 7500
contrarian_count = N // 2  # N/2 contrarians


# Helper functions
def global_alignment(theta):
    """Global alignment coefficient ψ."""
    return np.abs(np.mean(np.exp(1j * theta)))


def clustering_coefficient(positions, Rf):
    """Global clustering coefficient c."""
    N = positions.shape[0]
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(positions[i] - positions[j]) < Rf:
                count += 1
    return 2 * count / (N * (N - 1))


def get_neighbors(positions, idx, Rf, L):
    """Return indices of neighbors within radius Rf (excluding idx itself)."""
    diffs = positions - positions[idx]
    # Periodic boundary conditions
    diffs = diffs - L * np.round(diffs / L)
    dists = np.linalg.norm(diffs, axis=1)
    neighbors = np.where((dists < Rf) & (dists > 0))[0]
    return neighbors


def circular_average(theta_list):
    """Return the circular average angle."""
    return np.arctan2(np.mean(np.sin(theta_list)), np.mean(np.cos(theta_list)))


def contrarian_average(theta_list):
    """Return the contrarian circular average angle."""
    avg = np.arctan2(np.mean(-np.sin(theta_list)), np.mean(-np.cos(theta_list)))
    return avg


# Initialization
positions = np.random.uniform(0, L, (N, 2))
theta = np.random.uniform(-np.pi, np.pi, N)

contrarian_indices = np.arange(contrarian_count)
normal_indices = np.arange(contrarian_count, N)

# Data storage
save_times = [0, 250, 2500, 5000, 7500]
psi_list = []
c_list = []
theta_snapshots = []
positions_snapshots = []

for t in range(T_tot + 1):
    new_theta = np.copy(theta)
    for i in range(N):
        neighbors = get_neighbors(positions, i, Rf, L)
        if i in contrarian_indices:
            # Contrarian update
            if len(neighbors) == 0:
                new_theta[i] = theta[i] + eta * np.random.normal() * dt
            else:
                neighbor_angles = theta[neighbors]
                avg_angle = contrarian_average(neighbor_angles)
                new_theta[i] = avg_angle + eta * np.random.normal() * dt
        else:
            # Usual Vicsek update
            if len(neighbors) == 0:
                avg_angle = theta[i]
            else:
                neighbor_angles = theta[neighbors]
                avg_angle = circular_average(neighbor_angles)
            new_theta[i] = avg_angle + eta * np.random.normal() * dt

    theta = new_theta

    # Update positions
    positions += v * dt * np.column_stack((np.cos(theta), np.sin(theta)))
    positions = positions % L  # Periodic boundary

    # Save data at specified times
    if t in save_times:
        psi_list.append(global_alignment(theta))
        c_list.append(clustering_coefficient(positions, Rf))
        theta_snapshots.append(np.copy(theta))
        positions_snapshots.append(np.copy(positions))

# D - Plot configurations at requested times, distinguishing populations
plot_times = [0, 2500, 5000, 7500]
for idx, t in enumerate(plot_times):
    # Find index in save_times
    save_idx = save_times.index(t)
    plt.figure(figsize=(8, 8))
    # Normal particles
    plt.quiver(
        positions_snapshots[save_idx][normal_indices, 0],
        positions_snapshots[save_idx][normal_indices, 1],
        np.cos(theta_snapshots[save_idx][normal_indices]),
        np.sin(theta_snapshots[save_idx][normal_indices]),
        color="blue",
        label="Normal",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
        alpha=0.7,
    )
    # Contrarian particles
    plt.quiver(
        positions_snapshots[save_idx][contrarian_indices, 0],
        positions_snapshots[save_idx][contrarian_indices, 1],
        np.cos(theta_snapshots[save_idx][contrarian_indices]),
        np.sin(theta_snapshots[save_idx][contrarian_indices]),
        color="red",
        label="Contrarian",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
        alpha=0.7,
    )
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.title(f"Configuration (N/2 contrarians) at t = {t} Δt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"vicsek_config_N2contrarians_t{t}.png", dpi=300)
    plt.show()

# E - Plot global alignment coefficient ψ and clustering coefficient c
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(save_times, psi_list, marker="o")
plt.xlabel("Time step")
plt.ylabel("Global alignment ψ")
plt.title("Global Alignment Coefficient (N/2 contrarians)")

plt.subplot(1, 2, 2)
plt.plot(save_times, c_list, marker="s")
plt.xlabel("Time step")
plt.ylabel("Clustering coefficient c")
plt.title("Global Clustering Coefficient (N/2 contrarians)")
plt.tight_layout()
plt.savefig("vicsek_alignment_clustering_N2contrarians.png", dpi=300)
plt.show()
