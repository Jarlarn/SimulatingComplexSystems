import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


output_dir = "Question1"
os.makedirs(output_dir, exist_ok=True)


# Simulation parameters
N = 100
SIGMA = 1.0
MASS = 1.0
EPSILON = 1.0
TIME_STEP = 0.001
VELOCITY = 1.0
L = 10.0 * SIGMA
# L = 16.0 * SIGMA
box_min = -L / 2
box_max = L / 2

# Set up initial positions on a LxL lattice
num_particles_x = int(np.sqrt(N))
num_particles_y = num_particles_x
assert num_particles_x * num_particles_y == N, "N must be a perfect square."
spacing = L / (num_particles_x - 1)
positions = np.zeros((N, 2))
for row in range(num_particles_x):
    for column in range(num_particles_y):
        idx = row * num_particles_y + column
        positions[idx, 0] = box_min + column * spacing
        positions[idx, 1] = box_min + row * spacing

# Assign random velocities of magnitude VELOCITY
angles = np.random.uniform(0, 2 * np.pi, N)
velocities = np.zeros((N, 2))
velocities[:, 0] = VELOCITY * np.cos(angles)
velocities[:, 1] = VELOCITY * np.sin(angles)


def lennard_jones_force(pos1, pos2):
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    r2 = dx**2 + dy**2
    r = np.sqrt(r2)
    if r < 1e-12:
        return np.array([0.0, 0.0])  # Avoid self-interaction
    # Lennard-Jones force magnitude
    F_mag = 24 * EPSILON / r * (2 * (SIGMA / r) ** 12 - (SIGMA / r) ** 6)
    Fx = F_mag * dx / r
    Fy = F_mag * dy / r
    return np.array([Fx, Fy])


def reflect_boundary(pos, vel):
    # Reflecting boundaries for a box from box_min to box_max
    for i in range(2):
        if pos[i] < box_min:
            delta = box_min - pos[i]
            pos[i] = box_min + delta
            vel[i] = -vel[i]
        elif pos[i] > box_max:
            delta = pos[i] - box_max
            pos[i] = box_max - delta
            vel[i] = -vel[i]
    return pos, vel


def compute_forces(positions):
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(i + 1, N):
            f = lennard_jones_force(positions[i], positions[j])
            forces[i] += f
            forces[j] -= f  # Newton's third law
    return forces


# Plot initial configuration
plt.figure(figsize=(6, 6))
ax = plt.gca()
for i in range(N):
    circle = patches.Circle(
        (positions[i, 0], positions[i, 1]),
        SIGMA / 2,
        edgecolor="orange",
        facecolor="orange",
        linewidth=1.5,
        alpha=0.5,  # Slightly transparent
    )
    ax.add_patch(circle)
plt.scatter(positions[:, 0], positions[:, 1], s=30, color="blue", zorder=2)
plt.title("Initial Configuration at t = 0")
plt.xlim(box_min, box_max)
plt.xlim(box_min, box_max)
ax.set_xlim(box_min, box_max)
ax.set_ylim(box_min, box_max)
ax.set_aspect("equal")
plt.locator_params(axis="both", nbins=L)
plt.xlabel("x (sigma)")
plt.ylabel("y (sigma)")
plt.grid(True, which="both", linestyle="-", color="gray", alpha=0.3)
plt.savefig(f"{output_dir}/initial_configuration.png", dpi=600)
plt.show()


# Leapfrog integration
num_steps = 10000
positions = positions.copy()
velocities = velocities.copy()
trajectories = np.zeros((N, num_steps, 2))
trajectories[:, 0, :] = positions


for step in range(1, num_steps):
    forces = compute_forces(positions)
    # Velocity Verlet position update
    positions += velocities * TIME_STEP + 0.5 * forces / MASS * TIME_STEP**2
    # Compute new forces at updated positions
    new_forces = compute_forces(positions)
    # Velocity Verlet velocity update
    velocities += 0.5 * (forces + new_forces) / MASS * TIME_STEP
    # Reflections
    for i in range(N):
        positions[i], velocities[i] = reflect_boundary(positions[i], velocities[i])
    # Store trajectory
    trajectories[:, step, :] = positions
    if step % 100 == 0:
        percent = (step / num_steps) * 100
        print(f"Step {step}/{num_steps} ({percent:.1f}%) complete")

# Find central particle (closest to box center)
center = np.array([0.0, 0.0])
dists = np.linalg.norm(trajectories[:, 0, :] - center, axis=1)
central_idx = np.argmin(dists)
central_traj = trajectories[central_idx]

# Compute MSD for central particle
S = num_steps
msd = np.zeros(S)
for n in range(S):
    if n == 0:
        msd[n] = 0
    else:
        diffs = central_traj[n:] - central_traj[:-n]
        msd[n] = np.mean(np.sum(diffs**2, axis=1))


# Plot (A) final configuration
plt.figure(figsize=(6, 6))
ax = plt.gca()
for i in range(N):
    circle = patches.Circle(
        (positions[i, 0], positions[i, 1]),
        SIGMA / 2,
        edgecolor="orange",
        facecolor="orange",
        linewidth=1.5,
        alpha=0.5,
    )
    ax.add_patch(circle)
plt.scatter(positions[:, 0], positions[:, 1], s=30, color="blue", zorder=2)
plt.title("Final Configuration at t = Ttot")
plt.xlim(box_min, box_max)
plt.xlim(box_min, box_max)
ax.set_xlim(box_min, box_max)
ax.set_ylim(box_min, box_max)
plt.locator_params(axis="both", nbins=L)
ax.set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(f"{output_dir}/final_configuration.png", dpi=600)
plt.show()


# Plot (B) trajectory and MSD of central particle
plt.figure(figsize=(6, 6))
plt.plot(central_traj[:, 0], central_traj[:, 1], label="Trajectory")
plt.title("Trajectory of Central Particle")
plt.xlim(box_min, box_max)
plt.xlim(box_min, box_max)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig(f"{output_dir}/trajectory.png", dpi=600)
plt.show()

plt.figure()
plt.plot(np.arange(S) * TIME_STEP, msd)
plt.title("Mean Square Displacement (MSD) of Central Particle")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.savefig(f"{output_dir}/msd.png", dpi=600)
plt.show()
