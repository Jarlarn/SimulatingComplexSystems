import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# Create output directory for figures
os.makedirs("figures", exist_ok=True)


class LightSensitiveRobot:
    def __init__(self, x, y, phi, delay, L, I_c):
        self.x = x
        self.y = y
        self.phi = phi
        self.delay = delay
        self.L = L
        self.I_c = I_c
        self.trajectory_x = [x]
        self.trajectory_y = [y]
        # Store unwrapped positions for plotting
        self.unwrapped_x = [x]
        self.unwrapped_y = [y]

    def compute_intensity(self, robots):
        """Compute light intensity from other robots"""
        I_total = 0
        for robot in robots:
            if robot is not self:
                dx = self.x - robot.x
                dy = self.y - robot.y
                r_squared = dx**2 + dy**2
                if r_squared > 0:
                    I_total += robot.I_c * np.exp(-r_squared / robot.L**2)
        return I_total

    def compute_speed(self, I, V_oo, V_0, I_c):
        """Compute speed based on light intensity"""
        return V_oo + (V_0 - V_oo) * np.exp(-I / I_c)

    def update(self, robots, dt, V_oo, V_0, noise_scale=1.0):
        """Update robot position"""
        I = self.compute_intensity(robots)
        v = self.compute_speed(I * (1 + self.delay), V_oo, V_0, self.I_c)

        # Store old position for unwrapped tracking
        old_x = self.x
        old_y = self.y

        # Update position
        self.x += v * np.cos(self.phi) * dt
        self.y += v * np.sin(self.phi) * dt

        # Update angle with noise
        noise = np.random.normal(0, noise_scale)
        self.phi += np.sqrt(2 / np.pi) * noise * np.sqrt(dt)

        # Store trajectory (wrapped positions)
        self.trajectory_x.append(self.x)
        self.trajectory_y.append(self.y)

        # Store unwrapped trajectory for plotting
        dx = self.x - old_x
        dy = self.y - old_y
        self.unwrapped_x.append(self.unwrapped_x[-1] + dx)
        self.unwrapped_y.append(self.unwrapped_y[-1] + dy)

    def apply_periodic_boundary(self, L_arena):
        """Apply periodic boundary conditions"""
        self.x = self.x % L_arena
        self.y = self.y % L_arena


def run_simulation(
    N=10, N_pos=1, L_arena=4.0, T_cat=1800, dt=0.1, V_oo=0.2, V_0=0.01, I_c=1.0, L=1.0
):
    """Run the simulation with specified parameters"""

    # Initialize robots
    robots = []
    for i in range(N):
        x = np.random.uniform(0, L_arena)
        y = np.random.uniform(0, L_arena)
        phi = np.random.uniform(0, 2 * np.pi)

        # N_pos robots have positive delay, rest have negative
        if i < N_pos:
            delay = 1.0  # positive delay
        else:
            delay = -1.0  # negative delay

        robots.append(LightSensitiveRobot(x, y, phi, delay, L, I_c))

    # Run simulation
    n_steps = int(T_cat / dt)

    # Create 2D histogram for exploration
    n_bins = 20
    exploration_histogram = np.zeros((n_bins, n_bins))

    for step in range(n_steps):
        # Update all robots
        for robot in robots:
            robot.update(robots, dt, V_oo, V_0)
            robot.apply_periodic_boundary(L_arena)

            # Update exploration histogram
            bin_x = int((robot.x / L_arena) * n_bins)
            bin_y = int((robot.y / L_arena) * n_bins)
            bin_x = min(bin_x, n_bins - 1)
            bin_y = min(bin_y, n_bins - 1)
            exploration_histogram[bin_y, bin_x] += dt

    return robots, exploration_histogram


def plot_all_trajectories(robots, N_pos, L_arena, filename="all_trajectories.png"):
    """Plot trajectories of all robots"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot majority type (negative delay) in light blue
    for i in range(N_pos, len(robots)):
        robot = robots[i]
        ax.plot(
            robot.unwrapped_x,
            robot.unwrapped_y,
            color="cornflowerblue",
            linewidth=0.5,
            alpha=0.6,
            label="Majority Type" if i == N_pos else "",
        )

    # Plot minority type (positive delay) in red
    for i in range(N_pos):
        robot = robots[i]
        ax.plot(
            robot.unwrapped_x,
            robot.unwrapped_y,
            "r-",
            linewidth=1,
            label="Minority Type" if i == 0 else "",
        )
        # Mark start and end
        ax.plot(
            robot.unwrapped_x[0],
            robot.unwrapped_y[0],
            "go",
            markersize=8,
            label="Start" if i == 0 else "",
            zorder=5,
        )
        ax.plot(
            robot.unwrapped_x[-1],
            robot.unwrapped_y[-1],
            "kx",
            markersize=10,
            markeredgewidth=2,
            label="End" if i == 0 else "",
            zorder=5,
        )

    ax.set_xlim(0, L_arena)
    ax.set_ylim(0, L_arena)
    ax.set_aspect("equal")
    ax.set_title(f"Trajectories (Part 2: {N_pos} Pos, {len(robots)-N_pos} Neg)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(f"figures/{filename}", dpi=300, bbox_inches="tight")
    print(f"Saved figure: figures/{filename}")
    plt.close()


def plot_initial_final_config(robots, N_pos, L_arena, filename="config.png"):
    """Plot (D): Initial and final configuration"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Initial configuration
    ax1.set_xlim(0, L_arena)
    ax1.set_ylim(0, L_arena)
    ax1.set_aspect("equal")
    ax1.set_title("Initial configuration")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.grid(False)

    for i, robot in enumerate(robots):
        ax1.plot(
            robot.trajectory_x[0],
            robot.trajectory_y[0],
            "bo",
            markersize=8,
            label="Initial" if i == 0 else "",
        )
    ax1.legend()

    # Final configuration
    ax2.set_xlim(0, L_arena)
    ax2.set_ylim(0, L_arena)
    ax2.set_aspect("equal")
    ax2.set_title("Final configuration")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.grid(False)

    for i, robot in enumerate(robots):
        ax2.plot(robot.x, robot.y, "ro", markersize=8, label="Final" if i == 0 else "")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"figures/{filename}", dpi=300, bbox_inches="tight")
    print(f"Saved figure: figures/{filename}")
    plt.close()


def plot_exploration_histogram(
    exploration_histogram, L_arena, filename="exploration.png"
):
    """Plot (E): 2D histogram of exploration"""
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        exploration_histogram,
        origin="lower",
        extent=[0, L_arena, 0, L_arena],
        cmap="viridis",
        aspect="equal",
    )

    ax.set_title("2D histogram: Area explored by all robots")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Visits")

    plt.tight_layout()
    plt.savefig(f"figures/{filename}", dpi=300, bbox_inches="tight")
    print(f"Saved figure: figures/{filename}")
    plt.close()


def main():
    print("Running simulation for Question 2...")

    # Parameters
    N = 10
    N_pos = 1
    L_arena = 3.0
    T_cat = 1800
    dt = 0.1
    V_oo = 0.2
    V_0 = 0.01
    I_c = 1.0
    L = 1.0

    # Run simulation
    robots, exploration_histogram = run_simulation(
        N=N,
        N_pos=N_pos,
        L_arena=L_arena,
        T_cat=T_cat,
        dt=dt,
        V_oo=V_oo,
        V_0=V_0,
        I_c=I_c,
        L=L,
    )

    # Plot all trajectories
    plot_all_trajectories(robots, N_pos, L_arena, "all_trajectories.png")

    # Plot (D): Initial and final configuration
    plot_initial_final_config(robots, N_pos, L_arena, "initial_final_config.png")

    # Plot (E): Exploration histogram
    plot_exploration_histogram(
        exploration_histogram, L_arena, "exploration_histogram.png"
    )

    print("\nSimulation complete!")
    print(f"Total simulation time: {T_cat}s")
    print(f"Number of robots: {N} ({N_pos} with positive delay)")
    print(f"Arena size: {L_arena}x{L_arena}")


if __name__ == "__main__":
    main()
